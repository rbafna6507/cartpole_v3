"""Dependency-free source adapters for portable reset/act controllers.

Mappings are an explicit allowlist. Values and provenance are never rewritten.
The selected mapping is pinned until reset; partial/ambiguous input fails closed.
"""
from .contracts import ValidationError, validate_observation
from .policies.rig_observation import observation_contract, rig_channels, simulation_channels, select_rig_observation


class ObservationAdapter:
    def __init__(self, *, angle_source='filtered', sources=('simulation', 'magnetic-step')):
        self.contract = observation_contract(angle_source=angle_source)
        mappings = {'simulation': simulation_channels(angle_source=angle_source),
                    'magnetic-step': rig_channels(angle_source=angle_source)}
        if not sources or len(set(sources)) != len(sources) or any(s not in mappings for s in sources):
            raise ValidationError('choose explicit supported sources without duplicates')
        self.mappings = {s: mappings[s] for s in sources}
        self.session = None
        self.source = None

    def reset(self, session):
        self.session, self.source = session, None

    def select(self, observation):
        if self.session is None:
            raise ValidationError('reset(Session) before selection')
        validate_observation(self.session, observation)
        available = {s.channel for s in observation.samples}
        matches = [name for name, channels in self.mappings.items() if set(channels) <= available]
        if len(matches) != 1:
            raise ValidationError('observation needs exactly one complete declared source mapping')
        source = matches[0]
        if self.source is not None and self.source != source:
            raise ValidationError('observation source changed within session')
        samples = select_rig_observation(observation, self.mappings[source], contract=self.contract)
        self.source = source
        return samples


def load_local_policy(path):
    """Import a trusted local file. This helper is NOT a submission sandbox."""
    import importlib.util
    from pathlib import Path
    if Path(path).suffix == '.json':
        return PortableActor(path)
    spec = importlib.util.spec_from_file_location('local_controller', path)
    if spec is None or spec.loader is None:
        raise ValueError('expected a local Python controller file')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_policy()


def run_profile_policy(policy, output, *, profile_id=None, seconds=5., seed=0, state=None, policy_sha256=None):
    """Run one controller against the shared governed profile and record its input.

    Offline only. Runtime is not deadline tested. Replay is open-loop; simulation
    logs preserve all calls, including repeated/latest telemetry snapshots.
    """
    from dataclasses import asdict
    from pathlib import Path
    from .contracts import Acceleration
    from .recording import canonical_json, write_recording
    from .sim.profiles import PROFILE_ID, load_profile, make_profile_env
    profile_id = profile_id or PROFILE_ID
    profile = load_profile(profile_id)
    contract = getattr(policy, 'observation_contract', None)
    if contract is not None and contract['angle_source'] != profile['experiment']['observation_config']['angle_source']:
        raise ValidationError('policy angle source differs from named simulation profile')
    env = make_profile_env(profile_id, seconds=seconds)
    output = Path(output)
    rows, traces = [], []
    try:
        output.mkdir(parents=True, exist_ok=False)
        env.reset(seed=seed, options=None if state is None else {'state':state})
        initial_truth = asdict(env.ground_truth)
        policy.reset(env.session)
        while not env.terminated and not env.truncated:
            obs = env.sdk_observation
            rows.append(obs)
            intent = policy.act(obs)
            if intent is not None and not isinstance(intent, Acceleration):
                raise ValidationError('policy must return Acceleration or None')
            value = None if intent is None else intent.acceleration_m_s2
            _, reward, _, _, info = env.step(None if value is None else [value])
            traces.append(dict(observation_sequence=obs.sequence, requested_acceleration_m_s2=value,
                               reward=reward, fault_reason=info.get('fault_reason'),
                               executor=asdict(env.diagnostics)))
        write_recording(output/'observations.jsonl', env.session, rows)
        (output/'trace.jsonl').write_text(''.join(canonical_json(row)+'\n' for row in traces))
        metadata = dict(schema='cartpole-starter-run-v1', profile_id=profile_id,
            profile_sha256=profile['content_sha256'], simulation_config=asdict(env.config),
            observation_config=asdict(env.observation_config), seed=seed, steps=len(rows),
            initial_truth=initial_truth, policy_sha256=policy_sha256,
            policy_type=type(policy).__module__ + '.' + type(policy).__qualname__,
            terminated=env.terminated, truncated=env.truncated, fault_reason=env.fault_reason,
            hardware_accepted=False)
        (output/'run.json').write_text(canonical_json(metadata)+'\n')
        return metadata
    finally:
        env.close()


class PortableActor:
    """Same numeric actor bytes, explicit simulation/rig adapters selected per reset."""
    def __init__(self, path):
        from .policies.neural import load_neural_policy
        from .recording import strict_json_loads
        from .policies.neural import MAX_BYTES
        from pathlib import Path
        with Path(path).open('rb') as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValidationError('numeric actor exceeds size limit')
        actor = strict_json_loads(raw.decode('utf-8'))
        # Numeric schema/provenance validation remains the standard loader's job.
        angle_source = actor.get('angle_source')
        if angle_source not in ('raw', 'filtered'):
            raise ValidationError('portable starter supports raw or filtered encoder actors')
        self.policies = {name: load_neural_policy(path, channels=channels) for name, channels in
            {'simulation':simulation_channels(angle_source=angle_source),
             'magnetic-step':rig_channels(angle_source=angle_source)}.items()}
        self.observation_contract = self.policies['simulation'].observation_contract
        self.adapter = ObservationAdapter(angle_source=angle_source)
        self.fault_reason = None

    def reset(self, session):
        self.adapter.reset(session)
        self.fault_reason = None
        for policy in self.policies.values():
            policy.reset(session)

    def act(self, observation):
        if self.fault_reason is not None:
            return None
        try:
            self.adapter.select(observation)
            policy = self.policies[self.adapter.source]
            result = policy.act(observation)
            self.fault_reason = policy.fault_reason
            return result
        except ValidationError as exc:
            self.fault_reason = str(exc)
            return None
