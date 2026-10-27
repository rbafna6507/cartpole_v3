"""Closed-loop local runner and reproducible output bundle (requires sim extra)."""
from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import hashlib
from importlib.metadata import version
import platform

from .contracts import Acceleration, ActionRequest, Observation, Policy, Session, ValidationError
from .control.policy import ReferencePolicy
from .recording import canonical_json, write_recording
from .sim import (GroundTruth, MujocoBackend, SensorConfig, SimulationConfig, SimulationDiagnostics)


@dataclass(frozen=True, slots=True)
class SimulationStep:
    observation_sequence: int
    action: ActionRequest | None
    next_observation_sequence: int
    truth: GroundTruth
    diagnostics: SimulationDiagnostics
    controller: dict | None
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class SimulationRun:
    session: Session
    observations: tuple[Observation, ...]
    steps: tuple[SimulationStep, ...]
    config: SimulationConfig
    sensors: SensorConfig
    seed: int
    initial_truth: GroundTruth
    policy: dict
    environment: dict
    model_xml: str
    end_reason: str
    fault_reason: str | None


class ZeroPolicy:
    """Local physics demonstration: request zero cart acceleration every step."""
    def reset(self, session):
        pass

    def act(self, observation):
        return Acceleration(0.)


def run_simulation(policy: Policy, *, config: SimulationConfig | None = None,
                   sensors: SensorConfig | None = None, seed: int = 0,
                   state: tuple[float, float, float, float] | None = None,
                   session_id: str = 'simulation') -> SimulationRun:
    """Run until rail/horizon/reference fault; no wall-clock scheduling or devices.

    Intent leases last one control interval. None issues no new lease. A latched
    ReferencePolicy fault stops the rollout before the next physics transition.
    """
    config = config if config is not None else SimulationConfig()
    sensors = sensors if sensors is not None else SensorConfig()
    if not isinstance(config, SimulationConfig) or not isinstance(sensors, SensorConfig):
        raise ValidationError('expected SimulationConfig and SensorConfig')
    interval = config.physics_dt_ns * config.control_steps
    if isinstance(policy, ReferencePolicy) and interval != policy.period_ns:
        raise ValidationError('reference policy requires a 1 ms control interval')
    if interval > config.max_action_ttl_ns:
        raise ValidationError('control interval exceeds the configured action lifetime limit')
    backend = MujocoBackend(config, sensors)
    try:
        observation = backend.reset(seed=seed, state=state, session_id=session_id)
        initial_truth = backend.ground_truth
        policy.reset(backend.session)
        policy_metadata = {'type': f'{type(policy).__module__}.{type(policy).__qualname__}'}
        if isinstance(policy, ReferencePolicy):
            policy_metadata.update(start_mode=policy.start_mode, max_age_ns=policy.max_age_ns,
                angle=asdict(policy.angle), position=asdict(policy.position),
                parameters=asdict(policy.controller.parameters))
        observations, steps = [observation], []
        action_sequence = 0
        reason, fault = 'horizon', None
        while not backend.terminated and not backend.truncated:
            intent = policy.act(observation)
            if isinstance(policy, ReferencePolicy) and policy.fault_reason:
                reason, fault = 'policy_fault', policy.fault_reason
                break
            if intent is not None and not isinstance(intent, Acceleration):
                raise ValidationError('policy must return Acceleration or None')
            request = None
            if intent is not None:
                request = ActionRequest(backend.session.session_id, backend.session.clock_id,
                    action_sequence, observation.sequence, observation.available_ns,
                    observation.available_ns + interval, intent.acceleration_m_s2)
                action_sequence += 1
            following = backend.step(request)
            controller = asdict(policy.last_output) if isinstance(policy, ReferencePolicy) else None
            steps.append(SimulationStep(observation.sequence, request, following.sequence,
                backend.ground_truth, backend.diagnostics, controller, backend.terminated, backend.truncated))
            observations.append(following)
            observation = following
        if backend.terminated:
            reason = 'rail_limit'
        return SimulationRun(backend.session, tuple(observations), tuple(steps), config, sensors,
                             seed, initial_truth, policy_metadata,
                             {name:version(name) for name in ('cartpole-sdk','mujoco','numpy','gymnasium')} |
                             {'python':platform.python_version()}, backend.model_xml, reason, fault)
    finally:
        backend.close()


def write_simulation(path: str | Path, run: SimulationRun) -> None:
    """Publish a new output directory; remove only our incomplete bundle on failure."""
    path = Path(path)
    path.mkdir()
    try:
        metadata = {'version':1, 'mode':'closed_loop_simulation',
            'model_status':'provisional surrogate; not identified or validated against this rig',
            'config':asdict(run.config), 'sensors':asdict(run.sensors), 'seed':run.seed,
            'session':asdict(run.session), 'initial_truth':asdict(run.initial_truth),
            'policy':run.policy, 'environment':run.environment,
            'model_sha256':hashlib.sha256(run.model_xml.encode()).hexdigest(), 'end_reason':run.end_reason, 'fault_reason':run.fault_reason,
            'steps':len(run.steps),
            'diagnostics_timing':'force and acceleration describe the final physics substep; truth is post-step',
            'expiry_behavior':'zero requested acceleration after expiry; this may require nonzero force'}
        (path/'model.xml').write_text(run.model_xml, encoding='utf-8')
        (path/'run.json').write_text(canonical_json(metadata)+'\n', encoding='utf-8')
        write_recording(path/'observations.jsonl',run.session,run.observations)
        with (path/'trace.jsonl').open('x', encoding='utf-8') as stream:
            stream.write(canonical_json({'type':'header', 'version':1, 'mode':'closed_loop_simulation',
                                        'session':asdict(run.session)})+'\n')
            for step in run.steps:
                stream.write(canonical_json({'type':'step', **asdict(step)})+'\n')
            stream.write(canonical_json({'type':'footer','count':len(run.steps),
                                        'end_reason':run.end_reason,'fault_reason':run.fault_reason})+'\n')
    except BaseException:
        shutil.rmtree(path)
        raise
