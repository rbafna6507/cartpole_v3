"""Finite offline engineering stresses for an already loaded, frozen policy.

The caller validates checkpoint loading. This module pins that caller's binding,
the evaluation runtime and actual in-memory weights; no hardware is accessed.
Passing these scenarios is not an identified uncertainty or hardware guarantee.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np

from .diagnostics import preserved_rng
from .model_experiment import digest, validate_experiment
from .model_search import _seeds
from .ppo import TrainingEnv, _write_new, experiment_identity, provenance


def source_identity():
    return dict(runtime=provenance(), evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        rng_helper_sha256=sha256(Path(__file__).with_name('diagnostics.py').read_bytes()).hexdigest())


def _weights_digest(state):
    result = sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def scenarios(experiment):
    """Point bounds use the training randomizer, retaining all other inputs."""
    obs = experiment['observation_config']
    def point(name, *, plant=None, observation=None):
        p = dict(effective_length_scale=1., angular_damping_scale=1., coulomb_friction_scale=1.)
        p.update(plant or {})
        o = {key: obs[key] for key in ('angle_noise_std', 'telemetry_delay_steps', 'command_delay_steps')}
        o.update(observation or {})
        return dict(name=name, expected='task_success', angle_offset_rad=0., observation_override={},
            randomization=dict(schema='cartpole-training-randomization-v1',
                rationale='Fixed engineering stress point, not measured uncertainty or a fitted posterior.',
                plant={k:[v,v] for k,v in p.items()}, observation={k:[v,v] for k,v in o.items()}))
    rows = [dict(name='nominal', expected='task_success', angle_offset_rad=0.,
                 observation_override={}, randomization=None),
        point('length-minus5', plant=dict(effective_length_scale=.95)),
        point('length-plus5', plant=dict(effective_length_scale=1.05)),
        point('friction-minus20', plant=dict(coulomb_friction_scale=.8)),
        point('friction-plus20', plant=dict(coulomb_friction_scale=1.2)),
        point('encoder-noise-003', observation=dict(angle_noise_std=.003)),
        point('delay-8-2', observation=dict(telemetry_delay_steps=8, command_delay_steps=2))]
    for sign, name in ((-1, 'angle-minus1deg'), (1, 'angle-plus1deg')):
        rows.append(dict(rows[0], name=name, angle_offset_rad=sign*math.pi/180))
    # Fault conditions deliberately exceed the learning randomizer's renewal
    # budget. Declare them in a copied observation config, never alter the plant
    # admission or weaken randomization validation.
    rows.extend([
        dict(name='delay-9-2-fault', expected='fault_response', randomization=None,
            angle_offset_rad=0., observation_override=dict(telemetry_delay_steps=9, command_delay_steps=2)),
        dict(name='telemetry-invalid-fault', expected='fault_response', randomization=None,
            angle_offset_rad=0., observation_override=dict(dropout_probability=1.))])
    return rows


def _actor_observation(observation, offset):
    result = {key: np.array(value, copy=True) for key, value in observation.items()}
    if offset and result['valid'][2] and result['valid'][3]:
        angle = math.atan2(float(result['state'][2]), float(result['state'][3]))+offset
        result['state'][2:4] = math.sin(angle), math.cos(angle)
    return result


def _action(model, observation):
    action, _ = model.predict(observation, deterministic=True)
    action = np.asarray(action)
    if action.shape != (1,) or not np.all(np.isfinite(action)) or abs(float(action[0])) > 1:
        raise ValueError('frozen policy must return one finite normalized action in [-1,1]')
    return action


def _episode(model, env, *, seed, start, scenario, root):
    observation, info = env.reset(options=dict(episode_seed=seed, start=start))
    relative = f"traces/{scenario['name']}-{start}-{seed}.jsonl"
    rows, raw_actions, sensitivities = [], [], []
    reset_ns = env.core.ground_truth.time_ns
    with (root/relative).open('x') as stream:
        while True:
            actor_input = _actor_observation(observation, scenario['angle_offset_rad'])
            # Counterfactual command on exactly the same observation is measured
            # only for injected angle offset; it is never applied to the plant.
            reference = (_action(model, observation) if scenario['angle_offset_rad'] else None)
            action = _action(model, actor_input)
            proposed = float(action[0])*env.action_scale
            unperturbed = proposed if reference is None else float(reference[0])*env.action_scale
            previous_ns = env.core.ground_truth.time_ns
            observed_sequence = env.core.sdk_observation.sequence
            observed_ns = env.core.sdk_observation.available_ns
            observation, _, terminated, truncated, info = env.step(action)
            truth = asdict(env.core.ground_truth)
            row = dict(decision_index=len(rows), simulation_time_ns=truth['time_ns'],
                elapsed_since_reset_s=(truth['time_ns']-reset_ns)/1e9,
                observation_sequence=observed_sequence, observation_available_ns=observed_ns,
                telemetry_transport_age_before_ns=previous_ns-observed_ns,
                actor_observation={k:v.tolist() for k,v in actor_input.items()},
                ground_truth_after=truth, requested_raw_action_m_s2=proposed,
                unperturbed_raw_action_m_s2=unperturbed,
                actual_executor_raw_m_s2=env.core.diagnostics.raw_acceleration_m_s2,
                executed_acceleration_m_s2=env.core.diagnostics.executed_acceleration_m_s2,
                physical_time_advanced=truth['time_ns']>previous_ns,
                fault_reason=env.fault, terminal=bool(terminated or truncated))
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            rows.append(row); raw_actions.append(proposed); sensitivities.append(proposed-unperturbed)
            if terminated or truncated:
                break
    result = env.result(hold_seconds=5.)
    fault = result['fault_reason']
    # This checks terminal detection/no continuation, not disabled cart travel or
    # the real host runner's STOP protocol. Both remain separate acceptance work.
    expected_fault = ((scenario['name']=='delay-9-2-fault' and fault=='action lease expired') or
        (scenario['name']=='telemetry-invalid-fault' and fault is not None and
         any(word in fault.lower() for word in ('invalid', 'unusable', 'valid', 'lease'))))
    tail = [r for r in rows if r['elapsed_since_reset_s'] > env.core.config.max_steps/100.-5.]
    tail_x = np.array([r['ground_truth_after']['x'] for r in tail])
    raw = np.array(raw_actions); delta = np.diff(raw); sensitivity = np.array(sensitivities)
    result.update(scenario=scenario['name'], expected=scenario['expected'],
        fault_response_pass=bool(expected_fault) if scenario['expected']=='fault_response' else None,
        task_success=bool(result['hold_success']) if scenario['expected']=='task_success' else None,
        trace_path=relative, trace_sha256=sha256((root/relative).read_bytes()).hexdigest(),
        reset_warmup_ns=reset_ns, elapsed_physical_s=(env.core.ground_truth.time_ns-reset_ns)/1e9,
        final_window_x_rms_m=(float(np.sqrt(np.mean(tail_x**2))) if result['final_window_complete'] else None),
        whole_episode_centering_margin_m=.12-result['max_abs_x_m'],
        raw_action_rms_m_s2=float(np.sqrt(np.mean(raw**2))),
        raw_action_change_rms_m_s2=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
        raw_action_at_limit_fraction=float(np.mean(np.abs(raw)>=.999*env.action_scale)),
        angle_offset_command_delta_rms_m_s2=float(np.sqrt(np.mean(sensitivity**2))),
        angle_offset_command_delta_max_m_s2=float(np.max(np.abs(sensitivity))))
    return result


def run_suite(model, experiment, output, *, seeds, seconds=25, checkpoint_sha256, source_binding):
    """Persist an immutable recipe, actual traces and unqualified stress report.

    ``source_binding`` is the caller's validated checkpoint/recipe provenance.
    An optional ``evaluation_source`` member must match source_identity(). The
    checkpoint digest is a caller assertion; this function never deserializes it.
    All model weights and Python/NumPy/Torch CPU RNG states are preserved.
    """
    validate_experiment(experiment)
    seeds = _seeds(seeds, 'stress evaluation seeds')
    if (type(seconds) not in (float, int) or not math.isfinite(seconds) or
            not .01 <= seconds <= 60 or not math.isclose(seconds*100, round(seconds*100))):
        raise ValueError('seconds must contain whole 100 Hz decisions within .01..60')
    if (not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256)!=64 or
            any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('explicit checkpoint SHA-256 required')
    if not isinstance(source_binding, dict) or not source_binding:
        raise ValueError('nonempty caller source binding required')
    binding = json.loads(json.dumps(source_binding, allow_nan=False))
    sources = source_identity()
    if 'evaluation_source' in binding and binding['evaluation_source'] != sources:
        raise ValueError('evaluation source differs from caller source binding')
    state = {key:value.detach().clone() for key,value in model.policy.state_dict().items()}
    weights = _weights_digest(state)
    modes = [(module, module.training) for module in model.policy.modules()]
    exp = deepcopy(experiment)
    exp['simulation_config']['max_steps'] = round(seconds*100)
    matrix = scenarios(exp)
    # Validate all scenarios before creating artifacts.
    for scenario in matrix:
        candidate = deepcopy(exp)
        candidate['observation_config'].update(scenario['observation_override'])
        env = TrainingEnv(candidate, seeds, randomization=scenario['randomization'])
        env.close()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root/'traces').mkdir()
    recipe = dict(schema='cartpole-transfer-checks-recipe-v1', status='DEVELOPMENT_ONLY',
        simulation_only=True, hardware_accepted=False, deployable=False, seconds=seconds,
        seeds=list(seeds), starts=['upright','hanging'], scenarios=matrix,
        experiment=exp, experiment_sha256=experiment_identity(exp),
        checkpoint_sha256=checkpoint_sha256, in_memory_weights_sha256=weights,
        source_binding=binding, source_binding_sha256=digest(binding), evaluation_source=sources,
        scope='Finite engineering stress scenarios; no fitted uncertainty, physical tracking or hardware approval.',
        binding_boundary='Caller validates checkpoint loading and seed independence; supplied checkpoint hash is not revalidated against a file here.',
        angle_offset_semantics='Rotate only actor input sin/cos by a fixed one degree; preserve validity, ages, magnetic rate and true plant angle. No recalibration.',
        fault_semantics='Terminal simulation detection only. Invalid telemetry stops TrainingEnv before another step; physical disabled-motor dynamics and runner STOP not tested.',
        success_criterion='Full requested horizon; final >=5 s continuously abs(theta)<.15, abs(omega)<1, abs(x)<.12; no fault.',
        initial_state_scope='Seeded x +/-15 mm, v +/-0.025 m/s, theta +/-0.07 rad around upright/hanging, omega +/-0.15 rad/s.',
        action_trace_semantics='requested_raw_action is proposed demand; actual_executor_raw is held/accepted executor demand. Invalid-input decisions may advance no physical time.')
    _write_new(root/'recipe.json', recipe)
    episodes = []
    try:
        with preserved_rng():
            for scenario in matrix:
                candidate = deepcopy(exp)
                candidate['observation_config'].update(scenario['observation_override'])
                env = TrainingEnv(candidate, seeds, randomization=scenario['randomization'])
                try:
                    for seed in seeds:
                        for start in recipe['starts']:
                            episodes.append(_episode(model, env, seed=seed, start=start,
                                scenario=scenario, root=root))
                finally:
                    env.close()
            if source_identity() != sources:
                raise ValueError('evaluation source changed during suite; no completed report written')
            if _weights_digest(model.policy.state_dict()) != weights:
                raise ValueError('policy weights changed during evaluation; no completed report written')
    finally:
        if _weights_digest(model.policy.state_dict()) != weights:
            model.policy.load_state_dict(state)
        for module, training in modes:
            module.training = training
    summary = []
    for scenario in matrix:
        for start in recipe['starts']:
            rows = [r for r in episodes if r['scenario']==scenario['name'] and r['start']==start]
            summary.append(dict(scenario=scenario['name'], start=start, expected=scenario['expected'],
                episodes=len(rows), holds=sum(r['hold_success'] for r in rows),
                faults=sum(r['fault_reason'] is not None for r in rows),
                fault_response_passes=sum(r['fault_response_pass'] is True for r in rows),
                worst_max_abs_x_m=max(r['max_abs_x_m'] for r in rows)))
    report = dict(schema='cartpole-transfer-checks-report-v1', status='DEVELOPMENT_ONLY',
        completed=True, simulation_only=True, hardware_accepted=False, deployable=False,
        checkpoint_sha256=checkpoint_sha256, in_memory_weights_sha256=weights,
        recipe_sha256=sha256((root/'recipe.json').read_bytes()).hexdigest(),
        evaluation_source=sources, episodes=episodes, summary=summary,
        runner_replay=dict(status='outstanding',
            reason='This suite evaluates the supplied predictor through TrainingEnv. Candidate-specific numeric export parity and exact host/firmware runner replay were not performed.'),
        scenario_gate_pass=bool(seconds>=5 and all(
            r['task_success'] if r['expected']=='task_success' else r['fault_response_pass'] for r in episodes)))
    _write_new(root/'report.json', report)
    return report
