"""Candidate-specific offline export/inference parity, separate from task success.

The actual numeric SDK actor drives the unchanged governed simulator. SB3 is
evaluated on identical encoded observations as a comparison only. This does not
exercise live transport, host STOP, firmware execution or Jetson deadlines.
"""
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from cartpole_sdk.policies.neural import load_neural_policy
from cartpole_sdk.policies.rig_observation import encode_rig_observation
from cartpole_sdk.sim.rig_gym_env import RigCartpoleEnv
from .diagnostics import preserved_rng
from .model_experiment import validate_experiment
from .model_search import _seeds
from .ppo import _write_new, experiment_identity, export_actor, provenance
from .rollout import initial_state

RAW_ABSOLUTE_TOLERANCE_M_S2 = 1e-5


def source_identity():
    return dict(runtime=provenance(), evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        rng_helper_sha256=sha256(Path(__file__).with_name('diagnostics.py').read_bytes()).hexdigest())


def _state_digest(state):
    result = sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def model_weights_sha256(model):
    return _state_digest(model.policy.state_dict())


def _latency(values):
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
        p50=float(np.quantile(values,.5)) if values else None,
        p95=float(np.quantile(values,.95)) if values else None,
        max=max(values) if values else None)


def _episode(model, env, actor_path, root, *, seed, start, action_scale):
    initial = initial_state(seed, start)
    encoded, _ = env.reset(seed=seed, options={'state':initial})
    # Load the saved artifact through the real bounded, hash-checking SDK loader.
    policy = load_neural_policy(actor_path, channels=env.channels)
    policy.reset(env.session)
    trace_path = f'traces/{start}-{seed}.jsonl'
    reset_ns = env.ground_truth.time_ns
    steps = compared = rejected = 0
    dwell = longest = max_error = 0.
    max_x = abs(initial[0]); capture = None
    balance_history, cpu_times, wall_times = [], [], []
    policy_fault = None
    with (root/trace_path).open('x') as stream:
        while not (env.terminated or env.truncated):
            sdk_observation = env.sdk_observation
            sdk_encoded = encode_rig_observation(sdk_observation, env.channels)
            if any(not np.array_equal(encoded[key], sdk_encoded[key]) for key in encoded):
                raise ValueError('SDK and SB3 observations differ before parity comparison')
            reference, _ = model.predict(encoded, deterministic=True)
            reference = np.asarray(reference)
            if reference.shape != (1,) or not np.all(np.isfinite(reference)) or abs(float(reference[0]))>1:
                raise ValueError('SB3 must return one finite normalized deterministic action')
            sb3_raw = float(reference[0])*action_scale
            before_ns = env.ground_truth.time_ns
            wall_started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
            intent = policy.act(sdk_observation)
            cpu_ns, wall_ns = time.thread_time_ns()-cpu_started, time.perf_counter_ns()-wall_started
            cpu_times.append(cpu_ns); wall_times.append(wall_ns)
            actual = error = None
            if intent is None:
                rejected += 1
                policy_fault = policy.fault_reason or 'numeric policy emitted no action'
            else:
                actual = intent.acceleration_m_s2
                error = abs(actual-sb3_raw)
                max_error = max(max_error, error); compared += 1
                # No normalization round trip: the numeric SDK demand is the
                # exact raw acceleration handed to RigCartpoleEnv.
                encoded, _, _, _, _ = env.step([actual])
                steps += 1
                truth = env.ground_truth
                balanced = abs(truth.theta)<.15 and abs(truth.omega)<1 and abs(truth.x)<.12
                dt = (truth.time_ns-before_ns)/1e9
                dwell = dwell+dt if balanced else 0.
                longest = max(longest,dwell); balance_history.append(int(balanced))
                if capture is None and dwell>=1-1e-9:
                    capture = (truth.time_ns-reset_ns)/1e9
                max_x = max(max_x,abs(truth.x))
            row = dict(decision_index=compared+rejected-1, sdk_observation=asdict(sdk_observation),
                encoded_observation={key:list(value) for key,value in sdk_encoded.items()},
                sb3_raw_action_m_s2=sb3_raw, sdk_raw_action_m_s2=actual,
                absolute_raw_discrepancy_m_s2=error, sdk_inference_thread_cpu_ns=cpu_ns,
                sdk_inference_wall_ns=wall_ns, ground_truth_after=asdict(env.ground_truth),
                actual_executor_raw_m_s2=env.diagnostics.raw_acceleration_m_s2,
                executed_acceleration_m_s2=env.diagnostics.executed_acceleration_m_s2,
                policy_fault_reason=policy_fault, simulator_fault_reason=env.fault_reason,
                physical_time_advanced=env.ground_truth.time_ns>before_ns)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            if policy_fault:
                break
    full = policy_fault is None and env.fault_reason is None and steps==env.config.max_steps
    result = dict(seed=seed, start=start, initial_state=list(initial), final_state=asdict(env.ground_truth),
        steps=steps, compared_actions=compared, rejected_inputs=rejected,
        max_abs_raw_discrepancy_m_s2=max_error if compared else None,
        emission_parity_pass=bool(compared and not rejected and max_error<=RAW_ABSOLUTE_TOLERANCE_M_S2),
        policy_fault_reason=policy_fault, simulator_fault_reason=env.fault_reason,
        full_horizon_complete=bool(full), hold_success=bool(full and dwell>=5-1e-9),
        final_window_complete=bool(full and steps>=500),
        final_window_balanced_fraction=sum(balance_history[-500:])/500,
        final_balance_dwell_s=dwell, longest_balance_dwell_s=longest,
        first_one_second_capture_s=capture, max_abs_x_m=max_x,
        reset_warmup_ns=reset_ns, elapsed_physical_s=(env.ground_truth.time_ns-reset_ns)/1e9,
        trace_path=trace_path, trace_sha256=sha256((root/trace_path).read_bytes()).hexdigest())
    return result, cpu_times, wall_times


def check_candidate(model, experiment, output, *, checkpoint_sha256, seeds=(49000,49001), seconds=5):
    """Export and compare the actual candidate on bounded nominal trajectories.

    The caller validates checkpoint loading and seed independence. A supplied
    checkpoint digest is recorded, not independently verified against a file.
    The model's in-memory weights, CPU RNG states and training flags are retained.
    """
    config, parameters, observations, _ = validate_experiment(experiment)
    seeds = _seeds(seeds, 'parity evaluation seeds')
    if (type(seconds) not in (float,int) or not math.isfinite(seconds) or
            not .01<=seconds<=60 or not math.isclose(seconds*100,round(seconds*100))):
        raise ValueError('seconds must contain whole 100 Hz decisions within .01..60')
    if (not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64 or
            any(c not in '0123456789abcdef' for c in checkpoint_sha256)):
        raise ValueError('explicit checkpoint SHA-256 required')
    if any(parameter.device.type!='cpu' for parameter in model.policy.parameters()):
        raise ValueError('candidate parity requires an explicitly loaded CPU model')
    sources = source_identity()
    state = {name:value.detach().clone() for name,value in model.policy.state_dict().items()}
    before = _state_digest(state)
    modes = [(module,module.training) for module in model.policy.modules()]
    scale = min(config.max_acceleration_m_s2,max(parameters.amax_s,parameters.amax_b))
    manifest = dict(experiment=deepcopy(experiment), experiment_sha256=experiment_identity(experiment),
        checkpoint_sha256=checkpoint_sha256, action_scale_m_s2=scale)
    actor = export_actor(model, manifest)
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False); (root/'traces').mkdir()
    actor_path = root/'actor.json'
    _write_new(actor_path, actor)
    config = replace(config,max_steps=round(seconds*100))
    recipe = dict(schema='cartpole-candidate-parity-recipe-v1', status='DEVELOPMENT_ONLY',
        simulation_only=True, hardware_accepted=False, deployable=False,
        checkpoint_sha256=checkpoint_sha256, model_weights_sha256_before=before,
        manifest=manifest, evaluation_simulation_config=asdict(config), seeds=list(seeds), seconds=seconds,
        starts=['upright','hanging'], evaluation_source=sources,
        actor_file_sha256=sha256(actor_path.read_bytes()).hexdigest(),
        actor_content_sha256=actor['content_sha256'], actor_runtime_sha256=actor['runtime_sha256'],
        raw_absolute_tolerance_m_s2=RAW_ABSOLUTE_TOLERANCE_M_S2,
        comparison='Actual NeuralPolicy.reset/act on the same six SDK channels and encoded observation as SB3 deterministic predict; numeric SDK action drives RigCartpoleEnv.',
        binding_boundary='Caller validates checkpoint loading and seed independence; file hash supplied by caller. Actual in-memory weights are independently hashed here.',
        precision='SB3 float32 versus numeric Python inference; normalized clipping then one saved physical action scale.',
        task_success_criterion='Full horizon, no faults, entire final five seconds abs(theta)<.15, abs(omega)<1, abs(x)<.12.',
        host_firmware_stop_replay='Not performed; no real host transport, firmware execution, or physical stop model.')
    _write_new(root/'recipe.json',recipe)
    episodes, all_cpu, all_wall = [], [], []
    try:
        with preserved_rng():
            env = RigCartpoleEnv(config,parameters=parameters,observation_config=observations,
                ttl_ms=experiment['ttl_ms'])
            try:
                for seed in seeds:
                    for start in recipe['starts']:
                        row,cpu,wall = _episode(model,env,actor_path,root,seed=seed,start=start,action_scale=scale)
                        episodes.append(row); all_cpu.extend(cpu); all_wall.extend(wall)
            finally:
                env.close()
            if source_identity()!=sources:
                raise ValueError('evaluation source changed during parity check; no completed report written')
            if model_weights_sha256(model)!=before:
                raise ValueError('model weights changed during parity check; no completed report written')
    finally:
        if model_weights_sha256(model)!=before:
            model.policy.load_state_dict(state)
        for module,mode in modes:
            module.training=mode
    comparisons = sum(row['compared_actions'] for row in episodes)
    maximum = max((row['max_abs_raw_discrepancy_m_s2'] for row in episodes if row['compared_actions']),default=None)
    report = dict(schema='cartpole-candidate-parity-report-v1', status='DEVELOPMENT_ONLY', completed=True,
        simulation_only=True, hardware_accepted=False, deployable=False,
        checkpoint_sha256=checkpoint_sha256, model_weights_sha256_before=before,
        model_weights_sha256_after=model_weights_sha256(model), evaluation_source=sources,
        recipe_sha256=sha256((root/'recipe.json').read_bytes()).hexdigest(),
        actor_file_sha256=sha256(actor_path.read_bytes()).hexdigest(),
        actor_content_sha256=actor['content_sha256'], actor_runtime_sha256=actor['runtime_sha256'],
        episodes=episodes, compared_actions=comparisons,
        rejected_inputs=sum(row['rejected_inputs'] for row in episodes),
        max_abs_raw_discrepancy_m_s2=maximum, raw_absolute_tolerance_m_s2=RAW_ABSOLUTE_TOLERANCE_M_S2,
        emission_parity_pass=bool(comparisons and all(row['emission_parity_pass'] for row in episodes)),
        task_hold_successes=sum(row['hold_success'] for row in episodes),
        inference_latency=dict(scope=('Mac only, not Jetson deadline guarantee' if platform.system()=='Darwin'
            else 'Local CPU host only, not Jetson deadline guarantee'), host=platform.platform(),
            measured_operation='Actual NeuralPolicy.act, including validation and encoding; first call included.',
            thread_cpu_ns=_latency(all_cpu), wall_ns=_latency(all_wall)),
        host_firmware_stop_replay=dict(status='outstanding',
            reason='No exact host/firmware STOP replay or physical transport/disabled-motion test performed.'),
        scope='Emission agreement on these sampled observations only; task success is reported separately and neither grants hardware approval.')
    _write_new(root/'report.json',report)
    return report
