"""Model-bound CEM on the governed, observed simulator; development only.

This is numeric controller parameter search, not neural RL. Its distinct
artifact schema prevents legacy direct/provisional evaluation or export.
"""
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np

from cartpole_sdk.policies import HostRatePolicy, HostPolicyParameters
from cartpole_sdk.sim.config import integer
from cartpole_sdk.sim.rig_gym_env import RigCartpoleEnv
from .model_experiment import digest, read_json, validate_experiment, complete_config
from .rollout import OBJECTIVE, SUCCESS, initial_state, simulation_channels
from .search import SEARCH_BOUNDS

SCHEMA = 'cartpole-model-bound-cem-v1'
SOURCE_FILES = ('contracts.py', 'recording.py', 'policies/host.py', 'policies/rig_observation.py',
    'control/parameters.py', 'control/tuned_parameters.json', 'control/reference.py',
    'control/motion.py', 'sim/backend.py', 'sim/model.py', 'sim/config.py',
    'sim/executor.py', 'sim/rig_gym_env.py', 'learning/model_experiment.py',
    'learning/model_search.py', 'learning/rollout.py', 'learning/search.py')


def provenance():
    root = Path(__file__).resolve().parents[1]
    return dict(python=platform.python_version(),
        dependencies={name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')},
        code_sha256={name: sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES})


def _seeds(values, label):
    if not isinstance(values, (tuple, list)) or not values or len(set(values)) != len(values):
        raise ValueError(f'{label} must be distinct nonempty seeds')
    for seed in values:
        integer(seed, label)
    return tuple(values)


def rollout(parameters, experiment, *, seed, start, trajectory=False):
    config, executor, observations, _ = validate_experiment(experiment)
    env = RigCartpoleEnv(config, parameters=executor, observation_config=observations,
                         ttl_ms=experiment['ttl_ms'])
    state = initial_state(seed, start)
    score = dwell = 0.
    balanced = steps = saturated = braking = 0
    peak_x, peak_speed = abs(state[0]), abs(state[1])
    fault, trace = None, []
    try:
        env.reset(seed=seed, options={'state': state})
        policy = HostRatePolicy(simulation_channels(), rate_hz=100, parameters=parameters)
        policy.reset(env.session)
        for _ in range(config.max_steps):
            intent = policy.act(env.sdk_observation)
            if intent is None:
                fault = policy.fault_reason or 'policy returned no request'
                break
            before = env.ground_truth.time_ns
            _, _, terminated, truncated, _ = env.step([intent.acceleration_m_s2])
            steps += 1
            truth = env.ground_truth
            dt = (truth.time_ns - before) / 1e9
            ok = abs(truth.theta) < .15 and abs(truth.omega) < 1 and abs(truth.x) < .12
            balanced += int(ok)
            dwell = dwell + dt if ok else 0.
            peak_x, peak_speed = max(peak_x, abs(truth.x)), max(peak_speed, abs(truth.v))
            score += (math.cos(truth.theta) - .3 * (truth.x / config.rail_half_m)**2
                      - .001 * truth.omega**2 - .0002 * intent.acceleration_m_s2**2 + 2 * int(ok))
            saturated += int(env.diagnostics.force_saturated)
            braking += int(env.diagnostics.rail_braking)
            if trajectory:
                trace.append(dict(**asdict(truth), raw_acceleration_m_s2=intent.acceleration_m_s2,
                    executed_acceleration_m_s2=env.diagnostics.executed_acceleration_m_s2,
                    policy_mode=policy.mode, rail_braking=env.diagnostics.rail_braking))
            if terminated:
                fault = env.fault_reason or 'environment terminated'
                break
            if truncated:
                break
        final = asdict(env.ground_truth)
    finally:
        env.close()
    horizon = config.max_steps / 100
    score = (score - 3 * (config.max_steps - steps)) / config.max_steps + dwell / horizon
    result = dict(seed=seed, start=start, initial_state=list(state), final_state=final,
        score=float(score), success=bool(dwell >= 1 - 1e-9 and fault is None and steps == config.max_steps),
        final_balance_dwell_s=dwell, balanced_fraction=balanced / config.max_steps,
        max_abs_x_m=peak_x, max_abs_speed_m_s=peak_speed, steps=steps, fault_reason=fault,
        force_saturated_control_steps=saturated, rail_braking_control_steps=braking)
    if trajectory:
        result['trajectory'] = trace
    return result


def _evaluate(parameters, experiment, seeds, *, trajectory=False):
    episodes = [rollout(parameters, experiment, seed=seed, start=start, trajectory=trajectory)
                for seed in seeds for start in ('hanging', 'upright')]
    return dict(episodes=episodes, episodes_count=len(episodes),
                mean_score=float(np.mean([episode['score'] for episode in episodes])),
                successes=sum(episode['success'] for episode in episodes))


def train(experiment, *, seed, training_seeds, generations, population, progress=None):
    _, _, _, baseline = validate_experiment(experiment)
    training_seeds = _seeds(training_seeds, 'training seeds')
    for value, name in ((seed, 'seed'), (generations, 'generations'), (population, 'population')):
        integer(value, name, positive=name != 'seed')
    if not 1 <= generations <= 100 or not 4 <= population <= 1000:
        raise ValueError('require generations 1..100 and population 4..1000')
    sources = provenance()
    rng = np.random.default_rng(seed)
    names = list(SEARCH_BOUNDS)
    lower, upper = np.log(np.array(list(SEARCH_BOUNDS.values()))).T
    # The prior is supplied, never inferred from historical presets or live state.
    if any(getattr(baseline, name) <= 0 for name in names):
        raise ValueError('CEM requires positive searched prior parameters')
    mean = np.log([getattr(baseline, name) for name in names])
    std = np.full(len(names), .28)
    before = _evaluate(baseline, experiment, training_seeds)
    best, best_score, history = baseline, before['mean_score'], []
    for generation in range(generations):
        vectors = np.clip(rng.normal(mean, std, (population - 1, len(names))), lower, upper)
        candidates = [best] + [replace(baseline, **dict(zip(names, np.exp(vector).tolist()))) for vector in vectors]
        scores = [best_score] + [_evaluate(candidate, experiment, training_seeds)['mean_score']
                                for candidate in candidates[1:]]
        elite = np.argsort(-np.array(scores), kind='stable')[:max(2, population // 4)]
        elite_vectors = np.log([[getattr(candidates[i], name) for name in names] for i in elite])
        mean = .3 * mean + .7 * np.mean(elite_vectors, axis=0)
        std = np.maximum(.06, .3 * std + .7 * np.std(elite_vectors, axis=0))
        winner = int(elite[0])
        best, best_score = candidates[winner], float(scores[winner])
        entry = dict(generation=generation + 1, best_score=best_score,
                     candidate_scores=[float(value) for value in scores],
                     elite_indices=[int(i) for i in elite], parameters=asdict(best))
        history.append(entry)
        if progress is not None:
            progress(entry)
    if provenance() != sources:
        raise ValueError('training implementation changed during the experiment')
    artifact = dict(schema=SCHEMA, status='DEVELOPMENT_ONLY', simulation_only=True,
        hardware_accepted=False, experiment=experiment, parameters=asdict(best), provenance=sources,
        training=dict(algorithm='cross_entropy_policy_search', seed=seed,
            training_seeds=list(training_seeds), generations=generations, population=population,
            search_bounds={name: list(bounds) for name, bounds in SEARCH_BOUNDS.items()}, objective=OBJECTIVE, success_definition=SUCCESS,
            pretraining=before, best_score=best_score, history=history))
    artifact['content_sha256'] = digest(artifact)
    return artifact


def validate_artifact(artifact):
    keys = {'schema', 'status', 'simulation_only', 'hardware_accepted', 'experiment',
            'parameters', 'provenance', 'training', 'content_sha256'}
    if (not isinstance(artifact, dict) or set(artifact) != keys or artifact['schema'] != SCHEMA or
            artifact['status'] != 'DEVELOPMENT_ONLY' or artifact['simulation_only'] is not True or
            artifact['hardware_accepted'] is not False):
        raise ValueError('unsupported model-bound artifact or acceptance claim')
    content = dict(artifact)
    pin = content.pop('content_sha256')
    if digest(content) != pin:
        raise ValueError('model-bound artifact content hash mismatch')
    validate_experiment(artifact['experiment'])
    complete_config(artifact['parameters'], HostPolicyParameters, 'policy parameters')
    _seeds(artifact['training'].get('training_seeds'), 'training seeds')
    return artifact


def save_artifact(artifact, path):
    validate_artifact(artifact)
    raw = json.dumps(artifact, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if len(raw.encode()) > 4 * 1024 * 1024:
        raise ValueError('model-bound artifact exceeds 4 MiB')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(raw)


def read_artifact(path):
    value, _ = read_json(path)
    return validate_artifact(value)


def evaluate(artifact, *, seeds, trajectory=False):
    validate_artifact(artifact)
    seeds = _seeds(seeds, 'evaluation seeds')
    if set(seeds) & set(artifact['training']['training_seeds']):
        raise ValueError('held-out evaluation seeds overlap training seeds')
    current = provenance()
    if current != artifact['provenance']:
        raise ValueError('evaluation implementation/dependencies differ from frozen training provenance')
    experiment = artifact['experiment']
    baseline = HostPolicyParameters(**experiment['prior_parameters'])
    learned = HostPolicyParameters(**artifact['parameters'])
    before = _evaluate(baseline, experiment, seeds, trajectory=trajectory)
    after = _evaluate(learned, experiment, seeds, trajectory=trajectory)
    report = dict(schema='cartpole-model-bound-evaluation-v1', status='DEVELOPMENT_ONLY',
        simulation_only=True, hardware_accepted=False, artifact_content_sha256=artifact['content_sha256'],
        experiment_sha256=digest(experiment), model=experiment['admission']['model'],
        task_validation=experiment['admission']['task_validation'], provenance=current,
        objective=OBJECTIVE, success_definition=SUCCESS, seeds=list(seeds),
        baseline=before, learned=after, mean_score_improvement=after['mean_score'] - before['mean_score'])
    report['content_sha256'] = digest(report)
    return report
