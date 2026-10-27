"""Offline PPO training on an explicit governed/observed model experiment.

Checkpoints are trusted SB3 training files. actor.json is a bounded numeric SDK
inference artifact; neither file approves physical deployment.
"""
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.policies.neural import SCHEMA as ACTOR_SCHEMA, RUNTIME_FILES, validate_actor
from cartpole_sdk.policies.rig_observation import (SCHEMA as OBSERVATION_SCHEMA,
    contract_from_config, select_rig_observation)
from cartpole_sdk.sim.config import integer
from cartpole_sdk.sim.rig_gym_env import RigCartpoleEnv
from .model_experiment import digest, read_json, validate_experiment
from .model_search import provenance as base_provenance, _seeds
from .rollout import OBJECTIVE, SUCCESS, initial_state
from .training_recipe import (DEFAULT_TRAINING_SEEDS, SAMPLING_ALGORITHM,
                              sample_configs, validate_randomization)


def provenance():
    result = base_provenance()
    root = Path(__file__).resolve().parents[1]
    for name in ('learning/ppo.py', 'learning/training_recipe.py', '../cartpole_vision/state.py', *RUNTIME_FILES):
        result['code_sha256'][name] = sha256((root / name).read_bytes()).hexdigest()
    result['dependencies'].update({name: version(name) for name in ('torch', 'stable-baselines3')})
    return result


def experiment_identity(experiment):
    # Relocating identical evidence from a laptop to Colab does not change it.
    value = json.loads(json.dumps(experiment))
    value['admission']['model'].pop('source_path', None)
    return digest(value)


class TrainingEnv(gym.Env):
    """Normalized PPO action maps once to raw m/s²; no actor ground-truth input."""
    metadata = {'render_modes': []}

    def __init__(self, experiment, seeds, *, randomization=None, randomization_seed=0, reward_scale=1.):
        config, parameters, observations, _ = validate_experiment(experiment)
        self.nominal_config, self.nominal_observations = config, observations
        self.parameters, self.ttl_ms = parameters, experiment['ttl_ms']
        self.randomization = validate_randomization(randomization)
        integer(randomization_seed, 'randomization seed')
        self.randomization_seed = randomization_seed
        if type(reward_scale) not in (int, float) or not math.isfinite(reward_scale) or not 0 < reward_scale <= 1:
            raise ValueError('reward_scale must be finite in (0,1]')
        self.reward_scale = reward_scale
        self.core = RigCartpoleEnv(config, parameters=parameters, observation_config=observations,
                                  ttl_ms=experiment['ttl_ms'])
        self.seeds = _seeds(seeds, 'episode seeds')
        self.action_scale = min(config.max_acceleration_m_s2, max(parameters.amax_s, parameters.amax_b))
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = self.core.observation_space
        self._episode_index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_index = 0
        if options is None:
            episode_seed = self.seeds[(self._episode_index // 2) % len(self.seeds)]
            start = 'hanging' if self._episode_index % 2 == 0 else 'upright'
        else:
            episode_seed, start = options['episode_seed'], options['start']
        sample_index = self._episode_index
        config, observations = sample_configs(self.nominal_config, self.nominal_observations,
            self.randomization, seed=self.randomization_seed, episode=sample_index)
        if self.randomization is not None:
            self.core.close()
            self.core = RigCartpoleEnv(config, parameters=self.parameters,
                                      observation_config=observations, ttl_ms=self.ttl_ms)
        self.episode_parameters = dict(randomization_episode=sample_index,
            randomization_seed=self.randomization_seed, simulation_config=asdict(config),
            observation_config=asdict(observations))
        self._episode_index += 1
        state = self.initial_state(episode_seed, start)
        observation, info = self.core.reset(seed=episode_seed, options={'state': state})
        self._previous = None
        self.steps, self.total_reward, self.dwell, self.balanced = 0, 0., 0., 0
        self.max_x, self.max_speed = abs(state[0]), abs(state[1])
        self.saturated, self.braking, self.fault = 0, 0, None
        self.longest_dwell, self.capture_s, self._balance_history = 0., None, []
        self.episode_seed, self.start, self.initial = episode_seed, start, state
        self._last_observation = observation
        self._ended = False
        info['episode_parameters'] = self.episode_parameters
        return observation, info

    def initial_state(self, episode_seed, start):
        """Reset-family hook; default retains the original seeded task starts."""
        return initial_state(episode_seed, start)

    def step(self, action):
        if self._ended:
            raise RuntimeError('episode ended; reset before stepping')
        array = np.asarray(action)
        if array.shape != (1,) or array.dtype.kind not in 'fiu' or not np.all(np.isfinite(array)) or abs(float(array[0])) > 1:
            raise ValueError('PPO action must be one finite normalized value in [-1,1]')
        sdk = self.core.sdk_observation
        try:
            select_rig_observation(sdk, self.core.channels, contract=self.core.observation_contract)
            if self._previous is not None:
                sequence, previous_ns = self._previous
                if sdk.sequence <= sequence or abs(sdk.available_ns - previous_ns - 10_000_000) > 5_000_000:
                    raise ValidationError('replayed observation or incompatible 100 Hz cadence')
        except ValidationError as exc:
            self.fault, self._ended = str(exc), True
            reward = -3. * (self.core.config.max_steps - self.steps)
            self.total_reward += reward
            return self._last_observation, reward * self.reward_scale, True, False, {'fault_reason': self.fault,
                                                                  'episode_result': self.result()}
        self._previous = sdk.sequence, sdk.available_ns
        raw = float(array[0]) * self.action_scale
        before = self.core.ground_truth.time_ns
        observation, _, terminated, truncated, info = self.core.step([raw])
        self._last_observation = observation
        self.steps += 1
        truth = self.core.ground_truth
        balanced = abs(truth.theta) < .15 and abs(truth.omega) < 1 and abs(truth.x) < .12
        self.dwell = self.dwell + (truth.time_ns - before) / 1e9 if balanced else 0.
        self.balanced += int(balanced)
        self._balance_history.append(int(balanced))
        self.longest_dwell = max(self.longest_dwell, self.dwell)
        if self.capture_s is None and self.dwell >= 1 - 1e-9:
            self.capture_s = self.steps / 100.
        self.max_x, self.max_speed = max(self.max_x, abs(truth.x)), max(self.max_speed, abs(truth.v))
        self.saturated += int(self.core.diagnostics.force_saturated)
        self.braking += int(self.core.diagnostics.rail_braking)
        reward = (math.cos(truth.theta) - .3 * (truth.x / self.core.config.rail_half_m)**2
                  - .001 * truth.omega**2 - .0002 * raw**2 + 2 * int(balanced))
        self.fault = self.core.fault_reason
        if terminated or truncated:
            reward += self.dwell * 100 - 3 * (self.core.config.max_steps - self.steps)
            self._ended = True
            # This objective has a real finite terminal payoff. A Gym time-limit
            # truncation would make SB3 add gamma*V(terminal_obs), changing it.
            terminated, truncated = True, False
        self.total_reward += reward
        if self._ended:
            info['episode_result'] = self.result()
        return observation, float(reward * self.reward_scale), terminated, truncated, info

    def result(self, *, hold_seconds=5.):
        if type(hold_seconds) not in (int, float) or not math.isfinite(hold_seconds) or not 0 < hold_seconds <= 60:
            raise ValueError('hold_seconds must be positive and at most 60')
        window = max(1, math.ceil(hold_seconds * 100))
        full = self.fault is None and self.steps == self.core.config.max_steps
        return dict(seed=self.episode_seed, start=self.start, initial_state=list(self.initial),
            final_state=asdict(self.core.ground_truth), score=self.total_reward / self.core.config.max_steps,
            success=bool(self.dwell >= 1 - 1e-9 and self.fault is None and self.steps == self.core.config.max_steps),
            hold_success=bool(full and self.dwell >= hold_seconds - 1e-9),
            hold_seconds=hold_seconds, first_one_second_capture_s=self.capture_s,
            longest_balance_dwell_s=self.longest_dwell,
            final_window_balanced_fraction=sum(self._balance_history[-window:]) / window,
            final_window_complete=bool(full and self.steps >= window),
            episode_parameters=self.episode_parameters,
            steps=self.steps, fault_reason=self.fault, final_balance_dwell_s=self.dwell,
            balanced_fraction=self.balanced / self.core.config.max_steps,
            max_abs_x_m=self.max_x, max_abs_speed_m_s=self.max_speed,
            force_saturated_control_steps=self.saturated, rail_braking_control_steps=self.braking)

    def close(self):
        self.core.close()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def export_actor(model, manifest):
    layers = []
    for module in [*model.policy.mlp_extractor.policy_net, model.policy.action_net]:
        if isinstance(module, torch.nn.Linear):
            layers.append(dict(weights=module.weight.detach().cpu().tolist(),
                               bias=module.bias.detach().cpu().tolist(), activation='identity'))
        elif isinstance(module, torch.nn.Tanh):
            layers[-1]['activation'] = 'tanh'
        else:
            raise ValueError('unsupported PPO actor architecture')
    root = Path(__file__).resolve().parents[1]
    actor = dict(schema=ACTOR_SCHEMA, observation_schema=OBSERVATION_SCHEMA,
        status='DEVELOPMENT_ONLY', hardware_accepted=False, rate_hz=100,
        angle_source=manifest['experiment']['observation_config']['angle_source'],
        observation_contract=contract_from_config(manifest['experiment']['observation_config']),
        executor_parameters=manifest['experiment']['executor_parameters'],
        feature_order=list(model.policy.features_extractor.extractors),
        layers=layers, action_scale_m_s2=manifest['action_scale_m_s2'],
        experiment_sha256=manifest['experiment_sha256'], checkpoint_sha256=manifest['checkpoint_sha256'],
        model_content_sha256=manifest['experiment']['admission']['model']['content_sha256'],
        runtime_sha256={name: sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES})
    actor['content_sha256'] = digest(actor)
    validate_actor(actor)
    return actor


def _save_checkpoint(model, directory, common):
    if provenance() != common['provenance']:
        raise ValueError('training source/dependencies changed; refusing misleading checkpoint')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'checkpoint.zip').exists() or (directory / 'manifest.json').exists():
        raise FileExistsError(directory)
    model.save(directory / 'checkpoint.zip')
    manifest = dict(common, timesteps_completed=int(model.num_timesteps),
                    checkpoint_sha256=sha256((directory / 'checkpoint.zip').read_bytes()).hexdigest())
    actor = export_actor(model, manifest)
    _write_new(directory / 'actor.json', actor)
    manifest['actor_sha256'] = sha256((directory / 'actor.json').read_bytes()).hexdigest()
    manifest['content_sha256'] = digest(manifest)
    _write_new(directory / 'manifest.json', manifest)
    return manifest


def load_checkpoint(directory):
    """Load only a trusted training checkpoint after integrity/config checks."""
    directory = Path(directory)
    manifest, _ = read_json(directory / 'manifest.json')
    content = dict(manifest)
    pin = content.pop('content_sha256', None)
    if (manifest.get('schema') != 'cartpole-ppo-checkpoint-v1' or digest(content) != pin or
            manifest.get('status') != 'DEVELOPMENT_ONLY' or manifest.get('hardware_accepted') is not False):
        raise ValueError('invalid PPO checkpoint manifest/hash')
    validate_experiment(manifest['experiment'])
    if experiment_identity(manifest['experiment']) != manifest['experiment_sha256']:
        raise ValueError('checkpoint experiment hash mismatch')
    if provenance() != manifest['provenance']:
        raise ValueError('checkpoint source/dependencies differ from this runtime')
    if sha256((directory / 'checkpoint.zip').read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('checkpoint archive hash mismatch')
    if sha256((directory / 'actor.json').read_bytes()).hexdigest() != manifest['actor_sha256']:
        raise ValueError('checkpoint actor hash mismatch')
    return PPO.load(directory / 'checkpoint.zip', device='cpu'), manifest


def train_ppo(experiment, output, *, training_seeds=DEFAULT_TRAINING_SEEDS, seed, total_timesteps,
              n_steps=1024, batch_size=64, n_epochs=10, checkpoint_every=10000, resume=None,
              gamma=.999, gae_lambda=.98, learning_rate=.0003, reward_scale=.01, randomization=None):
    """Train on a reproducible seed pool; held-out evaluation remains nominal."""
    validate_experiment(experiment)
    seeds = _seeds(training_seeds, 'training seeds')
    for value, name in ((seed, 'seed'), (total_timesteps, 'total_timesteps'), (n_steps, 'n_steps'),
                        (batch_size, 'batch_size'), (n_epochs, 'n_epochs'), (checkpoint_every, 'checkpoint_every')):
        integer(value, name, positive=name != 'seed')
    if n_steps < 2 or batch_size < 2 or batch_size > n_steps or n_steps % batch_size or total_timesteps % n_steps:
        raise ValueError('steps must be a multiple of n_steps; batch_size >=2 must divide n_steps')
    for value, name in ((gamma, 'gamma'), (gae_lambda, 'gae_lambda'), (learning_rate, 'learning_rate'),
                        (reward_scale, 'reward_scale')):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'{name} must be finite in (0,1]')
    recipe = dict(randomization=validate_randomization(randomization), sampling_algorithm=SAMPLING_ALGORITHM,
                  reset_schedule='ordered seed pool, alternating hanging/upright; restarts on resume',
                  seed=seed)
    settings = dict(n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=learning_rate,
                    reward_scale=reward_scale,
                    gamma=gamma, gae_lambda=gae_lambda, clip_range=.2, ent_coef=0., vf_coef=.5, max_grad_norm=.5,
                    policy='MultiInputPolicy', hidden_layers=[64, 64], activation='tanh', device='cpu')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model, old = (None, None) if resume is None else load_checkpoint(resume)
    if old is not None and (old['experiment_sha256'] != experiment_identity(experiment) or
            old['training_seeds'] != list(seeds) or old['seed'] != seed or old['hyperparameters'] != settings or
            old.get('training_recipe') != recipe):
        raise ValueError('resume configuration differs from frozen model/seeds/hyperparameters')
    torch.set_num_threads(1)
    env = TrainingEnv(experiment, seeds, randomization=recipe['randomization'], randomization_seed=seed,
                      reward_scale=reward_scale)
    output.mkdir(parents=True)
    common = dict(schema='cartpole-ppo-checkpoint-v1', algorithm='PPO', status='DEVELOPMENT_ONLY',
        hardware_accepted=False, simulation_only=True, experiment=experiment,
        experiment_sha256=experiment_identity(experiment), provenance=provenance(),
        training_seeds=list(seeds), seed=seed, hyperparameters=settings,
        training_recipe=recipe,
        requested_additional_timesteps=total_timesteps, checkpoint_every=checkpoint_every,
        action_scale_m_s2=env.action_scale, observation_schema=OBSERVATION_SCHEMA,
        action_semantics='normalized [-1,1] scaled to raw m/s²; single 1 kHz executor governor',
        objective=OBJECTIVE, success_definition=SUCCESS,
        reward_semantics='Optimizer reward is benchmark reward times reward_scale; episode score and evaluation preserve the unscaled benchmark.',
        resumed_from_checkpoint_sha256=None if old is None else old['checkpoint_sha256'],
        resume_semantics='Optimizer/timestep continuation; episodes and randomization index restart at zero from the fixed training pool, not bitwise uninterrupted RNG continuation.')
    _write_new(output / 'experiment.json', experiment)

    class SaveProgress(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos', []):
                if 'episode_result' in info:
                    row = dict(timesteps=int(self.num_timesteps), **info['episode_result'])
                    with (output / 'episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            if self.num_timesteps % checkpoint_every == 0:
                _save_checkpoint(self.model, output / 'checkpoints' / f'step-{self.num_timesteps:09d}', common)
                print(f'DEVELOPMENT_ONLY PPO checkpoint steps={self.num_timesteps}', flush=True)
            return True

    try:
        if model is None:
            model = PPO('MultiInputPolicy', env, seed=seed, device='cpu', verbose=0,
                n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda, clip_range=.2,
                ent_coef=0., vf_coef=.5, max_grad_norm=.5,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh))
        else:
            model.set_env(env)
        model.set_logger(configure(str(output / 'logs'), ['csv']))
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=old is None,
                    callback=SaveProgress(), log_interval=1)
        return _save_checkpoint(model, output, common)
    finally:
        env.close()


def evaluate_ppo(directory, *, seeds, hold_seconds=5.):
    seeds = _seeds(seeds, 'evaluation seeds')
    # Check split before deserializing a training archive.
    recorded, _ = read_json(Path(directory) / 'manifest.json')
    if set(seeds) & set(recorded.get('training_seeds', [])):
        raise ValueError('held-out evaluation seeds overlap training seeds')
    model, manifest = load_checkpoint(directory)
    env = TrainingEnv(manifest['experiment'], seeds)
    episodes = []
    try:
        for seed in seeds:
            for start in ('hanging', 'upright'):
                observation, _ = env.reset(options=dict(episode_seed=seed, start=start))
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                episodes.append(env.result(hold_seconds=hold_seconds))
    finally:
        env.close()
    report = dict(schema='cartpole-ppo-evaluation-v1', status='DEVELOPMENT_ONLY', hardware_accepted=False,
        checkpoint_sha256=manifest['checkpoint_sha256'], experiment_sha256=manifest['experiment_sha256'],
        model_content_sha256=manifest['experiment']['admission']['model']['content_sha256'],
        task_validation=manifest['experiment']['admission']['task_validation'], seeds=list(seeds),
        objective=OBJECTIVE, success_definition=SUCCESS, episodes=episodes, episodes_count=len(episodes),
        evaluation_mode='nominal frozen model; no training randomization', hold_seconds=hold_seconds,
        hold_success_definition='full horizon without fault; final continuous balance dwell >= hold_seconds using the unchanged balance bounds',
        hold_successes=sum(episode['hold_success'] for episode in episodes),
        successes=sum(episode['success'] for episode in episodes),
        mean_score=float(np.mean([episode['score'] for episode in episodes])))
    report['by_start'] = {start: dict(episodes_count=len(rows),
        successes=sum(row['success'] for row in rows), hold_successes=sum(row['hold_success'] for row in rows),
        mean_score=float(np.mean([row['score'] for row in rows])),
        mean_final_balance_dwell_s=float(np.mean([row['final_balance_dwell_s'] for row in rows])),
        fault_episodes=sum(row['fault_reason'] is not None for row in rows))
        for start in ('hanging', 'upright') if (rows := [row for row in episodes if row['start'] == start])}
    report['content_sha256'] = digest(report)
    return report
