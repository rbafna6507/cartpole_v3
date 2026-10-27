"""Offline learning probes; checkpoints deliberately are not SDK deployment actors."""
from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import time
import zipfile

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from .model_experiment import digest, validate_experiment
from .model_search import _seeds
from .ppo import TrainingEnv, provenance, experiment_identity, _write_new

VARIANTS = ('mixed', 'upright', 'upright-scaled', 'upright-lowstd')


class DiagnosticEnv(TrainingEnv):
    """Change only reset family and explicit fixed feature units for a probe."""
    def __init__(self, experiment, seeds, *, variant='mixed', reward_scale=.01):
        if variant not in VARIANTS:
            raise ValueError('unknown diagnostic variant')
        super().__init__(experiment, seeds, reward_scale=reward_scale)
        self.variant = variant
        self.state_scales = np.array([self.parameters.rail, self.parameters.vmax,
                                       1., 1., 10., self.action_scale, 1.])
        if variant == 'upright-scaled':
            self.observation_space = spaces.Dict(dict(self.observation_space.spaces,
                age_s=spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)))

    def _format(self, observation):
        result = {k: np.array(v, copy=True) for k, v in observation.items()}
        if self.variant == 'upright-scaled':
            result['state'] /= self.state_scales
            result['age_s'] /= .02
        return result

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._episode_index = 0
        if options is None and self.variant != 'mixed':
            options = dict(episode_seed=self.seeds[self._episode_index % len(self.seeds)], start='upright')
        observation, info = super().reset(seed=seed, options=options)
        return self._format(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self._format(observation), reward, terminated, truncated, info


@contextmanager
def preserved_rng():
    """Evaluation must not change the learner's future exploration RNG."""
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def evaluate_model(model, experiment, *, variant, seeds, seconds=10., traces=False):
    if not math.isfinite(seconds) or not .01 <= seconds <= 60 or not math.isclose(seconds*100, round(seconds*100)):
        raise ValueError('evaluation seconds must contain whole 100 Hz decisions in .01..60')
    evaluation = json.loads(json.dumps(experiment))
    evaluation['simulation_config']['max_steps'] = round(seconds*100)
    episodes = []
    with preserved_rng():
        env = DiagnosticEnv(evaluation, seeds, variant=variant)
        try:
            for seed in seeds:
                for start in ('hanging', 'upright'):
                    observation, _ = env.reset(options=dict(episode_seed=seed, start=start))
                    raw_actions, accelerations, trace = [], [], []
                    while True:
                        action, _ = model.predict(observation, deterministic=True)
                        raw_actions.append(float(action[0])*env.action_scale)
                        observation, _, terminated, truncated, _ = env.step(action)
                        truth = env.core.ground_truth
                        accelerations.append(env.core.diagnostics.executed_acceleration_m_s2)
                        if traces:
                            trace.append(dict(t=env.steps/100., x=truth.x, theta=truth.theta,
                                omega=truth.omega, raw=raw_actions[-1], executed=accelerations[-1]))
                        if terminated or truncated:
                            break
                    episode = env.result(hold_seconds=5.)
                    episode.update(raw_action_rms=float(np.sqrt(np.mean(np.square(raw_actions)))),
                        raw_action_delta_rms=float(np.sqrt(np.mean(np.square(np.diff(raw_actions))))) if len(raw_actions)>1 else 0.,
                        raw_action_at_limit_fraction=float(np.mean(np.abs(raw_actions)>=env.action_scale*.999)),
                        executed_acceleration_delta_rms=float(np.sqrt(np.mean(np.square(np.diff(accelerations))))) if len(accelerations)>1 else 0.)
                    if traces:
                        episode['trajectory'] = trace
                    episodes.append(episode)
        finally:
            env.close()
    by_start = {}
    for start in ('hanging','upright'):
        rows = [e for e in episodes if e['start']==start]
        by_start[start] = dict(episodes=len(rows), hold_successes=sum(e['hold_success'] for e in rows),
            faults=sum(e['fault_reason'] is not None for e in rows),
            **{name:float(np.mean([e[name] for e in rows])) for name in
               ('score','balanced_fraction','longest_balance_dwell_s','max_abs_x_m',
                'rail_braking_control_steps','raw_action_delta_rms')})
    return dict(episodes=episodes, by_start=by_start)


def diagnostic_provenance():
    return dict(provenance(), diagnostics_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def _source_snapshot(path):
    root=Path(__file__).resolve().parents[3]
    selected=[root/'pyproject.toml',root/'README.md',root/'uv.lock', *sorted((root/'src').rglob('*.py'))]
    selected += [root/'examples/rl_diagnostics.py',root/'scripts/rl-diagnostics']
    selected += list((root/'src/cartpole_sdk/sim').glob('*.xml'))
    selected += list((root/'src/cartpole_sdk/sim').glob('*.json'))
    selected += list((root/'src/cartpole_sdk/control').glob('*.json'))
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
        for p in selected:
            if p.is_file() and not p.is_symlink():
                z.write(p,p.relative_to(root))


def run_pilot(experiment, output, *, variant, seed, steps=20480, every=10240,
              training_seeds=tuple(range(10000,10128)), evaluation_seeds=(40000,40001),
              n_steps=1024, batch_size=64, epochs=10, evaluation_seconds=10.):
    validate_experiment(experiment)
    if variant not in VARIANTS:
        raise ValueError('unknown diagnostic variant')
    train, dev = _seeds(training_seeds,'training seeds'), _seeds(evaluation_seeds,'evaluation seeds')
    if set(train)&set(dev):
        raise ValueError('training/evaluation seed overlap')
    if any(type(x) is not int or x<1 for x in (steps,every,n_steps,batch_size,epochs)) or type(seed) is not int or seed<0:
        raise ValueError('positive integer budgets and nonnegative optimizer seed required')
    if batch_size<2 or n_steps<2 or n_steps%batch_size or every%n_steps or steps%every:
        raise ValueError('budgets must divide evaluation/rollout/batch intervals')
    if not math.isfinite(evaluation_seconds) or not .01<=evaluation_seconds<=60 or not math.isclose(evaluation_seconds*100,round(evaluation_seconds*100)):
        raise ValueError('invalid evaluation duration')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    pins = diagnostic_provenance()
    recipe = dict(schema='cartpole-local-rl-probe-v1', hardware_accepted=False, deployable=False,
        variant=variant, optimizer_seed=seed, training_seeds=list(train), evaluation_seeds=list(dev),
        evaluation_role='development, inspected repeatedly', steps=steps, every=every,
        evaluation_seconds=evaluation_seconds, experiment=experiment,
        experiment_sha256=experiment_identity(experiment), provenance=pins,
        hyperparameters=dict(n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,gamma=.999,
            gae_lambda=.98,learning_rate=.0003,reward_scale=.01,ent_coef=0.,
            net_arch=[64,64],device='cpu',initial_log_std=math.log(.05) if variant=='upright-lowstd' else 0.),
        observation_scaling='fixed state [rail,vmax,1,1,10,amax,1] and ages/.02' if variant=='upright-scaled' else 'none',
        randomization=None, note='Curriculum probe; no production actor or convergence guarantee.')
    _write_new(output/'recipe.json',recipe)
    _source_snapshot(output/'source.zip')
    report=dict(recipe, recipe_sha256=sha256((output/'recipe.json').read_bytes()).hexdigest(),
                source_zip_sha256=sha256((output/'source.zip').read_bytes()).hexdigest(),
                evaluations=[], checkpoints={}, completed=False)
    start_time=time.monotonic()
    torch.set_num_threads(1)
    env=DiagnosticEnv(experiment,train,variant=variant)
    class Episodes(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode_result' in info:
                    with (output/'training-episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(timesteps=int(self.num_timesteps),**info['episode_result']),allow_nan=False)+'\n')
            return True
    def save_report():
        tmp=output/'report.tmp'
        tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        tmp.replace(output/'report.json')
    try:
        model=PPO('MultiInputPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,
            batch_size=batch_size,n_epochs=epochs,gamma=.999,gae_lambda=.98,learning_rate=.0003,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,
                               log_std_init=recipe['hyperparameters']['initial_log_std']))
        model.set_logger(configure(str(output/'logs'),['csv']))
        (output/'checkpoints').mkdir()
        for checkpoint in range(0,steps+1,every):
            if checkpoint:
                model.learn(total_timesteps=every,reset_num_timesteps=checkpoint==every,callback=Episodes(),log_interval=1)
            if diagnostic_provenance()!=pins:
                raise ValueError('source/dependency changed during probe')
            evaluation=evaluate_model(model,experiment,variant=variant,seeds=dev,seconds=evaluation_seconds,
                                      traces=checkpoint==steps)
            evaluation.update(steps=checkpoint,wall_seconds=time.monotonic()-start_time,
                              policy_std=float(model.policy.log_std.exp().item()))
            report['evaluations'].append(evaluation)
            cp=output/'checkpoints'/f'step-{checkpoint:09d}.zip'
            model.save(cp)
            report['checkpoints'][str(cp.relative_to(output))]=sha256(cp.read_bytes()).hexdigest()
            save_report()
            print(json.dumps(dict(variant=variant,seed=seed,steps=checkpoint,by_start=evaluation['by_start'])),flush=True)
        report['completed']=True
        report['wall_seconds']=time.monotonic()-start_time
        save_report()
        return report
    except BaseException as error:
        report['error']=str(error)
        save_report()
        raise
    finally:
        env.close()


def summarize_suite(paths):
    paths=tuple(paths)
    reports=[json.loads((Path(p)/'report.json').read_text()) for p in paths]
    if not reports:
        raise ValueError('at least one pilot required')
    seen=set()
    reference=None
    variants={}
    for path,r in zip(paths,reports):
        if not r.get('completed'):
            raise ValueError('incomplete pilot')
        folder=Path(path)
        recipe_raw=(folder/'recipe.json').read_bytes()
        if sha256(recipe_raw).hexdigest()!=r['recipe_sha256']:
            raise ValueError('recipe hash mismatch')
        recipe=json.loads(recipe_raw)
        if any(r.get(k)!=v for k,v in recipe.items()):
            raise ValueError('report metadata differs from verified recipe')
        if sha256((folder/'source.zip').read_bytes()).hexdigest()!=r['source_zip_sha256']:
            raise ValueError('source archive hash mismatch')
        hyperparameters=dict(r['hyperparameters'])
        expected_std=math.log(.05) if r['variant']=='upright-lowstd' else 0.
        if hyperparameters.pop('initial_log_std')!=expected_std:
            raise ValueError('initial exploration differs from declared variant')
        key=(r['variant'],r['optimizer_seed'])
        if key in seen:
            raise ValueError('duplicate variant/optimizer seed')
        seen.add(key)
        signature=(r['steps'],r['every'],r['evaluation_seconds'],r['experiment_sha256'],
                   tuple(r['training_seeds']),tuple(r['evaluation_seeds']),digest(r['provenance']),
                   digest(hyperparameters),digest(r['randomization']))
        if reference is not None and signature!=reference:
            raise ValueError('incomparable budgets, source, model or seed splits')
        reference=signature
        for name,pin in r['checkpoints'].items():
            if sha256((folder/name).read_bytes()).hexdigest()!=pin:
                raise ValueError('checkpoint hash mismatch')
        episodes_path=folder/'training-episodes.jsonl'
        completed=[json.loads(line) for line in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
        row=dict(optimizer_seed=r['optimizer_seed'],completed_training_episodes=len(completed),
                 unique_completed_training_pose_seeds=len({e['seed'] for e in completed}),
                 final=r['evaluations'][-1]['by_start'],
                 initial=r['evaluations'][0]['by_start'],path=str(path))
        variants.setdefault(r['variant'],dict(optimizer_seeds=[],results=[]))['results'].append(row)
        variants[r['variant']]['optimizer_seeds'].append(r['optimizer_seed'])
    for v in variants.values():
        scores=[r['final']['upright']['balanced_fraction'] for r in v['results']]
        v['upright_balanced_fraction']=dict(median=float(np.median(scores)),minimum=min(scores),maximum=max(scores))
    return dict(schema='cartpole-local-rl-summary-v1',convergence_proven=False,hardware_accepted=False,
                variants=variants,note='Report all seeds. Small diagnostic samples cannot establish long-run convergence or hardware transfer.')
