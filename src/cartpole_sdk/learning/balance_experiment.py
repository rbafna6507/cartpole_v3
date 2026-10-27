"""Local balance-first PPO comparisons. No hardware access or deployment export."""
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from .diagnostics import DiagnosticEnv, preserved_rng, diagnostic_provenance, _source_snapshot
from .model_experiment import validate_experiment
from .model_search import _seeds
from .ppo import _write_new, experiment_identity

GAIN_NAMES = ('position_gain', 'velocity_gain', 'theta_gain', 'omega_gain')


def source_identity():
    return dict(diagnostic_provenance(), balance_experiment_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                projected_ppo_sha256=sha256(Path(__file__).with_name('projected_ppo.py').read_bytes()).hexdigest())


def teacher_gains(value):
    if not isinstance(value, dict) or any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) for k in GAIN_NAMES):
        raise ValueError('four finite teacher gains required')
    return {k: float(value[k]) for k in GAIN_NAMES}


def fit_teacher(model, gains, action_scale, *, updates=1500, samples=8192):
    """Fit only actor weights; preserve critic, exploration and global RNG state."""
    gains = teacher_gains(gains)
    if type(updates) is not int or updates < 1 or type(samples) is not int or samples < 2:
        raise ValueError('positive updates and at least two samples required')
    began = time.monotonic()
    def dataset(n, seed):
        rng = np.random.default_rng(seed)
        q = rng.uniform([-.12, -.3, -.25, -2., -3.], [.12, .3, .25, 2., 3.], (n, 5))
        x, v, theta, omega, acc = q.T
        obs = dict(state=torch.tensor(np.stack((x, v, np.sin(theta), np.cos(theta), omega, acc, np.zeros(n)), axis=1), dtype=torch.float32),
                   valid=torch.ones((n, 7)), age_s=torch.tensor(rng.uniform(0., .02, (n, 7)), dtype=torch.float32))
        targets = np.clip((gains['position_gain']*x + gains['velocity_gain']*v + gains['theta_gain']*theta + gains['omega_gain']*omega)/action_scale, -1, 1)
        return obs, torch.tensor(targets[:, None], dtype=torch.float32)
    train, y = dataset(samples, 51000)
    valid, vy = dataset(max(128, samples//2), 51001)
    actor = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor, lr=.001)
    def rmse():
        with torch.no_grad():
            return float(((model.policy.get_distribution(valid).distribution.mean-vy)*action_scale).square().mean().sqrt())
    initial = rmse()
    best_error, best_weights = initial, [p.detach().clone() for p in actor]
    history = []
    with preserved_rng():
        generator = torch.Generator().manual_seed(51002)
        for step in range(updates):
            indices = torch.randint(samples, (min(512, samples),), generator=generator)
            mean = model.policy.get_distribution({k:v[indices] for k,v in train.items()}).distribution.mean
            loss = ((mean-y[indices])*action_scale).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if (step+1)%100 == 0 or step == updates-1:
                error = rmse()
                history.append(dict(update=step+1, validation_raw_rmse=error))
                if error < best_error:
                    best_error, best_weights = error, [p.detach().clone() for p in actor]
        with torch.no_grad():
            for p, value in zip(actor, best_weights):
                p.copy_(value)
    return dict(updates=updates, train_samples=samples, validation_samples=len(vy),
                train_seed=51000, validation_seed=51001, batch_seed=51002,
                validation_raw_rmse_initial=initial, validation_raw_rmse_final=rmse(),
                selection='minimum synthetic validation RMSE at recorded updates', history=history,
                wall_seconds=time.monotonic()-began)


def evaluate_balance(model, experiment, *, seeds, seconds=25., starts=('upright',), traces=True):
    seeds = _seeds(seeds, 'evaluation seeds')
    if not math.isfinite(seconds) or not .01 <= seconds <= 60 or not math.isclose(seconds*100, round(seconds*100)):
        raise ValueError('evaluation seconds must contain whole 100 Hz decisions in .01..60')
    if not starts or any(s not in ('upright', 'hanging') for s in starts) or len(set(starts)) != len(starts):
        raise ValueError('unique upright/hanging starts required')
    exp = deepcopy(experiment)
    exp['simulation_config']['max_steps'] = round(seconds*100)
    episodes = []
    with preserved_rng():
        env = DiagnosticEnv(exp, seeds, variant='upright')
        try:
            for seed in seeds:
                for start in starts:
                    obs, _ = env.reset(options=dict(episode_seed=seed, start=start))
                    trace = []
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(action)
                        truth = env.core.ground_truth
                        trace.append(dict(t=env.steps/100., x=truth.x, v=truth.v, theta=truth.theta, omega=truth.omega,
                            raw=float(action[0])*env.action_scale, executed=env.core.diagnostics.executed_acceleration_m_s2))
                        if terminated or truncated:
                            break
                    row = env.result(hold_seconds=5.)
                    tail = trace[-500:]
                    x = np.array([v['x'] for v in tail]); theta = np.array([v['theta'] for v in tail])
                    raw = np.array([v['raw'] for v in trace]); executed = np.array([v['executed'] for v in trace])
                    row.update(tail_x_rms_m=float(np.sqrt(np.mean(x*x))), tail_x_mean_m=float(x.mean()),
                        tail_x_peak_to_peak_m=float(np.ptp(x)), tail_x_max_m=float(np.max(np.abs(x))),
                        tail_theta_rms_rad=float(np.sqrt(np.mean(theta*theta))),
                        centering_margin_m=float(.12-np.max(np.abs(x))),
                        raw_action_rms=float(np.sqrt(np.mean(raw*raw))),
                        raw_action_delta_rms=float(np.sqrt(np.mean(np.diff(raw)**2))) if len(raw)>1 else 0.,
                        executed_acceleration_delta_rms=float(np.sqrt(np.mean(np.diff(executed)**2))) if len(executed)>1 else 0.)
                    if traces: row['trajectory'] = trace
                    episodes.append(row)
        finally:
            env.close()
    by_start = {}
    for start in starts:
        rows = [e for e in episodes if e['start']==start]
        by_start[start] = dict(episodes=len(rows), holds=sum(e['hold_success'] for e in rows),
            faults=sum(e['fault_reason'] is not None for e in rows),
            **{k:float(np.mean([e[k] for e in rows])) for k in ('balanced_fraction', 'longest_balance_dwell_s',
              'tail_x_rms_m', 'tail_x_peak_to_peak_m', 'tail_theta_rms_rad', 'centering_margin_m', 'raw_action_delta_rms')},
            worst_max_abs_x_m=max(e['max_abs_x_m'] for e in rows))
    return dict(episodes=episodes, by_start=by_start, seconds=seconds, seeds=list(seeds), starts=list(starts))


def run_trial(experiment, output, *, initialization, seed, gains=None, steps=20480, every=10240,
              training_seeds=tuple(range(12000,12128)), evaluation_seeds=(43000,43001),
              evaluation_seconds=10., n_steps=1024, batch_size=64, epochs=10,
              learning_rate=.00003, policy_std=.02, imitation_updates=1500, constraint='none'):
    validate_experiment(experiment)
    train, dev = _seeds(training_seeds, 'training seeds'), _seeds(evaluation_seeds, 'evaluation seeds')
    if set(train)&set(dev): raise ValueError('training/evaluation seed overlap')
    if initialization not in ('scratch', 'teacher'): raise ValueError('unknown initialization')
    if constraint not in ('none', 'equilibrium'): raise ValueError('unknown learning constraint')
    if type(seed) is not int or seed < 0: raise ValueError('nonnegative optimizer seed required')
    if any(type(v) is not int or v < 1 for v in (steps, every, n_steps, batch_size, epochs, imitation_updates)):
        raise ValueError('positive integer budgets required')
    if min(n_steps,batch_size)<2 or n_steps%batch_size or every%n_steps or steps%every:
        raise ValueError('budgets must divide evaluation/rollout/batch intervals')
    if any(not math.isfinite(v) or v<=0 for v in (learning_rate,policy_std)):
        raise ValueError('positive finite learning rate and policy std required')
    if not math.isfinite(evaluation_seconds) or not .01<=evaluation_seconds<=60 or not math.isclose(evaluation_seconds*100, round(evaluation_seconds*100)):
        raise ValueError('invalid evaluation seconds')
    gains = teacher_gains(gains) if gains is not None else None
    if initialization=='teacher' and gains is None: raise ValueError('teacher initialization requires gains')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pins = source_identity()
    recipe = dict(schema='cartpole-balance-learning-v1', initialization=initialization, optimizer_seed=seed,
        constraint=constraint, constraint_semantics=('Project initial actor and after every PPO rollout optimizer update to zero mean action at state[0,0,0,1,0,0,0],valid=ones,age_s=zeros; ordinary MLP inference. Constrained PPO, no trust-region guarantee after projection.' if constraint=='equilibrium' else 'Vanilla PPO'),
        teacher_gains=gains, steps=steps, every=every, training_seeds=list(train), evaluation_seeds=list(dev),
        evaluation_seconds=evaluation_seconds, evaluation_role='development, inspected repeatedly',
        experiment=experiment, experiment_sha256=experiment_identity(experiment), provenance=pins,
        hyperparameters=dict(n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,learning_rate=learning_rate,
            gamma=.999,gae_lambda=.98,policy_std=policy_std,reward_scale=.01,net_arch=[64,64],device='cpu',threads=1),
        imitation_updates=imitation_updates if initialization=='teacher' else 0,
        hardware_accepted=False,deployable=False,randomization=None,
        note='Equal online interaction budgets. Teacher initialization has additional supervised compute. No teacher fallback.')
    _write_new(output/'recipe.json', recipe)
    _source_snapshot(output/'source.zip')
    root = Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(output/'source.zip','a',zipfile.ZIP_DEFLATED) as z:
        for name in ('examples/balance_learning.py','scripts/balance-learning'):
            if (root/name).exists(): z.write(root/name,name)
    report = dict(recipe, recipe_sha256=sha256((output/'recipe.json').read_bytes()).hexdigest(),
        source_zip_sha256=sha256((output/'source.zip').read_bytes()).hexdigest(), evaluations=[],completed=False)
    torch.set_num_threads(1)
    env = DiagnosticEnv(experiment, train, variant='upright')
    began = time.monotonic()
    class Episodes(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode_result' in info:
                    with (output/'training-episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(timesteps=int(self.num_timesteps),**info['episode_result']), allow_nan=False)+'\n')
            return True
    def save():
        tmp=output/'report.tmp'; tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); tmp.replace(output/'report.json')
    try:
        from .projected_ppo import EquilibriumPPO
        algorithm=EquilibriumPPO if constraint=='equilibrium' else PPO
        model=algorithm('MultiInputPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,batch_size=batch_size,
            n_epochs=epochs,gamma=.999,gae_lambda=.98,learning_rate=learning_rate,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=math.log(policy_std)))
        model.set_logger(configure(str(output/'logs'),['csv']))
        if initialization=='teacher':
            report['imitation']=fit_teacher(model,gains,env.action_scale,updates=imitation_updates)
        if constraint=='equilibrium':
            report['initial_projection_normalized']=model.project_equilibrium()
        (output/'checkpoints').mkdir()
        for checkpoint in range(0,steps+1,every):
            if checkpoint:
                model.learn(total_timesteps=every,reset_num_timesteps=checkpoint==every,callback=Episodes(),log_interval=1)
                # SB3 normally logs before train(); flush the completed update too.
                model.logger.record('time/total_timesteps', int(model.num_timesteps))
                model.logger.dump(step=int(model.num_timesteps))
            if source_identity()!=pins: raise ValueError('source/dependency changed during trial')
            cp=output/'checkpoints'/f'step-{checkpoint:09d}.zip'
            model.save(cp)
            result=evaluate_balance(model,experiment,seeds=dev,seconds=evaluation_seconds)
            from .projected_ppo import equilibrium_observation
            result['equilibrium_raw_action']=float(model.predict(equilibrium_observation(),deterministic=True)[0][0])*env.action_scale
            result.update(steps=int(model.num_timesteps),checkpoint=str(cp.relative_to(output)),
                checkpoint_sha256=sha256(cp.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-began,
                policy_std=float(model.policy.log_std.exp().item()))
            report['evaluations'].append(result); save()
            print(json.dumps(dict(initialization=initialization,seed=seed,steps=checkpoint,summary=result['by_start'])),flush=True)
        report.update(completed=True,wall_seconds=time.monotonic()-began); save()
        return report
    except BaseException as exc:
        report['error']=str(exc);save();raise
    finally:
        env.close()


def evaluate_saved(run, output, *, seeds, seconds=60., checkpoint=None, starts=('upright',)):
    run, output = Path(run), Path(output)
    report=json.loads((run/'report.json').read_text())
    raw=(run/'recipe.json').read_bytes();recipe=json.loads(raw)
    if sha256(raw).hexdigest()!=report['recipe_sha256'] or any(report.get(k)!=v for k,v in recipe.items()):
        raise ValueError('recipe hash or metadata mismatch')
    if sha256((run/'source.zip').read_bytes()).hexdigest()!=report['source_zip_sha256']:
        raise ValueError('source archive hash mismatch')
    if source_identity()!=recipe['provenance']: raise ValueError('source/dependency changed since training')
    seeds=_seeds(seeds,'evaluation seeds')
    if set(seeds)&set(recipe['training_seeds']+recipe['evaluation_seeds']): raise ValueError('evaluation seed overlap')
    selected=report['evaluations'][-1] if checkpoint is None else next((e for e in report['evaluations'] if e['steps']==checkpoint),None)
    if selected is None: raise ValueError('checkpoint not found')
    cp=run/selected['checkpoint']
    if cp.parent.resolve()!=(run/'checkpoints').resolve() or sha256(cp.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:
        raise ValueError('checkpoint hash or path mismatch')
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    with preserved_rng():
        model=PPO.load(cp,device='cpu')
        result=evaluate_balance(model,recipe['experiment'],seeds=seeds,seconds=seconds,starts=starts)
    result.update(checkpoint_sha256=selected['checkpoint_sha256'],steps=selected['steps'],run=str(run),
        provenance=source_identity(),hardware_accepted=False,deployable=False,
        role='Held out from this run; caller must track prior inspection across other runs.')
    output.mkdir(parents=True,exist_ok=False)
    _write_new(output/'evaluation.json',result)
    return result
