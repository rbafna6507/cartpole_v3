"""Offline balance learning with explicit actor-input angle errors.

This perturbation is not a calibrated encoder law. Physics, telemetry, truth
reward, validity, ages and angular rate retain their existing definitions.
No hardware API, SDK actor export, or acceptance transition is provided.
"""
import argparse
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

from .balance_experiment import fit_teacher, source_identity as balance_source_identity
from .diagnostics import DiagnosticEnv, preserved_rng, _source_snapshot
from .model_experiment import build_experiment, validate_experiment
from .model_search import _seeds
from .ppo import _write_new, experiment_identity
from .projected_ppo import EquilibriumPPO

EVALUATION_BIASES_DEG = (0., -.5, .5, -1., 1.)
TEACHER_GAINS = dict(position_gain=1., velocity_gain=2., theta_gain=40., omega_gain=2.)


def _bias_degrees(value, *, signed=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            not (-3. if signed else 0.) <= value <= 3.):
        raise ValueError('angle bias must be finite and within '+('±3' if signed else '0..3')+' degrees')
    return float(value)


def _seconds(value):
    if (type(value) not in (int, float) or not math.isfinite(value) or not .01 <= value <= 60 or
            not math.isclose(value*100, round(value*100))):
        raise ValueError('seconds must contain whole 100 Hz decisions in .01..60')


def bias_observation(observation, bias_rad):
    """Rotate valid quantized sin/cos only; never edit an input array in place."""
    if (type(bias_rad) not in (int, float) or not math.isfinite(bias_rad) or
            abs(bias_rad) > math.radians(3.)):
        raise ValueError('actor angle bias must be finite and within ±3 degrees')
    result = {key: np.array(value, copy=True) for key, value in observation.items()}
    if bias_rad and result['valid'][2] and result['valid'][3]:
        angle = math.atan2(float(result['state'][2]), float(result['state'][3]))+bias_rad
        result['state'][2:4] = math.sin(angle), math.cos(angle)
    return result


class AngleErrorEnv(DiagnosticEnv):
    """Private training RNG draws a constant actor-input offset each episode."""
    def __init__(self, experiment, seeds, *, bias_degrees, bias_seed, fixed_bias_degrees=None):
        self.bias_degrees = _bias_degrees(bias_degrees)
        if type(bias_seed) is not int or bias_seed < 0:
            raise ValueError('bias RNG seed must be a nonnegative integer')
        self.bias_seed = bias_seed
        self.fixed_bias_degrees = (None if fixed_bias_degrees is None else
                                   _bias_degrees(fixed_bias_degrees, signed=True))
        self._bias_rng = np.random.default_rng(bias_seed)
        self.actor_angle_bias_rad = 0.
        super().__init__(experiment, seeds, variant='upright')

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._bias_rng = np.random.default_rng(np.random.SeedSequence([self.bias_seed, seed]))
        offset = (self._bias_rng.uniform(-self.bias_degrees, self.bias_degrees)
                  if self.fixed_bias_degrees is None else self.fixed_bias_degrees)
        self.actor_angle_bias_rad = math.radians(offset)
        observation, info = super().reset(seed=seed, options=options)
        return bias_observation(observation, self.actor_angle_bias_rad), dict(info,
            actor_angle_bias_rad=self.actor_angle_bias_rad)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info, actor_angle_bias_rad=self.actor_angle_bias_rad)
        if 'episode_result' in info:
            info['episode_result'] = dict(info['episode_result'], actor_angle_bias_rad=self.actor_angle_bias_rad)
        return bias_observation(observation, self.actor_angle_bias_rad), reward, terminated, truncated, info


def _file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[3]
    return dict(balance_source_identity(), error_training_sha256=_file_hash(__file__),
                entrypoint_sha256=_file_hash(root/'scripts/balance-error-learning'))


def load_profile(profile, *, observer_bw=None, seconds=5.):
    """Build a new admitted experiment; original profile bytes stay untouched."""
    if observer_bw is not None and (type(observer_bw) not in (int, float) or observer_bw not in (6, 8, 10)):
        raise ValueError('observer bandwidth override must be 6, 8 or 10 Hz')
    profile = Path(profile).resolve()
    values = {name: json.loads((profile/f'{name}.json').read_text()) for name in
              ('executor-parameters', 'observation-config', 'prior-parameters')}
    if observer_bw is not None:
        values['executor-parameters']['bw'] = float(observer_bw)
    experiment = build_experiment(profile/'model.json', executor_parameters=values['executor-parameters'],
        observation_config=values['observation-config'], prior_parameters=values['prior-parameters'],
        development_only=True, seconds=seconds)
    return experiment, {str(p): _file_hash(p) for p in sorted(profile.glob('*.json'))}


def _weights_hash(model):
    digest = sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate_errors(model, experiment, *, seeds, seconds=10.):
    """Fixed bias matrix on independent pose seeds; all quality metrics use truth."""
    validate_experiment(experiment)
    seeds = _seeds(seeds, 'evaluation seeds'); _seconds(seconds)
    exp = deepcopy(experiment); exp['simulation_config']['max_steps'] = round(seconds*100)
    episodes = []
    with preserved_rng():
        for bias in EVALUATION_BIASES_DEG:
            env = AngleErrorEnv(exp, seeds, bias_degrees=0., bias_seed=0, fixed_bias_degrees=bias)
            try:
                for seed in seeds:
                    obs, _ = env.reset(options=dict(episode_seed=seed, start='upright'))
                    trajectory = []
                    while True:
                        action, _ = model.predict(obs, deterministic=True)
                        observed_angle = math.atan2(float(obs['state'][2]), float(obs['state'][3]))
                        before_ns = env.core.ground_truth.time_ns
                        obs, _, terminated, truncated, _ = env.step(action)
                        truth = env.core.ground_truth
                        trajectory.append(dict(t=env.steps/100., x=truth.x, v=truth.v, theta=truth.theta,
                            omega=truth.omega, actor_angle_before_rad=observed_angle,
                            raw=float(action[0])*env.action_scale,
                            executed=env.core.diagnostics.executed_acceleration_m_s2,
                            physical_time_advanced=truth.time_ns > before_ns))
                        if terminated or truncated: break
                    advanced = [t for t in trajectory if t['physical_time_advanced']]
                    # Initial invalid input has no advanced samples; report null RMS.
                    tail = advanced[-500:]
                    def rms(rows, key):
                        return float(np.sqrt(np.mean([t[key]**2 for t in rows]))) if rows else None
                    episode = env.result(hold_seconds=5.)
                    episode.update(bias_degrees=bias, trajectory=trajectory,
                        proposed_decisions=len(trajectory), advanced_decisions=len(advanced),
                        true_x_rms_m=rms(advanced, 'x'), true_theta_rms_rad=rms(advanced, 'theta'),
                        tail_true_x_rms_m=rms(tail, 'x'), tail_true_theta_rms_rad=rms(tail, 'theta'),
                        continuous_full_balance=bool(episode['fault_reason'] is None and
                            episode['steps']==exp['simulation_config']['max_steps'] and
                            all(abs(t['x'])<.12 and abs(t['theta'])<.15 and abs(t['omega'])<1 for t in advanced)))
                    episodes.append(episode)
            finally: env.close()
    summary = []
    for bias in EVALUATION_BIASES_DEG:
        rows = [e for e in episodes if e['bias_degrees']==bias]
        def mean_metric(key):
            values = [r[key] for r in rows]
            return float(np.mean(values)) if all(v is not None for v in values) else None
        summary.append(dict(bias_degrees=bias, episodes=len(rows), holds=sum(e['hold_success'] for e in rows),
            continuous_full_balance=sum(e['continuous_full_balance'] for e in rows),
            faults=sum(e['fault_reason'] is not None for e in rows), worst_max_abs_x_m=max(e['max_abs_x_m'] for e in rows),
            **{key:mean_metric(key) for key in ('true_x_rms_m','true_theta_rms_rad','tail_true_x_rms_m','tail_true_theta_rms_rad')}))
    return dict(seeds=list(seeds), seconds=seconds, biases_degrees=list(EVALUATION_BIASES_DEG),
                episodes=episodes, summary=summary)


def run_trial(experiment, output, *, seed, bias_degrees, steps=4096, every=4096,
              training_seeds=tuple(range(13000,13128)), evaluation_seeds=(66000,66001),
              evaluation_seconds=10., n_steps=1024, batch_size=64, epochs=10,
              learning_rate=3e-6, policy_std=.02, imitation_updates=1500, input_files=None):
    validate_experiment(experiment); bias_degrees = _bias_degrees(bias_degrees); _seconds(evaluation_seconds)
    train, dev = _seeds(training_seeds, 'training seeds'), _seeds(evaluation_seeds, 'evaluation seeds')
    if set(train)&set(dev): raise ValueError('training/evaluation seed overlap')
    if type(seed) is not int or seed < 0: raise ValueError('nonnegative optimizer seed required')
    if any(type(v) is not int or v<1 for v in (steps,every,n_steps,batch_size,epochs,imitation_updates)):
        raise ValueError('positive integer budgets required')
    if min(n_steps,batch_size)<2 or n_steps%batch_size or every%n_steps or steps%every:
        raise ValueError('budgets must divide rollout, batch and evaluation intervals')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (learning_rate,policy_std)):
        raise ValueError('positive finite learning rate and policy std required')
    experiment = deepcopy(experiment)
    inputs = dict(input_files or {experiment['admission']['model']['source_path']: experiment['admission']['model']['raw_sha256']})
    def verify_inputs():
        if any(_file_hash(p)!=pin for p,pin in inputs.items()): raise ValueError('profile input hash changed')
    verify_inputs()
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    pins = source_identity()
    recipe = dict(schema='cartpole-angle-error-learning-v1',status='DEVELOPMENT_ONLY',simulation_only=True,
        hardware_accepted=False,deployable=False,optimizer_seed=seed,bias_rng_seed=seed,
        actor_input_bias=dict(distribution='uniform per episode',amplitude_degrees=bias_degrees,
            semantics='Rotate only valid actor sin/cos after quantization; rate, validity, ages, true plant and true reward unchanged. Not a calibrated sensor law.'),
        teacher_gains=dict(TEACHER_GAINS),imitation_updates=imitation_updates,constraint='equilibrium',
        reward='Existing TrainingEnv ground-truth balance/dwell objective, scaled .01; no new RMS objective.',
        steps=steps,every=every,training_seeds=list(train),evaluation_seeds=list(dev),evaluation_seconds=evaluation_seconds,
        evaluation_biases_degrees=list(EVALUATION_BIASES_DEG),evaluation_role='development, repeatedly inspected; not heldout selection',
        experiment=experiment,experiment_sha256=experiment_identity(experiment),provenance=pins,input_files=inputs,
        hyperparameters=dict(n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,learning_rate=learning_rate,
            gamma=.999,gae_lambda=.98,policy_std=policy_std,net_arch=[64,64],device='cpu',threads=1),
        limitations=['Memoryless actor may not distinguish constant angle bias from true tilt.',
            'Frozen passive dynamics and declared noise/delay remain exploratory; no new identified nonlinear model.',
            'Observer bandwidth belongs to this experiment; future SDK deployment must still match the complete live executor.'])
    _write_new(output/'recipe.json',recipe); _source_snapshot(output/'source.zip')
    root=Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(output/'source.zip','a',zipfile.ZIP_DEFLATED) as archive:
        archive.write(root/'scripts/balance-error-learning','scripts/balance-error-learning')
        for index,p in enumerate(inputs): archive.write(p,f'frozen-inputs/{index}-{Path(p).name}')
    report=dict(recipe,recipe_sha256=_file_hash(output/'recipe.json'),source_zip_sha256=_file_hash(output/'source.zip'),
                completed=False,evaluations=[])
    def save():
        tmp=output/'report.tmp';tmp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    torch.set_num_threads(1)
    env=AngleErrorEnv(experiment,train,bias_degrees=bias_degrees,bias_seed=seed)
    began=time.monotonic()
    class Episodes(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode_result' in info:
                    with (output/'training-episodes.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(timesteps=int(self.num_timesteps),**info['episode_result']),allow_nan=False)+'\n')
            return True
    try:
        model=EquilibriumPPO('MultiInputPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,
            batch_size=batch_size,n_epochs=epochs,gamma=.999,gae_lambda=.98,learning_rate=learning_rate,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=math.log(policy_std)))
        model.set_logger(configure(str(output/'logs'),['csv']))
        report['imitation']=fit_teacher(model,TEACHER_GAINS,env.action_scale,updates=imitation_updates)
        report['initial_projection_normalized']=model.project_equilibrium()
        (output/'checkpoints').mkdir()
        for checkpoint in range(0,steps+1,every):
            if checkpoint:
                model.learn(total_timesteps=every,reset_num_timesteps=checkpoint==every,callback=Episodes(),log_interval=1)
                model.logger.record('time/total_timesteps',int(model.num_timesteps));model.logger.dump(step=int(model.num_timesteps))
            verify_inputs()
            if source_identity()!=pins: raise ValueError('source/dependency changed during trial')
            cp=output/'checkpoints'/f'step-{checkpoint:09d}.zip';model.save(cp)
            before=_weights_hash(model)
            evaluation=evaluate_errors(model,experiment,seeds=dev,seconds=evaluation_seconds)
            if before!=_weights_hash(model): raise ValueError('evaluation changed policy weights')
            evaluation.update(steps=int(model.num_timesteps),checkpoint=str(cp.relative_to(output)),
                checkpoint_sha256=_file_hash(cp),policy_weights_sha256=before,wall_seconds=time.monotonic()-began)
            report['evaluations'].append(evaluation);save()
            print(json.dumps(dict(seed=seed,bias_degrees=bias_degrees,steps=checkpoint,summary=evaluation['summary'])),flush=True)
        verify_inputs()
        if source_identity()!=pins: raise ValueError('source/dependency changed during final evaluation')
        report.update(completed=True,wall_seconds=time.monotonic()-began);save();return report
    except BaseException as exc:
        report['error']=str(exc);save();raise
    finally:env.close()


def evaluate_saved(run, output, *, seeds, seconds=10., checkpoint=None):
    run,output=Path(run),Path(output)
    report=json.loads((run/'report.json').read_text());recipe=json.loads((run/'recipe.json').read_text())
    if (recipe.get('schema')!='cartpole-angle-error-learning-v1' or not report.get('completed') or
            _file_hash(run/'recipe.json')!=report['recipe_sha256'] or any(report.get(k)!=v for k,v in recipe.items()) or
            _file_hash(run/'source.zip')!=report['source_zip_sha256'] or
            experiment_identity(recipe['experiment'])!=recipe['experiment_sha256']):
        raise ValueError('recipe/source hash or completion mismatch')
    if source_identity()!=recipe['provenance']: raise ValueError('training source/dependency changed')
    seeds=_seeds(seeds,'evaluation seeds');_seconds(seconds)
    if set(seeds)&set(recipe['training_seeds']+recipe['evaluation_seeds']): raise ValueError('evaluation seed overlap')
    selected=report['evaluations'][-1] if checkpoint is None else next((e for e in report['evaluations'] if e['steps']==checkpoint),None)
    if selected is None: raise ValueError('checkpoint missing')
    cp=run/selected['checkpoint']
    if cp.parent.resolve()!=(run/'checkpoints').resolve() or _file_hash(cp)!=selected['checkpoint_sha256']:
        raise ValueError('checkpoint hash or path mismatch')
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(1)
    with preserved_rng():
        model=PPO.load(cp,device='cpu')
        if _weights_hash(model)!=selected['policy_weights_sha256']: raise ValueError('loaded weight hash mismatch')
        result=evaluate_errors(model,recipe['experiment'],seeds=seeds,seconds=seconds)
    result.update(checkpoint_sha256=selected['checkpoint_sha256'],steps=selected['steps'],run=str(run),
        source=source_identity(),hardware_accepted=False,deployable=False,
        role='New poses for this run; caller tracks prior inspection across experiments.')
    output.mkdir(parents=True,exist_ok=False);_write_new(output/'evaluation.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train',help='New teacher-initialized offline trial; no actor export.')
    train.add_argument('--profile',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    train.add_argument('--seed',type=int,required=True);train.add_argument('--bias-degrees',type=float,required=True)
    train.add_argument('--observer-bw',type=int,choices=(6,8,10))
    for name,default in (('steps',4096),('every',4096),('imitation-updates',1500)):
        train.add_argument('--'+name,type=int,default=default)
    train.add_argument('--seconds',type=float,default=5.);train.add_argument('--evaluation-seconds',type=float,default=10.)
    train.add_argument('--training-seeds',type=int,nargs='+',default=list(range(13000,13128)))
    train.add_argument('--evaluation-seeds',type=int,nargs='+',default=[66000,66001])
    train.add_argument('--learning-rate',type=float,default=3e-6);train.add_argument('--policy-std',type=float,default=.02)
    evaluate=sub.add_parser('evaluate',help='Fixed angle-error matrix on new independent pose seeds.')
    evaluate.add_argument('run',type=Path);evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--seeds',type=int,nargs='+',required=True);evaluate.add_argument('--seconds',type=float,default=10.)
    evaluate.add_argument('--checkpoint',type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=='train':
            exp,inputs=load_profile(args.profile,observer_bw=args.observer_bw,seconds=args.seconds)
            run_trial(exp,args.output,seed=args.seed,bias_degrees=args.bias_degrees,steps=args.steps,every=args.every,
                training_seeds=args.training_seeds,evaluation_seeds=args.evaluation_seeds,evaluation_seconds=args.evaluation_seconds,
                learning_rate=args.learning_rate,policy_std=args.policy_std,imitation_updates=args.imitation_updates,input_files=inputs)
        else:evaluate_saved(args.run,args.output,seeds=args.seeds,seconds=args.seconds,checkpoint=args.checkpoint)
    except (ValueError,OSError) as exc:parser.exit(2,f'angle-error learning: {exc}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
