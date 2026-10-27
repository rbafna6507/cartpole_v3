"""Offline swing-up/capture learning. No hardware access or acceptance claim."""
from copy import deepcopy
from hashlib import sha256
import json
import io
import math
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from .ppo import TrainingEnv, _write_new, experiment_identity
from .diagnostics import preserved_rng, _source_snapshot
from .balance_experiment import evaluate_balance, fit_teacher, source_identity as balance_identity
from .projected_ppo import EquilibriumPPO
from .model_experiment import validate_experiment
from .model_search import _seeds


class SwingEnv(TrainingEnv):
    """Mixed observed-state task; curriculum adds deterministic intermediate poses."""
    def __init__(self, experiment, seeds, *, reset_mode='mixed', curriculum_steps=None):
        if reset_mode not in ('mixed','curriculum'):
            raise ValueError('unknown reset mode')
        if curriculum_steps is not None and (type(curriculum_steps) is not int or curriculum_steps<1):raise ValueError('positive curriculum budget required')
        self.curriculum_steps,self.total_training_steps=curriculum_steps,0
        self.reset_mode, self.stage = reset_mode, 0
        super().__init__(experiment,seeds,reward_scale=.01)

    def set_stage(self, stage):
        if type(stage) is not int or not 0<=stage<=2:
            raise ValueError('stage must be 0, 1 or 2')
        self.stage=stage

    def initial_state(self, episode_seed, start):
        if start!='capture':return super().initial_state(episode_seed,start)
        rng=np.random.default_rng(np.random.SeedSequence([episode_seed,self.stage,991]))
        limit=(self.stage+1)*math.pi/3
        return (float(rng.uniform(-.03,.03)),float(rng.uniform(-.1,.1)),
                float(rng.uniform(-limit,limit)),float(rng.uniform(-.5,.5)))

    def reset(self, *, seed=None, options=None):
        if seed is not None:self._episode_index=0
        if options is None and self.reset_mode=='curriculum':
            family=('upright','capture','hanging','capture')[self._episode_index%4]
            options=dict(episode_seed=self.seeds[(self._episode_index//4)%len(self.seeds)],start=family)
        observation,info=super().reset(seed=seed,options=options)
        self.episode_stage=self.stage
        info['curriculum_stage']=self.stage
        return observation,info


    def step(self, action):
        result=super().step(action)
        self.total_training_steps+=1
        if self.curriculum_steps is not None:
            self.set_stage(min(2,self.total_training_steps*3//self.curriculum_steps))
        return result

    def result(self, *, hold_seconds=5.):
        return dict(super().result(hold_seconds=hold_seconds),reset_curriculum_stage=self.episode_stage)


def load_demonstrations(path, *, train_episodes, validation_episodes):
    train=_seeds(train_episodes,'demonstration training episodes')
    valid=_seeds(validation_episodes,'demonstration validation episodes')
    if set(train)&set(valid):raise ValueError('demonstration episode overlap')
    with np.load(io.BytesIO(path) if isinstance(path,bytes) else path,allow_pickle=False) as z:
        if set(z.files)!={'state','valid','age_s','raw_action','episode'}:
            raise ValueError('demonstrations require observed channels, raw action and episode IDs')
        arrays={k:np.asarray(z[k]).copy() for k in z.files}
    n=len(arrays['raw_action'])
    if any(arrays[k].shape!=(n,7) for k in ('state','valid','age_s')) or any(arrays[k].shape!=(n,) for k in ('raw_action','episode')):
        raise ValueError('invalid demonstration shape')
    if not all(np.issubdtype(v.dtype,np.number) and np.all(np.isfinite(v)) for v in arrays.values()):
        raise ValueError('demonstrations must contain finite numbers')
    if not np.issubdtype(arrays['episode'].dtype,np.integer) or np.any(arrays['episode']<0):
        raise ValueError('episode IDs must be nonnegative integers')
    if np.any(np.abs(arrays['raw_action'])>25) or np.any(arrays['age_s']<0) or not np.all(arrays['valid']==1):
        raise ValueError('demonstrations require valid causal channels and bounded raw actions')
    result={}
    for name,ids in [('train',train),('validation',valid)]:
        if not set(ids)<=set(arrays['episode']):raise ValueError('missing demonstration episode')
        mask=np.isin(arrays['episode'],ids)
        result[name]={k:v[mask] for k,v in arrays.items() if k!='episode'}
    return result


def fit_demonstrations(model, data, action_scale, *, updates=3000, balance_anchor_weight=0.):
    """Supervise actor only, with whole-trajectory held-out rows supplied separately."""
    if type(updates) is not int or updates<1:raise ValueError('positive imitation updates required')
    if not math.isfinite(balance_anchor_weight) or balance_anchor_weight<0:raise ValueError('nonnegative finite balance anchor weight required')
    actor=list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters())
    optimizer=torch.optim.Adam(actor,lr=.001)
    def convert(rows):
        obs={k:torch.as_tensor(rows[k],dtype=torch.float32) for k in ('state','valid','age_s')}
        y=torch.as_tensor(rows['raw_action']/action_scale,dtype=torch.float32)[:,None]
        return obs,y
    train,y=convert(data['train']);valid,vy=convert(data['validation'])
    def error():
        with torch.no_grad():return float(((model.policy.get_distribution(valid).distribution.mean-vy)*action_scale).square().mean().sqrt())
    def anchor_rows(n,seed):
        rng=np.random.default_rng(seed)
        x,v,theta,omega,acc=rng.uniform([-.12,-.3,-.25,-2.,-3.],[.12,.3,.25,2.,3.],(n,5)).T
        return dict(state=np.stack((x,v,np.sin(theta),np.cos(theta),omega,acc,np.zeros(n)),axis=1),
                    valid=np.ones((n,7)),age_s=rng.uniform(0.,.02,(n,7)),raw_action=np.clip(x+2*v+40*theta+2*omega,-action_scale,action_scale))
    anchor,ay=convert(anchor_rows(8192,51100));anchor_valid,avy=convert(anchor_rows(4096,51101))
    def anchor_error():
        with torch.no_grad():return float(((model.policy.get_distribution(anchor_valid).distribution.mean-avy)*action_scale).square().mean().sqrt())
    initial=error();anchor_initial=anchor_error()
    best=initial**2+balance_anchor_weight*anchor_initial**2
    weights=[p.detach().clone() for p in actor];history=[]
    with preserved_rng():
        generator=torch.Generator().manual_seed(61002)
        for step in range(updates):
            ix=torch.randint(len(y),(min(512,len(y)),),generator=generator)
            mean=model.policy.get_distribution({k:v[ix] for k,v in train.items()}).distribution.mean
            loss=((mean-y[ix])*action_scale).square().mean()
            if balance_anchor_weight:
                ai=torch.randint(len(ay),(512,),generator=generator)
                amean=model.policy.get_distribution({k:v[ai] for k,v in anchor.items()}).distribution.mean
                loss=loss+balance_anchor_weight*((amean-ay[ai])*action_scale).square().mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            if (step+1)%100==0 or step==updates-1:
                demo_error=error();balance_error=anchor_error()
                score=demo_error**2+balance_anchor_weight*balance_error**2
                history.append(dict(update=step+1,validation_raw_rmse=demo_error,balance_anchor_rmse=balance_error,selection_score=score))
                if score<best:best=score;weights=[p.detach().clone() for p in actor]
        with torch.no_grad():
            for p,w in zip(actor,weights):p.copy_(w)
    return dict(initial_rmse=initial,final_rmse=error(),updates=updates,
                train_samples=len(y),validation_samples=len(vy),history=history,
                balance_anchor=dict(weight=balance_anchor_weight,initial_rmse=anchor_initial,final_rmse=anchor_error(),
                                    gains=dict(position=1.,velocity=2.,theta=40.,omega=2.),train_seed=51100,validation_seed=51101),
                selection='minimum demonstration validation MSE + weight * synthetic balance validation MSE at recorded updates')


def source_identity():
    return dict(balance_identity(),swing_experiment_sha256=sha256(Path(__file__).read_bytes()).hexdigest())


def run_trial(experiment, output, *, initialization, seed, steps=20480, every=10240,
              reset_mode='mixed',learning_rate=.000003,policy_std=.05,
              training_seeds=tuple(range(20000,20128)),evaluation_seeds=(53000,53001),
              evaluation_seconds=25.,demonstrations=None,train_episodes=(),validation_episodes=(),
              imitation_updates=3000,balance_anchor_weight=0.,n_steps=1024,batch_size=64,epochs=10):
    validate_experiment(experiment)
    train=_seeds(training_seeds,'training poses');dev=_seeds(evaluation_seeds,'development poses')
    if set(train)&set(dev):raise ValueError('pose seed overlap')
    if initialization not in ('scratch','balance','demonstrations'):raise ValueError('unknown initialization')
    if reset_mode not in ('mixed','curriculum'):raise ValueError('unknown reset mode')
    if type(seed) is not int or seed<0:raise ValueError('nonnegative optimizer seed required')
    if any(type(v) is not int or v<1 for v in (steps,every,n_steps,batch_size,epochs,imitation_updates)) or min(n_steps,batch_size)<2 or n_steps%batch_size or every%n_steps or steps%every:
        raise ValueError('invalid or indivisible budgets')
    if any(not math.isfinite(v) or v<=0 for v in (learning_rate,policy_std)):raise ValueError('positive finite optimizer settings required')
    if not math.isfinite(evaluation_seconds) or not .01<=evaluation_seconds<=60 or not math.isclose(evaluation_seconds*100,round(evaluation_seconds*100)):
        raise ValueError('invalid evaluation duration')
    if not math.isfinite(balance_anchor_weight) or balance_anchor_weight<0:raise ValueError('nonnegative finite balance anchor weight required')
    if balance_anchor_weight and initialization!='demonstrations':raise ValueError('balance anchor requires demonstrations')
    data=None;demonstration_bytes=None
    if initialization=='demonstrations':
        if demonstrations is None:raise ValueError('demonstrations required')
        if (set(train_episodes)|set(validation_episodes)) & set(dev):raise ValueError('demonstration/development overlap')
        demonstration_bytes=Path(demonstrations).read_bytes()
        data=load_demonstrations(demonstration_bytes,train_episodes=train_episodes,validation_episodes=validation_episodes)
    elif demonstrations is not None:raise ValueError('unexpected demonstrations')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    pins=source_identity();torch.set_num_threads(1)
    recipe=dict(schema='cartpole-swing-learning-v1',experiment=experiment,experiment_sha256=experiment_identity(experiment),
        initialization=initialization,optimizer_seed=seed,steps=steps,every=every,reset_mode=reset_mode,
        training_seeds=list(train),evaluation_seeds=list(dev),evaluation_seconds=evaluation_seconds,
        evaluation_role='development, inspected repeatedly',provenance=pins,
        hyperparameters=dict(learning_rate=learning_rate,policy_std=policy_std,n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,
                             gamma=.999,gae_lambda=.98,net_arch=[64,64],reward_scale=.01,device='cpu',threads=1),
        constraint='zero action at upright equilibrium after initialization and each PPO update',
        curriculum='If enabled: 1/4 upright, 1/4 hanging, 1/2 intermediate; angle widths pi/3,2pi/3,pi by online-budget thirds; changes apply at next reset.',
        demonstrations=None,hardware_accepted=False,deployable=False)
    if data is not None:
        raw=demonstration_bytes;(output/'demonstrations.npz').write_bytes(raw)
        recipe['demonstrations']=dict(sha256=sha256(raw).hexdigest(),train_episodes=list(train_episodes),validation_episodes=list(validation_episodes),updates=imitation_updates,balance_anchor_weight=balance_anchor_weight)
    _write_new(output/'recipe.json',recipe);_source_snapshot(output/'source.zip')
    root=Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(output/'source.zip','a',zipfile.ZIP_DEFLATED) as z:
        for name in ('examples/swing_learning.py','scripts/swing-learning'):
            if (root/name).is_file():z.write(root/name,name)
    report=dict(recipe,recipe_sha256=sha256((output/'recipe.json').read_bytes()).hexdigest(),source_zip_sha256=sha256((output/'source.zip').read_bytes()).hexdigest(),evaluations=[],completed=False)
    def save():
        tmp=output/'report.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output/'report.json')
    env=SwingEnv(experiment,train,reset_mode=reset_mode,curriculum_steps=steps);began=time.monotonic()
    class Episodes(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode_result' in info:
                    with (output/'training-episodes.jsonl').open('a') as stream:stream.write(json.dumps(dict(timesteps=int(self.num_timesteps),**info['episode_result']),allow_nan=False)+'\n')
            return True
    try:
        model=EquilibriumPPO('MultiInputPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,
            gamma=.999,gae_lambda=.98,learning_rate=learning_rate,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=math.log(policy_std)))
        model.set_logger(configure(str(output/'logs'),['csv']))
        if initialization!='scratch':
            report['balance_imitation']=fit_teacher(model,dict(position_gain=1.,velocity_gain=2.,theta_gain=40.,omega_gain=2.),env.action_scale)
        if data is not None:report['demonstration_imitation']=fit_demonstrations(model,data,env.action_scale,updates=imitation_updates,balance_anchor_weight=balance_anchor_weight)
        model.project_equilibrium();(output/'checkpoints').mkdir()
        for checkpoint in range(0,steps+1,every):
            if checkpoint:
                model.learn(total_timesteps=every,reset_num_timesteps=checkpoint==every,callback=Episodes(),log_interval=1)
                model.logger.record('time/total_timesteps',int(model.num_timesteps));model.logger.dump(step=int(model.num_timesteps))
            if source_identity()!=pins:raise ValueError('source/dependency changed during trial')
            cp=output/'checkpoints'/f'step-{checkpoint:09d}.zip';model.save(cp)
            result=evaluate_balance(model,experiment,seeds=dev,seconds=evaluation_seconds,starts=('hanging','upright'))
            result.update(steps=int(model.num_timesteps),checkpoint=str(cp.relative_to(output)),checkpoint_sha256=sha256(cp.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-began)
            report['evaluations'].append(result);save()
            print(json.dumps(dict(initialization=initialization,seed=seed,steps=checkpoint,by_start=result['by_start'])),flush=True)
        report.update(completed=True,wall_seconds=time.monotonic()-began);save();return report
    except BaseException as error:report['error']=str(error);save();raise
    finally:env.close()


def load_trial(run, checkpoint=None):
    run=Path(run);report=json.loads((run/'report.json').read_text());raw=(run/'recipe.json').read_bytes();recipe=json.loads(raw)
    if sha256(raw).hexdigest()!=report['recipe_sha256'] or any(report.get(k)!=v for k,v in recipe.items()):raise ValueError('recipe hash or metadata mismatch')
    if sha256((run/'source.zip').read_bytes()).hexdigest()!=report['source_zip_sha256']:raise ValueError('source archive hash mismatch')
    if recipe['provenance']!=source_identity():raise ValueError('source/dependency changed since training')
    if recipe['demonstrations'] is not None and sha256((run/'demonstrations.npz').read_bytes()).hexdigest()!=recipe['demonstrations']['sha256']:raise ValueError('demonstration hash mismatch')
    selected=report['evaluations'][-1] if checkpoint is None else next((e for e in report['evaluations'] if e['steps']==checkpoint),None)
    if selected is None:raise ValueError('checkpoint not found')
    path=run/selected['checkpoint']
    if path.parent.resolve()!=(run/'checkpoints').resolve() or sha256(path.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:raise ValueError('checkpoint path or hash mismatch')
    with preserved_rng():model=PPO.load(path,device='cpu')
    return model,recipe,selected


def evaluate_saved(run,output,*,seeds,seconds=60.,checkpoint=None,starts=('hanging','upright')):
    model,recipe,selected=load_trial(run,checkpoint)
    used=set(recipe['training_seeds']+recipe['evaluation_seeds'])
    if recipe['demonstrations'] is not None:used.update(recipe['demonstrations']['train_episodes']+recipe['demonstrations']['validation_episodes'])
    if set(seeds)&used:raise ValueError('evaluation seed overlap')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    result=evaluate_balance(model,recipe['experiment'],seeds=seeds,seconds=seconds,starts=starts)
    result.update(checkpoint_sha256=selected['checkpoint_sha256'],provenance=source_identity(),hardware_accepted=False,deployable=False)
    output.mkdir(parents=True,exist_ok=False);_write_new(output/'evaluation.json',result);return result
