"""Offline full-task actor transfer and matched engineering randomization.

Transfers numeric actor weights only. Critic, optimizer and exploration are fresh;
old-profile performance does not certify the new experiment. No hardware API or
actor export is provided. All physics and reward assumptions remain explicit.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import posixpath
import shutil
import time
import zipfile

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from cartpole_sdk.policies.neural import NeuralPolicy, validate_actor, MAX_BYTES, RUNTIME_FILES
from cartpole_sdk.recording import strict_json_loads
from cartpole_sdk.policies.rig_observation import rig_channels
from . import transfer_checks
from .diagnostics import preserved_rng, _source_snapshot
from .error_training import (bias_observation, load_profile, _bias_degrees, _seconds,
    _file_hash, _weights_hash, source_identity as error_source_identity)
from .model_experiment import validate_experiment
from .model_search import _seeds
from .ppo import _write_new, experiment_identity
from .projected_ppo import EquilibriumPPO
from .swing_experiment import SwingEnv, source_identity as swing_source_identity
from .training_recipe import default_randomization, validate_randomization, SAMPLING_ALGORITHM

# Prior datasets already inspected during the preserved anchored-policy work.
PRIOR_USED_SEEDS = frozenset((*range(20000,20128),*range(21000,21016),53000,53001,
    54000,54001,*range(55000,55008),*range(56000,56008),51100,51101,
    *range(10000,10128),*range(11000,11128),*range(12000,12128),*range(13000,13128),61000,61001,*range(62000,62008),63000,63001,64000,64001,
    65000,65001,66000,66001))


def source_identity():
    root=Path(__file__).resolve().parents[3]
    return dict(swing=swing_source_identity(),angle_error=error_source_identity(),transfer=transfer_checks.source_identity(),
        robust_swing_sha256=_file_hash(__file__),entrypoint_sha256=_file_hash(root/'scripts/robust-swing-learning'))


def validate_task(experiment):
    validate_experiment(experiment)
    observation=experiment['observation_config']
    if observation['angle_source']!='raw' or observation.get('angle_frame_id','encoder-zero')!='encoder-zero':
        raise ValueError('robust swing transfer supports raw encoder-zero actor inputs only')


class RobustSwingEnv(SwingEnv):
    """Original mixed task plus supported per-episode plant and actor-input draws."""
    def __init__(self,experiment,seeds,*,randomized,bias_degrees,randomization_seed,randomization=None):
        validate_task(experiment)
        if type(randomized) is not bool:raise ValueError('randomized must be boolean')
        self.bias_degrees=_bias_degrees(bias_degrees)
        if type(randomization_seed) is not int or randomization_seed<0:raise ValueError('nonnegative randomization seed required')
        settings=default_randomization(angle_source=experiment['observation_config']['angle_source']) if randomized and randomization is None else randomization
        settings=validate_randomization(settings)
        super().__init__(experiment,seeds,reset_mode='mixed')
        # TrainingEnv.reset consumes these validated instance configuration fields.
        # No methods, global RNGs or upstream modules are replaced.
        self.randomization=settings;self.randomization_seed=randomization_seed
        self.actor_angle_bias_rad=0.

    def reset(self,*,seed=None,options=None):
        index=0 if seed is not None else self._episode_index
        rng=np.random.default_rng(np.random.SeedSequence([self.randomization_seed,index,104729]))
        self.actor_angle_bias_rad=math.radians(float(rng.uniform(-self.bias_degrees,self.bias_degrees)))
        observation,info=super().reset(seed=seed,options=options)
        return bias_observation(observation,self.actor_angle_bias_rad),dict(info,actor_angle_bias_rad=self.actor_angle_bias_rad)

    def step(self,action):
        observation,reward,terminated,truncated,info=super().step(action)
        return bias_observation(observation,self.actor_angle_bias_rad),reward,terminated,truncated,dict(info,actor_angle_bias_rad=self.actor_angle_bias_rad)

    def result(self,*,hold_seconds=5.):
        return dict(super().result(hold_seconds=hold_seconds),actor_angle_bias_rad=self.actor_angle_bias_rad)


def _actor_layers(model):
    network=list(model.policy.mlp_extractor.policy_net)+[model.policy.action_net]
    if (len(network)!=5 or not isinstance(network[0],torch.nn.Linear) or not isinstance(network[1],torch.nn.Tanh) or
        not isinstance(network[2],torch.nn.Linear) or not isinstance(network[3],torch.nn.Tanh) or not isinstance(network[4],torch.nn.Linear) or
        [(network[i].in_features,network[i].out_features) for i in (0,2,4)]!=[(21,64),(64,64),(64,1)] or
        list(model.policy.features_extractor.extractors)!=['age_s','state','valid']):
        raise ValueError('warm-start requires identical 21-64-64-1 tanh feature architecture')
    return [network[i] for i in (0,2,4)]


def actor_weights_hash(model):
    value=[]
    for layer in _actor_layers(model):value.extend([layer.weight.detach().cpu().tolist(),layer.bias.detach().cpu().tolist()])
    return sha256(json.dumps(value,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def copy_actor_weights(model,actor,*,action_scale):
    validate_actor(actor)
    if (actor['action_scale_m_s2']!=action_scale or actor['angle_source']!='raw' or
        actor.get('observation_contract',{}).get('angle_frame_id')!='encoder-zero'):
        raise ValueError('warm-start action scale or observation contract mismatch')
    layers=_actor_layers(model)
    with torch.no_grad():
        for target,source in zip(layers,actor['layers']):
            target.weight.copy_(torch.tensor(source['weights'],dtype=target.weight.dtype,device=target.weight.device))
            target.bias.copy_(torch.tensor(source['bias'],dtype=target.bias.dtype,device=target.bias.device))
    return actor_weights_hash(model)


def load_warm_start(run,parity):
    """Verify immutable migration evidence; never resume the old optimizer."""
    run,parity=Path(run),Path(parity)
    recipe=json.loads((run/'recipe.json').read_text());report=json.loads((run/'report.json').read_text())
    if (recipe.get('schema')!='cartpole-swing-learning-v1' or not report.get('completed') or
        _file_hash(run/'recipe.json')!=report['recipe_sha256'] or any(report.get(k)!=v for k,v in recipe.items()) or
        _file_hash(run/'source.zip')!=report['source_zip_sha256'] or experiment_identity(recipe['experiment'])!=recipe['experiment_sha256']):
        raise ValueError('warm-start recipe/source hash mismatch or incomplete run')
    # Offline weight migration verifies the actor against its archived runtime,
    # while the deployment loader continues to require the current runtime.
    with (parity/'actor.json').open('rb') as stream:raw=stream.read(MAX_BYTES+1)
    if len(raw)>MAX_BYTES:raise ValueError('numeric actor exceeds size limit')
    actor=validate_actor(strict_json_loads(raw.decode('utf-8')))
    runtime=actor.get('runtime_sha256')
    if not isinstance(runtime,dict) or set(runtime)!=set(RUNTIME_FILES):
        raise ValueError('numeric actor lacks inference provenance')
    # Preserve channel/observation validation previously supplied by the loader.
    NeuralPolicy(actor,channels=rig_channels(angle_source='raw'))
    selected=next((e for e in report['evaluations'] if e['checkpoint_sha256']==actor['checkpoint_sha256']),None)
    if selected is None:raise ValueError('numeric actor checkpoint is absent from warm-start report')
    checkpoint=run/selected['checkpoint']
    if checkpoint.parent.resolve()!=(run/'checkpoints').resolve() or _file_hash(checkpoint)!=selected['checkpoint_sha256']:
        raise ValueError('warm-start checkpoint path or hash mismatch')
    pr=json.loads((parity/'report.json').read_text());pc=json.loads((parity/'recipe.json').read_text())
    if (not pr.get('completed') or pr.get('emission_parity_pass') is not True or
        _file_hash(parity/'recipe.json')!=pr['recipe_sha256'] or
        any(r.get('actor_file_sha256')!=_file_hash(parity/'actor.json') or r.get('checkpoint_sha256')!=actor['checkpoint_sha256'] for r in (pr,pc)) or
        pc['manifest']['experiment_sha256']!=recipe['experiment_sha256'] or
        experiment_identity(pc['manifest']['experiment'])!=recipe['experiment_sha256']):
        raise ValueError('warm-start numeric parity binding/hash mismatch')
    files={'run/recipe.json':run/'recipe.json','run/report.json':run/'report.json','run/source.zip':run/'source.zip',
        'run/'+selected['checkpoint']:checkpoint,'parity/actor.json':parity/'actor.json',
        'parity/recipe.json':parity/'recipe.json','parity/report.json':parity/'report.json'}
    demonstrations=recipe.get('demonstrations')
    used=set(recipe['training_seeds']+recipe['evaluation_seeds']+pc['seeds'])
    if demonstrations is not None:
        if _file_hash(run/'demonstrations.npz')!=demonstrations['sha256']:raise ValueError('demonstration hash mismatch')
        files['run/demonstrations.npz']=run/'demonstrations.npz'
        used.update(demonstrations['train_episodes']+demonstrations['validation_episodes'])
    with zipfile.ZipFile(run/'source.zip') as archive:
        for name,pin in recipe['provenance']['code_sha256'].items():
            path=posixpath.normpath('src/cartpole_sdk/'+name)
            if sha256(archive.read(path)).hexdigest()!=pin:raise ValueError('archived training source member hash mismatch')
        for key,name in [('diagnostics_sha256','diagnostics.py'),('balance_experiment_sha256','balance_experiment.py'),
                         ('projected_ppo_sha256','projected_ppo.py'),('swing_experiment_sha256','swing_experiment.py')]:
            if sha256(archive.read('src/cartpole_sdk/learning/'+name)).hexdigest()!=recipe['provenance'][key]:
                raise ValueError('archived training helper hash mismatch')
        for name,pin in actor['runtime_sha256'].items():
            if sha256(archive.read('src/cartpole_sdk/'+name)).hexdigest()!=pin:raise ValueError('archived actor runtime hash mismatch')
    # Tensor-only read binds the numeric artifact to actual checkpoint actor
    # weights without instantiating old code or loading optimizer state.
    with zipfile.ZipFile(checkpoint) as archive:
        weights=torch.load(io.BytesIO(archive.read('policy.pth')),map_location='cpu',weights_only=True)
    for source,prefix in zip(actor['layers'],('mlp_extractor.policy_net.0','mlp_extractor.policy_net.2','action_net')):
        for key,target in [('weights','weight'),('bias','bias')]:
            if not np.array_equal(np.asarray(source[key]),weights[prefix+'.'+target].cpu().numpy()):
                raise ValueError('numeric actor differs from actual checkpoint weights')
    root=Path(__file__).resolve().parents[1]
    current_runtime={name:_file_hash(root/name) for name in RUNTIME_FILES}
    binding=dict(run=str(run),parity=str(parity),checkpoint_sha256=selected['checkpoint_sha256'],
        original_actor_runtime_sha256=dict(runtime),migration_runtime_sha256=current_runtime,
        changed_runtime_files=sorted(name for name in runtime if runtime[name]!=current_runtime[name]),
        actor_file_sha256=_file_hash(parity/'actor.json'),actor_content_sha256=actor['content_sha256'],
        original_experiment_sha256=recipe['experiment_sha256'],original_executor=recipe['experiment']['executor_parameters'],
        optimizer_resume=False,transfer='Only actor Linear weights/bias; fresh critic, optimizer and exploration. Profile changes require new evaluation.',
        original_recipe_provenance=recipe['provenance'],files={name:_file_hash(path) for name,path in files.items()})
    return dict(actor=actor,binding=binding,used_seeds=sorted(used),files=files)


def migration_parity(model,actor,experiment):
    policy=NeuralPolicy(actor,channels=rig_channels(angle_source='raw'))
    env=RobustSwingEnv(experiment,(777001,),randomized=False,bias_degrees=0.,randomization_seed=0)
    errors=[];samples=[]
    try:
        with preserved_rng():
            for start in ('hanging','upright'):
                obs,_=env.reset(options=dict(episode_seed=777001,start=start))
                for index in range(min(16,experiment['simulation_config']['max_steps'])):
                    for degrees in (0.,-1.,1.):
                        changed=bias_observation(obs,math.radians(degrees))
                        action,_=model.predict(changed,deterministic=True)
                        raw=float(action[0])*actor['action_scale_m_s2'];numeric=policy.predict_encoded(changed)
                        error=abs(raw-numeric);errors.append(error)
                        samples.append(dict(start=start,index=index,bias_degrees=degrees,encoded={k:v.tolist() for k,v in changed.items()},
                            numeric_raw=numeric,new_sb3_raw=raw,absolute_error=error))
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,terminated,truncated,_=env.step(action)
                    if terminated or truncated:break
    finally:env.close()
    return dict(**{'pass':max(errors)<=1e-5},compared_actions=len(errors),
        max_abs_raw_error_m_s2=max(errors),tolerance_m_s2=1e-5,samples=samples,
        scope='Copied actor on current-profile observations and ±1degree counterfactual inputs; not closed-loop old/new profile equivalence.')


def evaluation_scenarios():
    nominal=dict(name='nominal',expected='task_success',angle_offset_rad=0.,observation_override={},randomization=None)
    rows=[nominal]+[dict(nominal,name=f'bias-{label}{magnitude}',angle_offset_rad=sign*magnitude*math.pi/180)
        for magnitude in (1,2) for label,sign in (('minus',-1),('plus',1))]
    for sign,scale,friction in ((-1,.95,.8),(1,1.05,1.2)):
        setting=default_randomization(angle_source='raw')
        setting['plant']=dict(effective_length_scale=[scale,scale],angular_damping_scale=[friction,friction],coulomb_friction_scale=[friction,friction])
        setting['observation']=dict(angle_noise_std=[.003,.003],telemetry_delay_steps=[5,5],command_delay_steps=[3,3])
        rows.append(dict(nominal,name='combined-minus' if sign<0 else 'combined-plus',angle_offset_rad=sign*math.pi/180,
                         randomization=validate_randomization(setting)))
    return rows


def evaluate_fulltask(model,experiment,output,*,seeds,seconds,checkpoint_sha256):
    seeds=_seeds(seeds,'evaluation seeds');_seconds(seconds);validate_task(experiment)
    exp=deepcopy(experiment);exp['simulation_config']['max_steps']=round(seconds*100)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'traces').mkdir()
    pins=source_identity();before=_weights_hash(model);episodes=[]
    scenarios=evaluation_scenarios()
    _write_new(out/'recipe.json',dict(source=pins,experiment=exp,seeds=list(seeds),seconds=seconds,scenarios=scenarios,
        checkpoint_sha256=checkpoint_sha256,hardware_accepted=False,deployable=False))
    with preserved_rng():
        for scenario in scenarios:
            env=RobustSwingEnv(exp,seeds,randomized=False,bias_degrees=0.,randomization_seed=0,randomization=scenario['randomization'])
            try:
                for seed in seeds:
                    for start in ('hanging','upright'):
                        episode=transfer_checks._episode(model,env,seed=seed,start=start,scenario=scenario,root=out)
                        trace=[json.loads(line) for line in (out/episode['trace_path']).open()]
                        advanced=[r['ground_truth_after'] for r in trace if r['physical_time_advanced']]
                        def rms(rows,key):return float(np.sqrt(np.mean([r[key]**2 for r in rows]))) if rows else None
                        episode.update(proposed_decisions=len(trace),advanced_decisions=len(advanced),
                            true_x_rms_m=rms(advanced,'x'),true_theta_rms_rad=rms(advanced,'theta'),
                            tail_true_x_rms_m=rms(advanced[-500:],'x'),tail_true_theta_rms_rad=rms(advanced[-500:],'theta'))
                        episodes.append(episode)
            finally:env.close()
    if source_identity()!=pins or _weights_hash(model)!=before:raise ValueError('evaluation changed source or weights')
    summary=[]
    for scenario in scenarios:
        for start in ('hanging','upright'):
            rows=[e for e in episodes if e['scenario']==scenario['name'] and e['start']==start]
            def mean(key):return float(np.mean([e[key] for e in rows])) if all(e[key] is not None for e in rows) else None
            summary.append(dict(scenario=scenario['name'],start=start,episodes=len(rows),holds=sum(e['hold_success'] for e in rows),
                faults=sum(e['fault_reason'] is not None for e in rows),worst_max_abs_x_m=max(e['max_abs_x_m'] for e in rows),
                **{key:mean(key) for key in ('true_x_rms_m','true_theta_rms_rad','tail_true_x_rms_m','tail_true_theta_rms_rad')}))
    report=dict(completed=True,hardware_accepted=False,deployable=False,source=pins,checkpoint_sha256=checkpoint_sha256,
        recipe_sha256=_file_hash(out/'recipe.json'),scenario_count=len(scenarios),episode_count=len(episodes),
        episodes=episodes,summary=summary,weights_sha256=before)
    _write_new(out/'report.json',report);return report


def run_trial(experiment,output,*,warm_run,warm_parity,seed,randomized,bias_degrees,steps=20480,every=5120,
              training_seeds=tuple(range(22000,22128)),evaluation_seeds=(68000,),heldout_seeds=(69000,),
              evaluation_seconds=25.,learning_rate=3e-6,policy_std=.05,n_steps=1024,batch_size=64,epochs=10,input_files=None,randomization=None):
    validate_task(experiment);_bias_degrees(bias_degrees);_seconds(evaluation_seconds)
    if type(randomized) is not bool or type(seed) is not int or seed<0:raise ValueError('boolean randomization and nonnegative seed required')
    if randomization is not None and not randomized:raise ValueError('explicit randomization requires randomized=True')
    settings=validate_randomization(default_randomization(angle_source=experiment['observation_config']['angle_source'])
        if randomized and randomization is None else randomization)
    if any(type(v) is not int or v<1 for v in (steps,every,n_steps,batch_size,epochs)) or min(n_steps,batch_size)<2 or n_steps%batch_size or every%n_steps or steps%every:
        raise ValueError('invalid or indivisible budgets')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (learning_rate,policy_std)):raise ValueError('positive finite optimizer settings required')
    warm=load_warm_start(warm_run,warm_parity)
    train,dev,heldout=(_seeds(v,label) for v,label in ((training_seeds,'training seeds'),(evaluation_seeds,'development seeds'),(heldout_seeds,'heldout seeds')))
    prior=set(warm['used_seeds'])|PRIOR_USED_SEEDS|{777001}
    if any(set(a)&set(b) for a,b in ((train,dev),(train,heldout),(dev,heldout))) or prior&set(train+dev+heldout):
        raise ValueError('new/prior pose seed overlap')
    experiment=deepcopy(experiment)
    inputs=dict(input_files or {experiment['admission']['model']['source_path']:experiment['admission']['model']['raw_sha256']})
    def verify_inputs():
        if any(_file_hash(p)!=pin for p,pin in inputs.items()):raise ValueError('profile input hash changed')
        if any(_file_hash(path)!=warm['binding']['files'][name] for name,path in warm['files'].items()):raise ValueError('warm-start source file hash changed')
    verify_inputs();out=Path(output);out.mkdir(parents=True,exist_ok=False);pins=source_identity()
    recipe=dict(schema='cartpole-robust-swing-v1',status='DEVELOPMENT_ONLY',hardware_accepted=False,deployable=False,
        warm_start=warm['binding'],experiment=experiment,experiment_sha256=experiment_identity(experiment),input_files=inputs,source=pins,
        optimizer_seed=seed,steps=steps,every=every,training_seeds=list(train),evaluation_seeds=list(dev),heldout_seeds=list(heldout),
        prior_excluded_seeds=sorted(prior),evaluation_seconds=evaluation_seconds,evaluation_scenarios=evaluation_scenarios(),
        reset_mode='mixed, alternating hanging/upright',randomization=settings,randomization_sampling=SAMPLING_ALGORITHM,
        bias=dict(amplitude_degrees=bias_degrees,sampling='PCG64 SeedSequence([run_seed,episode_index,104729]); uniform constant per episode',
            semantics='Actor sin/cos after quantization only; validity, age, rate, true dynamics and reward unchanged.'),
        hyperparameters=dict(learning_rate=learning_rate,policy_std=policy_std,n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,
            gamma=.999,gae_lambda=.98,device='cpu',threads=1,net_arch=[64,64],reward_scale=.01),
        constraint='Same observed-upright zero-action projection after transfer and each PPO update; no teacher loss or fallback.',
        reward='Unchanged true-state TrainingEnv balance/dwell reward; RMS is evaluation, not a new objective.',
        limitations=['Engineering randomization bounds are assumptions, not fitted uncertainty or validated coverage of current hardware.',
            'Angle bias is an actor-input stress; measured initial angle could also be physical pose. No cause has been identified.',
            f"{steps} decisions with {experiment['simulation_config']['max_steps']/100:g} second episodes is about {steps/experiment['simulation_config']['max_steps']:.2f} full episode equivalents; retention/feasibility, not convergence.",
            f"Actor bw{warm['binding']['original_executor']['bw']:g} transfers to bw{experiment['executor_parameters']['bw']:g}; no old-profile performance equivalence."])
    _write_new(out/'recipe.json',recipe);_source_snapshot(out/'source.zip')
    root=Path(__file__).resolve().parents[3]
    with zipfile.ZipFile(out/'source.zip','a',zipfile.ZIP_DEFLATED) as archive:
        archive.write(root/'scripts/robust-swing-learning','scripts/robust-swing-learning')
        archive.write(root/'scripts/balance-error-learning','scripts/balance-error-learning')
        for i,path in enumerate(inputs):archive.write(path,f'frozen-inputs/{i}-{Path(path).name}')
    for name,path in warm['files'].items():
        target=out/'warm-start'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    report=dict(recipe,completed=False,recipe_sha256=_file_hash(out/'recipe.json'),source_zip_sha256=_file_hash(out/'source.zip'),checkpoints=[])
    def save():
        temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(out/'report.json')
    torch.set_num_threads(1);began=time.monotonic()
    env=RobustSwingEnv(experiment,train,randomized=randomized,bias_degrees=bias_degrees,randomization_seed=seed,randomization=settings)
    class Episodes(BaseCallback):
        def _on_step(self):
            for info in self.locals.get('infos',[]):
                if 'episode_result' in info:
                    with (out/'training-episodes.jsonl').open('a') as stream:stream.write(json.dumps(dict(timesteps=int(self.num_timesteps),**info['episode_result']),allow_nan=False)+'\n')
            return True
    try:
        model=EquilibriumPPO('MultiInputPolicy',env,seed=seed,device='cpu',verbose=0,n_steps=n_steps,batch_size=batch_size,n_epochs=epochs,
            gamma=.999,gae_lambda=.98,learning_rate=learning_rate,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=torch.nn.Tanh,log_std_init=math.log(policy_std)))
        model.set_logger(configure(str(out/'logs'),['csv']))
        report['copied_actor_weights_sha256']=copy_actor_weights(model,warm['actor'],action_scale=env.action_scale)
        report['migration']=migration_parity(model,warm['actor'],experiment)
        if not report['migration']['pass']:raise ValueError('numeric-to-new-policy migration parity failed')
        report['initial_projection_normalized']=model.project_equilibrium();(out/'checkpoints').mkdir()
        for step in range(0,steps+1,every):
            if step:
                model.learn(total_timesteps=every,reset_num_timesteps=step==every,callback=Episodes(),log_interval=1)
                model.logger.record('time/total_timesteps',int(model.num_timesteps));model.logger.dump(step=int(model.num_timesteps))
            verify_inputs()
            if source_identity()!=pins:raise ValueError('training source/dependency changed')
            cp=out/'checkpoints'/f'step-{step:09d}.zip';model.save(cp)
            checkpoint=dict(steps=int(model.num_timesteps),checkpoint=str(cp.relative_to(out)),checkpoint_sha256=_file_hash(cp),actor_weights_sha256=actor_weights_hash(model))
            report['checkpoints'].append(checkpoint);save()
            if step in (0,steps):
                label='initial' if step==0 else 'final'
                evaluation=evaluate_fulltask(model,experiment,out/label,seeds=dev,seconds=evaluation_seconds,checkpoint_sha256=checkpoint['checkpoint_sha256'])
                report[label+'_evaluation']=dict(report=label+'/report.json',report_sha256=_file_hash(out/label/'report.json'),
                    scenario_count=evaluation['scenario_count'],episode_count=evaluation['episode_count'],summary=evaluation['summary']);save()
            print(json.dumps(dict(seed=seed,randomized=randomized,steps=step,event='checkpoint')),flush=True)
        evaluation=evaluate_fulltask(model,experiment,out/'heldout',seeds=heldout,seconds=evaluation_seconds,checkpoint_sha256=report['checkpoints'][-1]['checkpoint_sha256'])
        report['heldout_evaluation']=dict(report='heldout/report.json',report_sha256=_file_hash(out/'heldout/report.json'),
            scenario_count=evaluation['scenario_count'],episode_count=evaluation['episode_count'],summary=evaluation['summary'])
        verify_inputs()
        if source_identity()!=pins:raise ValueError('training source/dependency changed during final evaluation')
        report.update(completed=True,wall_seconds=time.monotonic()-began);save();return report
    except BaseException as exc:report['error']=str(exc);save();raise
    finally:env.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-run',type=Path,required=True);p.add_argument('--warm-parity',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--randomized',action='store_true')
    p.add_argument('--bias-degrees',type=float,default=0.);p.add_argument('--observer-bw',type=int,choices=(6,8,10),help='Override the profile observer bandwidth (Hz).')
    p.add_argument('--steps',type=int,default=20480);p.add_argument('--every',type=int,default=5120)
    p.add_argument('--seconds',type=float,default=25.);p.add_argument('--evaluation-seconds',type=float,default=25.)
    p.add_argument('--training-seeds',type=int,nargs='+',default=list(range(22000,22128)))
    p.add_argument('--evaluation-seeds',type=int,nargs='+',default=[68000]);p.add_argument('--heldout-seeds',type=int,nargs='+',default=[69000])
    p.add_argument('--learning-rate',type=float,default=3e-6);p.add_argument('--policy-std',type=float,default=.05)
    a=p.parse_args(argv)
    try:
        exp,inputs=load_profile(a.profile,observer_bw=a.observer_bw,seconds=a.seconds)
        randomization_path=a.profile/'randomization.json'
        randomization=(validate_randomization(json.loads(randomization_path.read_text()))
            if a.randomized and randomization_path.exists() else None)
        run_trial(exp,a.output,warm_run=a.warm_run,warm_parity=a.warm_parity,seed=a.seed,randomized=a.randomized,bias_degrees=a.bias_degrees,
            steps=a.steps,every=a.every,training_seeds=a.training_seeds,evaluation_seeds=a.evaluation_seeds,heldout_seeds=a.heldout_seeds,
            evaluation_seconds=a.evaluation_seconds,learning_rate=a.learning_rate,policy_std=a.policy_std,input_files=inputs,randomization=randomization)
    except (ValueError,OSError,KeyError) as exc:p.exit(2,f'robust swing learning: {exc}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
