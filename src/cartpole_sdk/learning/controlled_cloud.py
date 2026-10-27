"""Prepare a controlled-swing Colab bundle offline. Never allocate a runtime."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile
import math
from .colab_bundle import prepare_bundle, verify_bundle
from .controlled_swing import controlled_randomization
from .constrained_swing import OBJECTIVE
from .controlled_training import TRAINING_SEEDS, DEVELOPMENT_SEEDS, CONFIRMATION_SEEDS
from cartpole_sdk.sim.profiles import PROFILE_ID, load_profile, profile_experiment
from .ppo import provenance, experiment_identity


def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

JOB = r'''
def job():
    recipe=json.loads((PROJECT/'inputs/recipe.json').read_text())
    run=recipe['run']
    try:
        write_state(dict(status='training',pid=os.getpid(),run=run,hardware_accepted=False))
        command=[str(PROJECT/'.venv/bin/python'),'-m','cartpole_sdk.learning.controlled_training',
            '--output',run,'--objective',recipe['objective']['id'],'--steps',str(recipe['steps']),'--checkpoint-every',str(recipe['checkpoint_every']),
            '--seconds',str(recipe['seconds']),'--seed',str(recipe['seed']),
            '--init-actor','inputs/initial-actor.json','--evaluation-seeds',*[str(s) for s in recipe['evaluation_seeds']]]
        subprocess.run(command,cwd=PROJECT,check=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        report=PROJECT/run/'report.json'
        if not json.loads(report.read_text()).get('completed'):raise ValueError('training report incomplete')
        write_state(dict(status='completed',pid=os.getpid(),run=run,hardware_accepted=False,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    except BaseException as error:
        write_state(dict(status='failed',pid=os.getpid(),run=run,hardware_accepted=False,error=str(error)))
        raise

def run():
    (PROJECT/'job-launch.lock').mkdir(exist_ok=False)
    with (PROJECT/'training.log').open('xb') as stream:
        process=subprocess.Popen([sys.executable,'-u','/content/cartpole-launch.py','job'],cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT)
        print('CARTPOLE_FOREGROUND_STARTED '+json.dumps(dict(pid=process.pid)),flush=True)
        code=process.wait()
    if code:raise RuntimeError('training supervisor exited '+str(code))
'''


def prepare(repo, output, *, steps=102400, checkpoint_every=25600, seed=11):
    repo,output=Path(repo),Path(output)
    profile=repo/'evidence/policy-recovery-20261021/combined-next/profile'
    manifest=prepare_bundle(repo,profile,output,steps=steps,seed=seed,seconds=25.,checkpoint_every=checkpoint_every,evaluation_seeds=DEVELOPMENT_SEEDS)
    if steps%checkpoint_every:raise ValueError('steps must divide into checkpoint intervals')
    recipe=manifest['recipe']
    recipe.update(objective=OBJECTIVE,randomization=controlled_randomization(),randomize=True,
        training_seeds=list(TRAINING_SEEDS),confirmation_seeds=list(CONFIRMATION_SEEDS),
        learning_rate=3e-6,run='runs/controlled-swing-seed'+str(seed),launch_mode='foreground real training cell',
        profile_id=PROFILE_ID,profile_content_sha256=load_profile()['content_sha256'],
        initial_actor_sha256=filehash(repo/'policies/working.json'),
        algorithm='EquilibriumPPO',initialization='numeric actor only; fresh critic/optimizer/RNG')
    experiment=profile_experiment(seconds=25.)
    actor=json.loads((repo/'policies/working.json').read_text())
    names=('controlled_training.py','controlled_swing.py','energy_swing.py','constrained_swing.py','projected_ppo.py')
    recipe['checkpoint_binding']=dict(schema='cartpole-ppo-checkpoint-v1',algorithm='EquilibriumPPO',status='DEVELOPMENT_ONLY',
        hardware_accepted=False,simulation_only=True,experiment=experiment,experiment_sha256=experiment_identity(experiment),
        provenance=provenance(),experiment_sources_sha256={n:filehash(repo/'src/cartpole_sdk/learning'/n) for n in names},
        profile_id=PROFILE_ID,profile_content_sha256=load_profile()['content_sha256'],training_seeds=list(TRAINING_SEEDS),
        evaluation_seeds=list(DEVELOPMENT_SEEDS),confirmation_seeds=list(CONFIRMATION_SEEDS),seed=seed,
        action_scale_m_s2=25.,objective=OBJECTIVE,training_recipe=dict(randomization=controlled_randomization(),reward_scale=.01,
        transfer='Actor-only; fresh critic, optimizer and RNG. Not continuation of old training.',
        initial_actor_content_sha256=actor['content_sha256'],hyperparameters=dict(n_steps=1024,batch_size=64,n_epochs=10,
        learning_rate=3e-6,gamma=.999,gae_lambda=.98),log_std_init=math.log(.05),network=[21,64,64,1]))
    write(output/'recipe.json',recipe)
    with zipfile.ZipFile(output/'inputs.zip') as archive:entries={n:archive.read(n) for n in archive.namelist()}
    entries['recipe.json']=(output/'recipe.json').read_bytes()
    entries['initial-actor.json']=(repo/'policies/working.json').read_bytes()
    with zipfile.ZipFile(output/'inputs.new.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name,data in entries.items():archive.writestr(name,data)
    (output/'inputs.new.zip').replace(output/'inputs.zip')
    original=(repo/'src/cartpole_sdk/learning/colab_remote.py').read_text()
    header,sep,main=original.partition("if __name__ == '__main__':")
    if not sep:raise ValueError('remote helper entrypoint missing')
    main=main.replace("'setup','start','status','package','job'","'setup','run','status','package','job'")
    remote='EXPECTED_ARCHIVES = '+repr({n:filehash(output/n) for n in ('source.zip','inputs.zip')})+'\n'+header+'\n'+JOB+'\n'+sep+main
    compile(remote,'remote.py','exec');(output/'remote.py').write_text(remote)
    manifest.update(recipe=recipe,adapter='Controlled swing, real foreground training cell; HTTP checkpoint backups required',
        files_sha256={n:filehash(output/n) for n in ('source.zip','inputs.zip','recipe.json','remote.py')})
    write(output/'bundle.json',manifest);verify_bundle(output)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=102400)
    parser.add_argument('--checkpoint-every',type=int,default=25600)
    args=parser.parse_args()
    result=prepare(Path(__file__).resolve().parents[3],args.output,steps=args.steps,checkpoint_every=args.checkpoint_every)
    print(json.dumps(dict(bundle=str(args.output),sha256=filehash(args.output/'bundle.json'),cloud_started=False)))

if __name__=='__main__':main()
