"""Generated bundle's remote helper. Only stdlib; runs on an explicitly allocated VM."""
import hashlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

PROJECT = Path('/content/cartpole-training')


def write_state(value):
    value['time'] = time.time()
    temporary = PROJECT/'job-status.tmp'
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(PROJECT/'job-status.json')


def safe_extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = Path(item.filename)
            if path.is_absolute() or '..' in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('unsafe archive member')
        z.extractall(destination)


def setup():
    for name, expected in EXPECTED_ARCHIVES.items():
        if hashlib.sha256((Path('/content')/name).read_bytes()).hexdigest() != expected:
            raise ValueError('uploaded archive hash mismatch: '+name)
    PROJECT.mkdir(exist_ok=False)
    safe_extract('/content/source.zip', PROJECT)
    safe_extract('/content/inputs.zip', PROJECT/'inputs')
    recipe = json.loads((PROJECT/'inputs/recipe.json').read_text())
    subprocess.run([sys.executable,'-m','pip','install','uv=='+recipe['uv']],check=True)
    subprocess.run([sys.executable,'-m','uv','sync','--locked','--extra','rl','--no-dev',
                    '--python',recipe['python']],cwd=PROJECT,check=True)
    (PROJECT/'training-python.txt').write_text(recipe['python']+'\n')
    (PROJECT/'training-uv.txt').write_text(recipe['uv']+'\n')
    (PROJECT/'colab-archives.json').write_text(json.dumps(EXPECTED_ARCHIVES,sort_keys=True))
    write_state(dict(status='installed', hardware_accepted=False))


def verify_installed_binding():
    if json.loads((PROJECT/'colab-archives.json').read_text()) != EXPECTED_ARCHIVES:
        raise ValueError('selected session belongs to a different training bundle')


def job():
    recipe = json.loads((PROJECT/'inputs/recipe.json').read_text())
    python = str(PROJECT/'.venv/bin/python')
    run = 'runs/'+recipe['run']
    command = [python,'-u','examples/train_ppo.py','train']
    for name in ('model','executor-parameters','observation-config','prior-parameters'):
        command += ['--'+name,'inputs/'+name+'.json']
    command += ['--development-only']
    if 'randomization' in recipe:
        settings = recipe['randomization']
        if (not isinstance(settings, dict) or
                settings.get('schema') != 'cartpole-training-randomization-v1'):
            raise ValueError('invalid frozen randomization recipe')
        frozen = json.loads((PROJECT/'inputs/randomization.json').read_text())
        if frozen != settings:
            raise ValueError('randomization input differs from frozen recipe')
        command += ['--randomization', 'inputs/randomization.json']
    else:
        # Immutable older bundles declare the original engineering preset.
        command += ['--randomize']
    command += ['--training-seeds', *map(str, recipe['training_seeds'])]
    for key in ('seed','steps','seconds','n_steps','batch_size','epochs','gamma','gae_lambda','learning_rate','reward_scale','checkpoint_every'):
        command += ['--'+key.replace('_','-'),str(recipe[key])]
    command += ['--output',run]
    write_state(dict(status='training',pid=os.getpid(),run=run,hardware_accepted=False))
    try:
        subprocess.run(command,cwd=PROJECT,check=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        write_state(dict(status='evaluating',pid=os.getpid(),run=run,hardware_accepted=False))
        subprocess.run([python,'-u','examples/train_ppo.py','evaluate',run,
            '--seeds',*map(str,recipe['evaluation_seeds']), '--hold-seconds',str(recipe['hold_seconds']),
            '--output','runs/ppo-001-heldout.json'],cwd=PROJECT,check=True)
        write_state(dict(status='completed',run=run,hardware_accepted=False))
    except BaseException as error:
        write_state(dict(status='failed',error=str(error),run=run,hardware_accepted=False))
        raise


def start():
    # An ambiguous client timeout must not launch another trainer on retry.
    (PROJECT/'job-launch.lock').mkdir(exist_ok=False)
    with (PROJECT/'training.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable,'-u','/content/cartpole-launch.py','job'],
            cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(started_pid=process.pid, status='launched',
        note='Use status; VM lifetime is controlled by Colab. Download checkpoints regularly.')))


def status():
    path = PROJECT/'job-status.json'
    state = json.loads(path.read_text()) if path.exists() else {'status':'installed'}
    recorded = state.get('status')
    state['recorded_status'] = recorded
    state['checked_at'] = time.time()
    if recorded in {'training','evaluating'}:
        try:
            command = Path('/proc', str(int(state['pid'])), 'cmdline').read_bytes()
            alive = b'cartpole-launch.py' in command and b'job' in command.split(b'\0')
        except (OSError, KeyError, ValueError):
            alive = False
        state['supervisor_alive'] = alive
        if not alive:
            state['status'] = 'interrupted'
    run = PROJECT/'runs/ppo-001'
    progress = run/'logs/progress.csv'
    state['latest_progress'] = None
    if progress.exists():
        # Ignore an incomplete row while the trainer appends optimizer output.
        with progress.open() as stream:
            for row in csv.DictReader(stream):
                if None not in row and all(value is not None for value in row.values()) and row.get('time/total_timesteps'):
                    state['latest_progress'] = row
        state['progress_age_s'] = max(0.,time.time()-progress.stat().st_mtime)
    state['checkpoints'] = []
    checkpoints = run/'checkpoints'
    if checkpoints.exists():
        for folder in sorted(checkpoints.iterdir()):
            try:
                manifest = json.loads((folder/'manifest.json').read_text())
                if all((folder/name).is_file() for name in ('checkpoint.zip','actor.json')) and manifest.get('timesteps_completed'):
                    state['checkpoints'].append(folder.name)
            except (OSError, ValueError):
                pass
    print('CARTPOLE_STATUS '+json.dumps(state,allow_nan=False),flush=True)
    log = PROJECT/'training.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-5000))
            print(stream.read().decode(errors='replace'))


def package():
    destination = Path('/content/cartpole-results.zip')
    temporary = destination.with_suffix('.tmp')
    selected = ['pyproject.toml','README.md','uv.lock','training-python.txt','training-uv.txt',
                'src','examples','scripts','inputs','runs','job-status.json','training.log','colab-archives.json']
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for name in selected:
            root = PROJECT/name
            for item in (sorted(root.rglob('*')) if root.is_dir() else [root]):
                if item.is_file() and not item.is_symlink() and '__pycache__' not in item.parts:
                    z.write(item,item.relative_to(PROJECT))
    temporary.replace(destination)
    print('CARTPOLE_PACKAGE '+json.dumps(dict(path=str(destination),bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        note='A live snapshot may include an incomplete final run; resume only a complete checkpoint.')))


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv)>1 and sys.argv[1]=='job' else os.environ.get('CARTPOLE_ACTION')
    if action not in {'setup','start','status','package','job'}:
        raise ValueError('explicit CARTPOLE_ACTION required')
    if action != 'job' and not os.environ.get('CARTPOLE_NONCE'):
        raise ValueError('explicit per-invocation success nonce required')
    if action != 'setup':
        verify_installed_binding()
    globals()[action]()
    if action != 'job':
        print('CARTPOLE_REMOTE_OK '+json.dumps(dict(action=action,nonce=os.environ['CARTPOLE_NONCE'])),flush=True)
