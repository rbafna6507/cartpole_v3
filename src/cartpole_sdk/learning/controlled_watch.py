"""Bounded HTTP-only checkpoint backup. No kernel execution or VM lifecycle APIs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import posixpath
import tempfile
import time
import zipfile
from .colab_bundle import verify_bundle
from .model_experiment import digest


def filehash(path):return sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def poll(download,output,*,recipe_bytes,source_hashes):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    recipe=json.loads(recipe_bytes);run=recipe['run']
    if PurePosixPath(run).is_absolute() or '..' in PurePosixPath(run).parts:raise ValueError('invalid remote run')
    with tempfile.TemporaryDirectory(prefix='.staging-',dir=output) as temporary:
        staging=Path(temporary)
        def fetch(name,local):download(name,local);return local
        received=fetch('inputs/recipe.json',staging/'recipe.json')
        if received.read_bytes()!=recipe_bytes:raise ValueError('remote recipe differs from frozen bundle')
        state=json.loads(fetch('job-status.json',staging/'status.json').read_text())
        if state.get('run')!=run or state.get('status') not in ('training','completed','failed'):raise ValueError('unexpected job status/run')
        try:
            progress=json.loads(fetch(run+'/progress.json',staging/'progress.json').read_text())
        except (OSError,subprocess.SubprocessError,RuntimeError,TimeoutError):
            if state['status']!='failed':raise
            progress=dict(checkpoints=[],unavailable_after_failure=True)
            (staging/'progress.json').write_text(json.dumps(progress))
        count=0
        for relative in progress['checkpoints']:
            path=PurePosixPath(relative)
            if len(path.parts)!=2 or path.parts[0]!='checkpoints' or not path.parts[1].startswith('step-') or not path.parts[1][5:].isdigit():raise ValueError('invalid checkpoint path')
            dest=output/relative
            manifest=json.loads(fetch(run+'/'+relative+'/manifest.json',staging/'manifest.json').read_text())
            content=dict(manifest);pin=content.pop('content_sha256',None)
            if digest(content)!=pin or manifest['objective']!=recipe['objective']:raise ValueError('checkpoint manifest/objective mismatch')
            expected_binding=recipe['checkpoint_binding']
            variable={'timesteps_completed','checkpoint_sha256','actor_sha256','content_sha256'}
            if set(manifest)!=set(expected_binding)|variable:raise ValueError('checkpoint binding fields mismatch')
            for key,expected in expected_binding.items():
                if manifest[key]!=expected:raise ValueError('checkpoint binding mismatch: '+key)
            if manifest['timesteps_completed']!=int(path.parts[1][5:]):raise ValueError('checkpoint counter mismatch')
            for name,expected in manifest['provenance']['code_sha256'].items():
                source=posixpath.normpath('src/cartpole_sdk/'+name)
                if source_hashes.get(source)!=expected:raise ValueError('checkpoint runtime source mismatch')
            for name,expected in manifest['experiment_sources_sha256'].items():
                if source_hashes.get('src/cartpole_sdk/learning/'+name)!=expected:raise ValueError('checkpoint source mismatch')
            if dest.exists():
                if json.loads((dest/'manifest.json').read_text())!=manifest:raise ValueError('immutable checkpoint manifest conflict')
                if filehash(dest/'checkpoint.zip')!=manifest['checkpoint_sha256'] or filehash(dest/'actor.json')!=manifest['actor_sha256']:raise ValueError('retained checkpoint corruption')
            else:
                cp=staging/path.parts[1];cp.mkdir()
                (cp/'manifest.json').write_bytes((staging/'manifest.json').read_bytes())
                for name,key in (('checkpoint.zip','checkpoint_sha256'),('actor.json','actor_sha256')):
                    fetch(run+'/'+relative+'/'+name,cp/name)
                    if filehash(cp/name)!=manifest[key]:raise ValueError('download SHA mismatch')
                with zipfile.ZipFile(cp/'checkpoint.zip') as archive:
                    if archive.testzip() is not None:raise ValueError('checkpoint ZIP CRC mismatch')
                dest.parent.mkdir(parents=True,exist_ok=True);cp.rename(dest)
            count+=1
        metadata=output/'metadata';metadata.mkdir(exist_ok=True)
        for name in ('recipe.json','status.json','progress.json'):
            value=(staging/name).read_bytes();target=metadata/(sha256(value).hexdigest()+'.json')
            if not target.exists():target.write_bytes(value)
        if state['status']=='completed':
            fetch(run+'/report.json',staging/'report.json')
            if filehash(staging/'report.json')!=state['report_sha256']:raise ValueError('final report SHA mismatch')
            if not (output/'report.json').exists():(output/'report.json').write_bytes((staging/'report.json').read_bytes())
        result=dict(status=state['status'],error=state.get('error'),verified_checkpoints=count,checked_unix=time.time(),remote_run=run)
        atomic(output/'last-complete.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('--session',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cli',type=Path,required=True)
    parser.add_argument('--cli-pins',type=Path,required=True)
    parser.add_argument('--state-file',type=Path,default=Path.home()/'.config/colab-cli/sessions.json')
    parser.add_argument('--max-seconds',type=float,default=28800);parser.add_argument('--interval',type=float,default=30.)
    args=parser.parse_args()
    if not 0<args.max_seconds<=86400 or not 1<=args.interval<=60:raise ValueError('invalid watch timing')
    verify_bundle(args.bundle)
    pins=json.loads(args.cli_pins.read_text());root=Path(__file__).resolve().parents[3]
    if args.cli.resolve()!=(root/pins['executable']).resolve():raise ValueError('CLI executable differs from pins')
    for name,expected in pins['files'].items():
        if filehash(root/name)!=expected:raise ValueError('CLI source differs from inspected version')
    args.output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(args.bundle/'source.zip') as archive:source_hashes={n:sha256(archive.read(n)).hexdigest() for n in archive.namelist()}
    recipe_bytes=(args.bundle/'recipe.json').read_bytes()
    started=time.monotonic();deadline=started+args.max_seconds;failures=0;first_success=False
    def download(name,path):
        timeout=min(45,deadline-time.monotonic())
        if timeout<=0:raise TimeoutError('watch deadline reached')
        command=[str(args.cli),'--auth','oauth2','--config',str(args.state_file),'download','-s',args.session,'/content/cartpole-training/'+name,str(path)]
        subprocess.run(command,check=True,timeout=timeout,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if not path.is_file():raise RuntimeError('HTTP download did not produce a file')
    while time.monotonic()<deadline:
        try:
            result=poll(download,args.output,recipe_bytes=recipe_bytes,source_hashes=source_hashes)
            failures=0;first_success=True;atomic(args.output/'watch-status.json',result)
            if result['status'] in ('completed','failed'):return
        except ValueError as error:
            atomic(args.output/'watch-status.json',dict(status='integrity_failed',error=str(error)))
            raise
        except (subprocess.SubprocessError,OSError,RuntimeError,TimeoutError) as error:
            failures+=1
            atomic(args.output/'watch-status.json',dict(status='transport_error',consecutive_failures=failures,error=type(error).__name__))
            if failures>=3 and (first_success or time.monotonic()-started>=900):raise RuntimeError('three consecutive HTTP backup failures') from None
        time.sleep(min(args.interval,max(0,deadline-time.monotonic())))
    atomic(args.output/'watch-status.json',dict(status='deadline_reached'))

if __name__=='__main__':main()
