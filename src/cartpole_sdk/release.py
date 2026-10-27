"""Build and install trusted operator releases, without serial or service access.

This is deployment tooling for the repository owner's source, not a sandbox for
public submissions. Only install bundles produced from source you trust.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import uuid


def _name(value):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', value):
        raise ValueError('release name must be 1..80 lowercase letters, digits, hyphens or underscores')
    return value


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def build(repo, name, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    _name(name)
    subprocess.run(['git','diff','--exit-code','HEAD','--'],cwd=repo,check=True,stdout=subprocess.DEVNULL)
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().split('\0'))
    prefixes=('src/','scripts/','policies/','training/','profiles/','docs/','examples/','firmware/')
    files={n for n in tracked if n and (n.startswith(prefixes) or n in ('pyproject.toml','uv.lock','.python-version') or '/' not in n and n.endswith('.md'))}
    protocol='evidence/policy-recovery-20261021/combined-next/protocol.json'
    files.add(protocol)
    files.update(json.loads((repo/protocol).read_text())['bindings'])
    for p in (repo/'training/protocols').glob('*.json'):
        files.update(json.loads(p.read_text()).get('bindings',{}))
    if not files <= tracked:
        raise ValueError('release inputs must all be committed: '+str(sorted(files-tracked)))
    blobs={n:(repo/n).read_bytes() for n in sorted(files)}
    if any((repo/n).is_symlink() for n in blobs):
        raise ValueError('release inputs must be regular files')
    manifest=dict(name=name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),files={n:_sha(b) for n,b in blobs.items()})
    blobs['release-manifest.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream, tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for n,b in blobs.items():
            info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o755 if n.startswith('scripts/') else 0o644
            archive.addfile(info,io.BytesIO(b))
    return dict(archive=str(output),sha256=_sha(output.read_bytes()),**manifest)


def unpack(archive, root):
    root=Path(root).expanduser().resolve();blobs={};modes={};total=0
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or str(p)!=m.name or m.name in blobs:
                raise ValueError('unsafe or duplicate release member: '+m.name)
            total+=m.size
            if total>256*1024*1024 or len(blobs)>=10000:
                raise ValueError('release bundle exceeds 256 MiB/10000 files')
            blobs[m.name]=t.extractfile(m).read();modes[m.name]=m.mode
    manifest=json.loads(blobs['release-manifest.json'])
    name=_name(manifest['name']);expected=manifest['files']
    if set(blobs)-{'release-manifest.json'}!=set(expected):
        raise ValueError('release manifest inventory differs')
    for n,h in expected.items():
        if _sha(blobs[n])!=h:raise ValueError('release hash mismatch: '+n)
    target=root/'releases'/name
    target.mkdir(parents=True,exist_ok=False)
    for n,b in blobs.items():
        p=target/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
        p.chmod(0o755 if modes[n]&0o111 else 0o644)
    return target,manifest


def activate(root, name):
    root=Path(root).expanduser().resolve();target=root/'releases'/_name(name)
    if not (target/'release-ready.json').is_file():
        raise ValueError('release has not passed installation verification')
    active=root/'active'
    if active.exists() and not active.is_symlink():
        raise ValueError('active is an existing directory/file; refusing to replace it')
    temporary=root/('.active-'+uuid.uuid4().hex)
    temporary.symlink_to(Path('releases')/name)
    temporary.replace(active)
    return target


def verify_actors(root):
    from cartpole_sdk.learning.balance_hardware import _load_policy
    for name in ('working', '1m'):
        for start in ('upright', 'hanging'):
            _load_policy(Path(root)/'policies'/(name+'.json'), guard_profile='rig',
                         inference_backend='numpy', start_mode=start)


def install(archive, root, *, uv, python, select=True):
    target,manifest=unpack(archive,root)
    subprocess.run([str(uv),'sync','--locked','--no-dev','--python',str(python)],cwd=target,check=True)
    interpreter=target/'.venv/bin/python'
    check="from cartpole_sdk.release import verify_actors; verify_actors('.'); print('Offline NumPy actor validation passed')"
    subprocess.run([str(interpreter),'-c',check],cwd=target,check=True)
    subprocess.run([str(interpreter),'-m','cartpole_sdk.rig.operator','--dry-run'],cwd=target,check=True)
    for n,h in manifest['files'].items():
        if _sha((target/n).read_bytes())!=h:raise ValueError('installed source changed: '+n)
    (target/'release-ready.json').write_text(json.dumps(dict(git_commit=manifest['git_commit'],offline_verified=True,no_serial_access=True),indent=2)+'\n')
    if select:activate(root,manifest['name'])
    return target


def install_launcher(root, bin_dir):
    root=Path(root).expanduser().resolve();bin_dir=Path(bin_dir).expanduser();bin_dir.mkdir(parents=True,exist_ok=True)
    text='#!/bin/sh\n# cartpole-managed-launcher\nset -eu\ncd '+shlex.quote(str(root/'active'))+'\nexec .venv/bin/cartpole-console "$@"\n'
    for name in ('cartpole','cartpole-console'):
        p=bin_dir/name
        if p.is_symlink() or p.exists() and 'cartpole-managed-launcher' not in p.read_text():
            raise ValueError('refusing to replace unrelated launcher '+str(p))
    for name in ('cartpole','cartpole-console'):
        p=bin_dir/name;p.write_text(text);p.chmod(0o755)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    p=sub.add_parser('build',help='Package committed source and required model/protocol inputs.')
    p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--name',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('install',help='Install a trusted bundle into a new release directory; offline checks only.')
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--root',type=Path,default=Path.home()/'cartpole-v2-live')
    p.add_argument('--uv',type=Path,default=Path.home()/'.local/bin/uv');p.add_argument('--python',default='3.11')
    p.add_argument('--no-activate',action='store_true');p.add_argument('--bin-dir',type=Path,help='Also install stable cartpole/cartpole-console PATH wrappers here.')
    p=sub.add_parser('activate',help='Select an already verified release; does not start a console.')
    p.add_argument('--root',type=Path,default=Path.home()/'cartpole-v2-live');p.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    if args.operation=='build':print(json.dumps(build(args.repo,args.name,args.output),indent=2))
    elif args.operation=='activate':print(activate(args.root,args.name))
    else:
        target=install(args.archive,args.root,uv=args.uv,python=args.python,select=False)
        if args.bin_dir:install_launcher(args.root,args.bin_dir)
        if not args.no_activate:activate(args.root,target.name)
        print(target)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
