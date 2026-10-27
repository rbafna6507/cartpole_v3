"""Bounded data-only inspection and private extraction of immutable submissions.

This module never imports uploaded Python or loads an executable checkpoint.
Numeric actor inference and provenance loading happen only in the sandbox.
"""
import io
from pathlib import Path
import re
import stat
import struct
import zipfile
import zlib
from cartpole_sdk.recording import strict_json_loads

MAX_UPLOAD_BYTES=8*1024*1024
MAX_PYTHON_BYTES=65536
MAX_ACTOR_BYTES=1_000_000
MAX_EXPANDED_BYTES=32*1024*1024
MAX_ENTRIES=256
MAX_ENVELOPE_BYTES=((MAX_UPLOAD_BYTES+2)//3)*4+200_000

class ArtifactError(ValueError):pass

def _text(data):
    try:data.decode('utf-8')
    except UnicodeDecodeError as exc:raise ArtifactError('Python source must be UTF-8') from exc
    if b'\0' in data:raise ArtifactError('Python source cannot contain NUL')

def _zip_directory_bound(data):
    # Bound the central-directory object count before ZipFile allocates entries.
    end=data.rfind(b'PK\x05\x06',max(0,len(data)-65557))
    if end<0 or end+22>len(data):raise ArtifactError('invalid ZIP directory')
    _,disk,start_disk,on_disk,count,size,offset,comment=struct.unpack_from('<4s4H2LH',data,end)
    if disk or start_disk or on_disk!=count or not 1<=count<=MAX_ENTRIES or end+22+comment!=len(data) or offset+size!=end:
        raise ArtifactError('ZIP directory exceeds bounds or uses unsupported multipart/ZIP64 format')
    cursor=offset
    for _ in range(count):
        if cursor+46>end or data[cursor:cursor+4]!=b'PK\x01\x02':raise ArtifactError('invalid ZIP directory')
        names,extra,comments=struct.unpack_from('<3H',data,cursor+28)
        cursor+=46+names+extra+comments
    if cursor!=end:raise ArtifactError('ZIP directory count mismatch')

def _zip_files(data):
    _zip_directory_bound(data)
    result=[];seen=set();total=0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for entry in z.infolist():
                path=entry.filename
                if entry.orig_filename!=path:raise ArtifactError('ZIP NUL path forbidden')
                normalized=path[:-1] if entry.is_dir() else path
                parts=normalized.split('/')
                if (not normalized or len(path)>512 or len(parts)>16 or '\\' in path or
                        any(p in ('','.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',p) for p in parts)):
                    raise ArtifactError('unsafe ZIP path')
                key=normalized.casefold()
                if key in seen:raise ArtifactError('duplicate ZIP path')
                seen.add(key)
                mode=stat.S_IFMT(entry.external_attr>>16)
                if mode not in (0,stat.S_IFDIR if entry.is_dir() else stat.S_IFREG):raise ArtifactError('ZIP links and special files are forbidden')
                if entry.flag_bits&1 or entry.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ArtifactError('encrypted or unsupported ZIP entry')
                total+=entry.file_size
                if total>MAX_EXPANDED_BYTES:raise ArtifactError('ZIP expanded size exceeds 32 MiB')
                if entry.is_dir():continue
                with z.open(entry) as stream:contents=stream.read(entry.file_size+1)
                if len(contents)!=entry.file_size:raise ArtifactError('ZIP size mismatch')
                if path.endswith('.py'):_text(contents)
                result.append((path,contents))
    except (zipfile.BadZipFile,NotImplementedError,RuntimeError,EOFError,OSError,UnicodeError,zlib.error) as exc:
        raise ArtifactError('invalid ZIP payload') from exc
    names={path for path,_ in result}
    if 'policy.py' not in names:raise ArtifactError('ZIP requires root policy.py')
    for path in names:
        parts=path.split('/')
        if any('/'.join(parts[:i]) in names for i in range(1,len(parts))):raise ArtifactError('ZIP file/directory collision')
    return result

def _inspect(data,filename):
    if not isinstance(filename,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,75}\.(py|zip|json)',filename):
        raise ArtifactError('filename must be a plain .py, .zip, or .json name')
    if type(data) is not bytes or not 1<=len(data)<=MAX_UPLOAD_BYTES:raise ArtifactError('artifact must be 1 byte to 8 MiB')
    if filename.endswith('.py'):
        if len(data)>MAX_PYTHON_BYTES:raise ArtifactError('Python source exceeds 64 KiB')
        _text(data)
        files=[('policy.py',data)]
        info={'kind':'python','entrypoint':'policy.py'}
    elif filename.endswith('.zip'):
        files=_zip_files(data)
        info={'kind':'python-zip','entrypoint':'policy.py'}
    else:
        if len(data)>MAX_ACTOR_BYTES:raise ArtifactError('numeric actor exceeds 1000000 bytes')
        try:
            from cartpole_sdk.policies.neural import validate_actor
            actor=validate_actor(strict_json_loads(data.decode('utf-8')))
            if actor['angle_source'] not in ('raw','filtered'):raise ArtifactError('portal numeric actors support raw/filtered encoder sources only')
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:
            raise ArtifactError('invalid or unsupported numeric actor: '+str(exc)[:160]) from exc
        files=[('actor.json',data)]
        info={'kind':'actor-json','entrypoint':'actor.json','schema':actor['schema'],'rate_hz':actor['rate_hz'],
              'angle_source':actor['angle_source'],'action_scale_m_s2':actor['action_scale_m_s2'],
              'content_sha256':actor['content_sha256']}
    info['files']=[{'path':path,'size':len(contents)} for path,contents in files]
    info['expanded_bytes']=sum(len(contents) for _,contents in files)
    return info,files

def inspect_artifact(data,filename):return _inspect(data,filename)[0]

def prepare_artifact(data,filename,directory):
    """Revalidate and materialize into an empty trusted private worker directory."""
    info,files=_inspect(data,filename)
    root=Path(directory)
    if root.is_symlink() or not root.is_dir() or any(root.iterdir()):raise ArtifactError('preparation requires an empty private directory')
    for path,contents in files:
        target=root/path
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(contents)
        target.chmod(0o400)
    # Worker owns directory; bwrap additionally mounts the complete tree read-only.
    return info
