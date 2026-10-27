"""Fail-closed bwrap launcher and independently bounded newline JSON IPC."""
from dataclasses import asdict
import ctypes
import math
import os
from pathlib import Path
import queue
import secrets
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
from .observation_payload import observation_payload
from cartpole_sdk.recording import canonical_json, strict_json_loads

class SandboxError(RuntimeError):pass

def validate_acceleration_ceiling(value):
    if type(value) not in (int,float) or not 0<value<=25 or not math.isfinite(value):
        raise SandboxError('trusted acceleration ceiling must be finite and within (0,25]')
    return float(value)

class PolicyChannel:
    """Trusted process pipe adapter. Untrusted launches must use Sandbox.launch."""
    def __init__(self,process,*,action_timeout=.01,startup_timeout=1.5,stdout_limit=1024*1024,stderr_limit=16384,max_acceleration=.25,wall_seconds=60,preparation_timeout=None):
        self.process=process
        self.action_timeout=action_timeout;self.startup_timeout=startup_timeout
        self.max_acceleration=validate_acceleration_ceiling(max_acceleration)
        self.stdout_limit=stdout_limit;self.stderr_limit=stderr_limit
        self._messages=queue.Queue(maxsize=8);self._error=None;self._error_lock=threading.Lock()
        self._sequence=0;self._reset=False;self._closed=False;self._partial=0
        self.last_action_timing=None
        self._reply_timing={}
        if preparation_timeout is not None and (not isinstance(preparation_timeout,(int,float)) or not math.isfinite(preparation_timeout) or not 0<preparation_timeout<=35):
            raise SandboxError('invalid preparation timeout')
        self._deadline_lock=threading.Lock();self._wall_changed=threading.Event()
        self._separate_preparation=preparation_timeout is not None;self._run_started=False
        self._run_wall_seconds=wall_seconds+min(.5,wall_seconds)
        self._wall_end=time.monotonic()+(preparation_timeout if preparation_timeout is not None else startup_timeout+wall_seconds)
        self._closed_event=threading.Event()
        self._counts={'stdout':0,'stderr':0}
        self._stderr_tail=b''
        os.set_blocking(process.stdin.fileno(),False)
        self._readers=[]
        for name,stream in (('stdout',process.stdout),('stderr',process.stderr)):
            thread=threading.Thread(target=self._drain,args=(name,stream),daemon=True)
            thread.start();self._readers.append(thread)
        self._watcher=threading.Thread(target=self._watch_wall,daemon=True)
        self._watcher.start()
    def _watch_wall(self):
        while not self._closed_event.is_set():
            with self._deadline_lock:
                remaining=self._wall_end-time.monotonic()
                self._wall_changed.clear()
            if remaining<=0:
                self._fail('policy wall deadline exceeded');return
            self._wall_changed.wait(remaining)

    def begin_run(self):
        with self._deadline_lock:
            if not self._reset or self._run_started or self._closed or self._error is not None:
                raise SandboxError('policy is not ready to begin')
            if time.monotonic()>=self._wall_end:raise SandboxError('policy preparation expired')
            self._run_started=True
            if self._separate_preparation:
                self._wall_end=time.monotonic()+self._run_wall_seconds
                self._wall_changed.set()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    @property
    def stderr_tail(self):
        """Last 4KiB child stderr for trusted local debugging; never public results."""
        return self._stderr_tail.decode('utf-8',errors='replace')
    @property
    def buffered_bytes(self):return self._partial+self._messages.qsize()*4096
    def _fail(self,reason):
        with self._error_lock:
            if self._error is None:self._error=SandboxError(reason)
        self._kill()
    def _kill(self):
        if self.process.poll() is None:
            try:os.killpg(self.process.pid,signal.SIGKILL)
            except OSError:pass
            try:self.process.kill()
            except OSError:pass
    def _drain(self,name,stream):
        pending=bytearray()
        try:
            while True:
                chunk=os.read(stream.fileno(),4096)
                if not chunk:break
                self._counts[name]+=len(chunk)
                if self._counts[name]>(self.stdout_limit if name=='stdout' else self.stderr_limit):
                    self._fail('policy output quota exceeded');return
                if name=='stderr':
                    self._stderr_tail=(self._stderr_tail+chunk)[-4096:]
                    continue
                pending.extend(chunk)
                while b'\n' in pending:
                    line,_,rest=pending.partition(b'\n');pending=bytearray(rest)
                    if len(line)>4096:self._fail('oversize policy output');return
                    try:self._messages.put_nowait((bytes(line),time.monotonic_ns()))
                    except queue.Full:self._fail('policy output flood');return
                if len(pending)>4096:self._fail('oversize policy output');return
                self._partial=len(pending)
            if not self._closed:self._fail('isolated policy exited')
        except (OSError,ValueError):
            if not self._closed:self._fail('policy pipe failure')
    def _send(self,value,timeout):
        if self._error is not None:raise self._error
        data=(canonical_json(value)+'\n').encode()
        if len(data)>65536:raise SandboxError('observation exceeds IPC bound')
        deadline=time.monotonic()+timeout
        offset=0
        while offset<len(data):
            remaining=deadline-time.monotonic()
            if remaining<=0:self._fail('policy input timeout');raise self._error
            if not select.select([],[self.process.stdin.fileno()],[],remaining)[1]:continue
            try:offset+=os.write(self.process.stdin.fileno(),data[offset:])
            except BlockingIOError:continue
            except OSError:self._fail('policy input failure');raise self._error
    def _receive(self,timeout):
        deadline=time.monotonic()+timeout
        while True:
            if self._error is not None:raise self._error
            remaining=deadline-time.monotonic()
            if remaining<=0:self._fail('policy response timeout');raise self._error
            try:
                line,line_ns=self._messages.get(timeout=min(remaining,.005))
                self._reply_timing={"stdout_line_ns":line_ns,"reply_dequeued_ns":time.monotonic_ns()}
                break
            except queue.Empty:continue
        try:value=strict_json_loads(line.decode('utf-8'))
        except (ValueError,UnicodeError):self._fail('malformed policy output');raise self._error
        self._reply_timing['reply_decoded_ns']=time.monotonic_ns()
        if not isinstance(value,dict):self._fail('policy output must be an object');raise self._error
        return value
    def reset(self,session):
        if self._reset:raise SandboxError('policy already reset')
        nonce=secrets.token_hex(16)
        began=time.monotonic();self._send({'type':'reset','session':asdict(session),'nonce':nonce},self.startup_timeout)
        reply=self._receive(max(.001,self.startup_timeout-(time.monotonic()-began)))
        if reply!={'type':'reset_ack','nonce':nonce}:self._fail('invalid policy reset acknowledgement');raise self._error
        self._reset=True
    def act(self,observation):
        if not self._reset:raise SandboxError('policy reset acknowledgement required')
        if self._separate_preparation and not self._run_started:raise SandboxError('policy execution not started')
        self._sequence+=1;began=time.monotonic()
        timing={'sequence':self._sequence,'began_ns':time.monotonic_ns(),'stage':'sending'}
        self.last_action_timing=timing
        self._reply_timing={}
        try:
            self._send({'type':'act','sequence':self._sequence,'observation':observation_payload(observation)},self.action_timeout)
            timing['sent_ns']=time.monotonic_ns();timing['stage']='awaiting_reply'
            reply=self._receive(max(.000001,self.action_timeout-(time.monotonic()-began)))
            timing['received_ns']=time.monotonic_ns();timing['stage']='validating'
            # A queued reply can arrive before the deadline while scheduling or JSON
            # decoding resumes after it. Never accept that late action.
            if time.monotonic()-began>self.action_timeout:
                self._fail('policy total action deadline exceeded');raise self._error
            if set(reply)!={'type','sequence','acceleration'} or reply['type']!='action' or type(reply['sequence']) is not int or reply['sequence']!=self._sequence:
                self._fail('invalid policy action envelope');raise self._error
            value=reply['acceleration']
            if value is not None and (type(value) not in (int,float) or abs(value)>self.max_acceleration or not math.isfinite(value)):
                self._fail('policy acceleration exceeds bound or is nonfinite');raise self._error
            timing['stage']='validated'
            return value
        finally:
            timing.update(self._reply_timing)
            timing['finished_ns']=time.monotonic_ns()
    def close(self):
        if self._closed:return
        self._closed=True;self._closed_event.set();self._wall_changed.set();self._kill()
        try:self.process.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            self.process.kill();self.process.wait(timeout=.5)
        for thread in self._readers:thread.join(.2)
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr):stream.close()

class Sandbox:
    """Linux only. Missing namespaces/seccomp/runtime never falls back to host Python."""
    def __init__(self,source,*,sdk_root=None,usr_root='/usr',wall_seconds=60,action_timeout=.01,memory_bytes=512*1024*1024,python_path='/usr/bin/python3',numpy_root=None,max_acceleration=.25,artifact_kind='python',sensor_source='simulation',cpu_affinity=None,realtime=False):
        self.max_acceleration=validate_acceleration_ceiling(max_acceleration)
        self.source=Path(source)
        self.artifact_kind=artifact_kind;self.sensor_source=sensor_source
        if cpu_affinity is not None and (type(cpu_affinity) is not int or not 0<=cpu_affinity<(os.cpu_count() or 1)):
            raise SandboxError('sandbox CPU must be an available nonnegative processor index')
        if type(realtime) is not bool:raise SandboxError('invalid realtime option')
        self.realtime=realtime
        self.cpu_affinity=cpu_affinity
        self.sdk_root=Path(sdk_root) if sdk_root else Path(__file__).resolve().parents[2]
        self.usr_root=Path(usr_root);self.wall_seconds=wall_seconds;self.action_timeout=action_timeout
        self.memory_bytes=memory_bytes;self.python_path=python_path;self.numpy_root=None if numpy_root is None else Path(numpy_root)
    def preflight(self):
        if sys.platform!='linux' or not shutil.which('bwrap'):raise SandboxError('Linux bwrap sandbox unavailable')
        try:ctypes.CDLL('libseccomp.so.2')
        except OSError as exc:raise SandboxError('libseccomp unavailable') from exc
        from .artifacts import MAX_EXPANDED_BYTES, MAX_ENTRIES
        if self.artifact_kind not in ('python','python-zip','actor-json') or self.sensor_source not in ('simulation','magnetic-step'):raise SandboxError('invalid artifact runtime')
        if self.source.is_symlink():raise SandboxError('invalid policy artifact')
        if self.source.is_dir():
            entries=list(self.source.rglob('*'))
            files=[path for path in entries if path.is_file()]
            if (len(entries)>MAX_ENTRIES*16 or len(files)>MAX_ENTRIES or
                    any(path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries) or
                    sum(path.stat().st_size for path in files)>MAX_EXPANDED_BYTES or
                    not (self.source/('actor.json' if self.artifact_kind=='actor-json' else 'policy.py')).is_file()):
                raise SandboxError('invalid prepared policy directory')
        elif not self.source.is_file() or self.source.stat().st_size>65536 or self.artifact_kind!='python':raise SandboxError('invalid policy artifact')
        if not self.usr_root.is_dir() or not (self.sdk_root/'cartpole_sdk'/'portal'/'harness.py').is_file():raise SandboxError('readonly runtime/SDK missing')
        if not self.python_path.startswith('/usr/') or '..' in Path(self.python_path).parts or not (self.usr_root/Path(self.python_path).relative_to('/usr')).is_file():raise SandboxError('runtime Python missing')
        if type(self.wall_seconds) not in (int,float) or not math.isfinite(self.wall_seconds) or not .05<=self.wall_seconds<=120:raise SandboxError('invalid sandbox wall limit')
        if type(self.memory_bytes) is not int or not 64*1024*1024<=self.memory_bytes<=1024*1024*1024:raise SandboxError('invalid sandbox memory bound')
    def launch(self):
        self.preflight()
        if self.realtime:
            try:
                affinity=os.sched_getaffinity(0)
                valid=(self.cpu_affinity is not None and len(affinity)==1 and self.cpu_affinity not in affinity
                       and os.sched_getscheduler(0)==os.SCHED_RR and os.sched_getparam(0).sched_priority==10)
            except (AttributeError,OSError):valid=False
            if not valid:raise SandboxError('realtime sandbox requires RR10 broker on one separate CPU')
        # Ordinary scheduling remains the default; realtime is a trusted opt-in.
        # chrt executes before namespaces and before any submitted code imports.
        scheduler_reset=shutil.which('chrt')
        if scheduler_reset is None:raise SandboxError('sandbox scheduler reset unavailable')
        args=[scheduler_reset,'--rr' if self.realtime else '--other','1' if self.realtime else '0',shutil.which('bwrap'),'--unshare-all','--unshare-user','--unshare-pid','--unshare-net','--new-session','--die-with-parent','--cap-drop','ALL',
              '--ro-bind',str(self.usr_root),'/usr','--symlink','usr/lib','/lib','--symlink','usr/bin','/bin',
              '--proc','/proc','--remount-ro','/proc','--dir','/dev',
              '--ro-bind','/dev/null','/dev/null','--ro-bind','/dev/urandom','/dev/urandom',
              '--dir','/tmp','--dir','/policy',
              '--ro-bind',str(self.source.resolve()),'/policy' if self.source.is_dir() else '/policy/policy.py','--dir','/sdk',
              '--ro-bind',str((self.sdk_root/'cartpole_sdk').resolve()),'/sdk/cartpole_sdk',
              '--clearenv','--setenv','PATH','/usr/bin:/bin','--setenv','OPENBLAS_NUM_THREADS','1',
              '--setenv','OMP_NUM_THREADS','1','--setenv','PYTHONDONTWRITEBYTECODE','1',
              '--setenv','HOME','/tmp','--chdir','/tmp']
        if (self.usr_root/'lib64').exists():args+=['--symlink','usr/lib64','/lib64']
        if self.numpy_root is not None:
            if not (self.numpy_root/'numpy').is_dir() or not (self.numpy_root/'numpy.libs').is_dir():raise SandboxError('explicit readonly numpy runtime missing')
            args+=['--dir','/packages','--ro-bind',str(self.numpy_root/'numpy'),'/packages/numpy','--ro-bind',str(self.numpy_root/'numpy.libs'),'/packages/numpy.libs']
        args +=['--remount-ro','/',self.python_path,'-I','-S','/sdk/cartpole_sdk/portal/harness.py','--wall',str(self.wall_seconds),'--memory',str(self.memory_bytes),'--kind',self.artifact_kind,'--sensor-source',self.sensor_source]
        if self.realtime:args+=['--realtime']
        if self.cpu_affinity is not None:
            affinity_tool=shutil.which('taskset')
            if affinity_tool is None:raise SandboxError('sandbox CPU affinity tool unavailable')
            args=[affinity_tool,'-c',str(self.cpu_affinity)]+args
        if self.realtime:
            limiter=shutil.which('prlimit')
            if limiter is None:raise SandboxError('realtime sandbox resource limiter unavailable')
            # Only the data-only actor loader receives a larger startup allowance.
            # Arbitrary Python/ZIP imports retain the original hard ceiling.
            startup_rttime=1250000 if self.artifact_kind=='actor-json' else 500000
            args=[limiter,f'--rttime={startup_rttime}:{startup_rttime}','--rtprio=1:1','--']+args
        try:process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,start_new_session=True,env={})
        except OSError as exc:raise SandboxError('sandbox launch failed') from exc
        return PolicyChannel(process,action_timeout=self.action_timeout,wall_seconds=self.wall_seconds,max_acceleration=self.max_acceleration,preparation_timeout=35)
