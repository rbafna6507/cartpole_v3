"""Trusted bounded experiment recorder and immutable owner-authorized bundles.

The policy never sees these file handles. Disk writes happen on a separate thread;
queue overflow and byte limits are reported rather than blocking control.
"""
from dataclasses import asdict
import hashlib
import hmac
import io
import multiprocessing
import importlib.metadata
import sys
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import zipfile
from .observation_payload import observation_payload
from cartpole_sdk.recording import canonical_json, strict_json_loads

MAX_BUNDLE_BYTES=40*1024*1024
MAX_TELEMETRY_BYTES=32*1024*1024


MAX_SENSING_BYTES=3*1024*1024
MAX_DETAILED_SENSING_BYTES=8*1024*1024

def _capture_sensing(url,duration,directory,stop,detailed=False):
    """Spawned child: all camera network parsing/JSON work stays off control GIL."""
    from cartpole_sdk.sensing.collect import collect
    # A worker may itself use realtime scheduling; its recorder must not inherit it.
    try:
        if hasattr(os,'sched_setscheduler'):os.sched_setscheduler(0,os.SCHED_OTHER,os.sched_param(0))
        os.nice(10)
    except OSError:pass
    max_bytes=MAX_DETAILED_SENSING_BYTES if detailed else MAX_SENSING_BYTES
    sample_limit=128 if detailed else 1
    status={'max_bytes':max_bytes,'max_hz':30,'sample_limit':sample_limit,'status':'connecting','envelopes':0,'dropped_records':0,'bytes':0,'collector_pid':os.getpid(),
            'source':'developer_streams_v1','sampling':'30 Hz bounded history; original cursors and source gap metadata retained; not full-rate capture'}
    path=Path(directory)/'sensing.jsonl'
    class CaptureEnd(Exception):pass
    try:
        with path.open('xb',buffering=0) as stream:
            os.chmod(path,0o600)
            class Output:
                def write(self,line):
                    if stop.is_set():status['status']='stopped';raise CaptureEnd()
                    raw=(canonical_json({'type':'sensing_envelope','recorded_monotonic_ns':time.monotonic_ns(),
                                         'envelope':strict_json_loads(line)})+'\n').encode()
                    if len(raw)>1024*1024:status['status']='record_limit';status['dropped_records']+=1;raise CaptureEnd()
                    if status['bytes']+len(raw)>max_bytes:status['status']='byte_limit';status['dropped_records']+=1;raise CaptureEnd()
                    stream.write(raw);status['bytes']+=len(raw);status['envelopes']+=1
                def flush(self):pass
            collect(url,Output(),duration=min(3600,duration),max_hz=30,limit=sample_limit,max_bytes=max_bytes+1024*1024)
            status['status']='ended'
    except CaptureEnd:pass
    except Exception:status['status']='unavailable'
    finally:
        try:
            fd=os.open(Path(directory)/'sensing-status.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'w') as stream:stream.write(canonical_json(status))
        except OSError:pass

class RunRecorder:
    def __init__(self,root,job,*,max_bytes=MAX_TELEMETRY_BYTES):
        self.job={key:job[key] for key in ('id','artifact','filename','source','duration','rig','approved_by','approved_at','recording') if key in job}
        root=Path(root);root.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.directory=root/(job['id']+'-'+uuid.uuid4().hex)
        self.directory.mkdir(mode=0o700)
        self.path=self.directory/'telemetry.jsonl'
        self.stream=self.path.open('xb');os.chmod(self.path,0o600)
        self.max_bytes=max_bytes;self.bytes=0;self.count=0;self.dropped=0;self.error=None
        self.drop_lock=threading.Lock()
        self.emit_lock=threading.Lock();self.closed=False;self.pending_bytes=0;self.sensing_bytes=0
        self.sensing_process=None;self.sensing_stop=None;self.sensing_status=None
        self.max_sensing_bytes=MAX_SENSING_BYTES
        self.pending=queue.Queue(maxsize=2048)
        self.started_ns=time.monotonic_ns();self.started_unix_ns=time.time_ns()
        self.thread=threading.Thread(target=self._write,daemon=True);self.thread.start()
    def _drop(self):
        with self.drop_lock:self.dropped+=1
    def _write(self):
        while True:
            item=self.pending.get()
            if item is None:break
            with self.emit_lock:self.pending_bytes-=len(item)
            try:
                if self.error is not None or self.bytes+len(item)>self.max_bytes:self._drop();continue
                self.stream.write(item);self.bytes+=len(item);self.count+=1
            except OSError:self.error='telemetry_write_failed';self._drop()
        try:self.stream.flush();os.fsync(self.stream.fileno())
        except OSError:self.error='telemetry_flush_failed'
        finally:self.stream.close()
    def emit(self,kind,**fields):
        # Only small control records enter this thread queue. All camera JSON
        # parsing/serialization belongs to the spawned sensing process.
        if kind=='sensing_envelope':raise ValueError('sensing records require the isolated collector')
        try:raw=(canonical_json({'type':kind,'recorded_monotonic_ns':time.monotonic_ns(),**fields})+'\n').encode()
        except (ValueError,TypeError):self._drop();return
        with self.emit_lock:
            if self.closed:return
            if len(raw)>65536 or self.pending_bytes+len(raw)>4*1024*1024:
                self._drop();return
            try:self.pending.put_nowait(raw)
            except queue.Full:self._drop();return
            self.pending_bytes+=len(raw)
    def start_sensing(self,url,duration,*,detailed=False):
        """Spawn independent camera collection before backend setup/arming."""
        self.max_sensing_bytes=MAX_DETAILED_SENSING_BYTES if detailed else MAX_SENSING_BYTES
        context=multiprocessing.get_context('spawn')
        self.sensing_stop=context.Event()
        self.sensing_process=context.Process(target=_capture_sensing,args=(url,duration,str(self.directory),self.sensing_stop,detailed),daemon=True)
        self.sensing_process.start()
        self.sensing_status={'status':'collecting','collector_pid':self.sensing_process.pid}
    def _stop_sensing(self):
        if self.sensing_process is None:return
        if self.sensing_process.pid is None:
            self.sensing_status={'status':'collector failed to start'};return
        self.sensing_stop.set();self.sensing_process.join(timeout=.25)
        if self.sensing_process.is_alive():
            self.sensing_process.terminate();self.sensing_process.join(timeout=1)
        if self.sensing_process.is_alive():
            self.sensing_process.kill();self.sensing_process.join(timeout=1)
        if self.sensing_process.is_alive():raise TimeoutError('sensing process did not stop')
        status_path=self.directory/'sensing-status.json'
        if status_path.exists():
            with status_path.open('rb') as stream:self.sensing_status=strict_json_loads(stream.read(4096).decode())
        else:self.sensing_status['status']='interrupted; final source status unavailable'
        self.dropped+=self.sensing_status.get('dropped_records',0)
        self.sensing_status['merged_envelopes']=0
        self.sensing_status['partial_or_rejected_records']=0
    def _finish_sensing(self):
        if self.sensing_process is None:return
        path=self.directory/'sensing.jsonl'
        if not path.exists():return
        # Called only after the backend is stopped and scalar writer closed.
        # Merge complete lines without reparsing/serializing camera envelopes.
        with path.open('rb') as source,self.path.open('ab') as output:
            while True:
                raw=source.readline(1024*1024+1)
                if not raw:break
                if (len(raw)>1024*1024 or not raw.endswith(b'\n')
                        or self.sensing_bytes+len(raw)>self.max_sensing_bytes
                        or self.bytes+len(raw)>self.max_bytes):
                    self._drop();self.sensing_status['partial_or_rejected_records']+=1;break
                output.write(raw);self.bytes+=len(raw);self.sensing_bytes+=len(raw);self.count+=1
                self.sensing_status['merged_envelopes']+=1
            output.flush();os.fsync(output.fileno())
    def observation(self,observation):self.emit('observation',observation=observation_payload(observation))
    def finalize(self,status,outcome):
        self._stop_sensing()
        with self.emit_lock:self.closed=True
        self.pending.put(None,timeout=5);self.thread.join(timeout=5)
        if self.thread.is_alive():raise TimeoutError('recording writer did not finish')
        self._finish_sensing()
        manifest={'schema':'cartpole-run-bundle-v1','job':self.job,'status':status,'outcome':outcome,
            'started_unix_ns':self.started_unix_ns,'started_monotonic_ns':self.started_ns,
            'recording':{'records':self.count,'telemetry_bytes':self.bytes,'dropped_records':self.dropped,'error':self.error,
                         'max_telemetry_bytes':self.max_bytes,'max_sensing_bytes':self.max_sensing_bytes,'sensing_enqueued_bytes':self.sensing_bytes,'telemetry_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()},
            'coverage':{'observations':'all channels returned at control cadence, bounded by recorder limits',
                'actions':'requested acceleration; not proof of physical application',
                'applied_motion':'available observation feedback and backend diagnostics only',
                'decoded_board_state':'cached state sampled per observation in detailed hardware mode; no extra serial queries or guarantee of every board sample',
                'sensing_streams':'all streams returned by configured source; history/cursors/gaps retained within bounds; inspect vision status',
                'video':'not_recorded','vision':dict(self.sensing_status) if self.sensing_status else ('requested but no sensing source configured' if self.job.get('recording')=='detailed' else 'no sensing source configured'),
                'native_encoder_counts':'cached INFO encoder_raw when supplied, possibly stale; not continuously available',
                'firmware_internal_samples':'not_recorded','serial_bytes':'not_recorded',
                'user_stdout_stderr':'not_recorded; sandbox protocol errors are represented by outcome',
                'calibration_and_gains':'only metadata supplied by session/backend; no separate full snapshot'},
            'software':{'sdk_version':importlib.metadata.version('cartpole-sdk'),'python_version':sys.version.split()[0],
                'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('worker.py','run_recording.py','sandbox.py','harness.py','artifacts.py','../policies/numeric_backend.py','../sim/portal_bridge.py','../sim/profiles.py','../sim/rig_gym_env.py','../sim/executor.py')}},
            'clock_notes':'recorded_monotonic_ns is worker-host time, including the separate sensing process on the same host. Sensing records are appended after scalar records; sort by timestamp if needed. Observation clocks and sample times retain original semantics'}
        manifest_bytes=canonical_json(manifest).encode()
        if len(manifest_bytes)>1024*1024:raise ValueError('manifest too large')
        destination=self.directory/'run.zip'
        with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('manifest.json',manifest_bytes)
            archive.write(self.path,'telemetry.jsonl')
        os.chmod(destination,0o600)
        return destination

class RecordingStoreMixin:
    def put_recording(self,identifier,worker,token,bundle):
        from .queue import QueueError,name
        name(identifier);name(worker)
        if type(bundle) is not bytes or not 1<=len(bundle)<=MAX_BUNDLE_BYTES:raise QueueError('invalid recording size')
        try:
            with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                entries=archive.infolist()
                if len(entries)!=2 or {e.filename for e in entries}!={'manifest.json','telemetry.jsonl'}:raise ValueError()
                if any(e.compress_type!=zipfile.ZIP_STORED or e.flag_bits&1 for e in entries):raise ValueError()
                if archive.getinfo('manifest.json').file_size>1024*1024 or archive.getinfo('telemetry.jsonl').file_size>MAX_TELEMETRY_BYTES:raise ValueError()
                manifest=strict_json_loads(archive.read('manifest.json').decode())
                if manifest['schema']!='cartpole-run-bundle-v1' or manifest['job']['id']!=identifier:raise ValueError()
                recording=manifest['recording']
                if type(recording['dropped_records']) is not int or recording['dropped_records']<0:raise ValueError()
                if recording['error'] is not None and (not isinstance(recording['error'],str) or len(recording['error'])>128):raise ValueError()
                telemetry=archive.read('telemetry.jsonl')
                if hashlib.sha256(telemetry).hexdigest()!=manifest['recording']['telemetry_sha256']:raise ValueError()
        except (ValueError,KeyError,TypeError,UnicodeError,zipfile.BadZipFile,RuntimeError) as exc:raise QueueError('invalid recording bundle') from exc
        sha=hashlib.sha256(bundle).hexdigest()
        with self._tx() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None or row['status'] not in ('succeeded','failed','cancelled') or row['worker']!=worker or not isinstance(token,str) or not isinstance(row['lease'],str) or not hmac.compare_digest(row['lease'],token):raise QueueError('recording worker identity mismatch')
            if manifest['job']['artifact']!=row['artifact']:raise QueueError('recording artifact mismatch')
            directory=self.root/'recordings'
            if directory.is_symlink():raise QueueError('unsafe recording directory')
            directory.mkdir(mode=0o700,exist_ok=True)
            directory_fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                try:
                    fd=os.open(identifier+'.zip',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400,dir_fd=directory_fd)
                    try:
                        with os.fdopen(fd,'wb') as stream:stream.write(bundle);stream.flush();os.fsync(stream.fileno())
                    except BaseException:
                        os.unlink(identifier+'.zip',dir_fd=directory_fd);raise
                except FileExistsError:
                    fd=os.open(identifier+'.zip',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory_fd)
                    with os.fdopen(fd,'rb') as stream:
                        if stream.read(MAX_BUNDLE_BYTES+1)!=bundle:raise QueueError('recording already exists')
                os.fsync(directory_fd)
            except OSError as exc:raise QueueError('recording storage failed') from exc
            finally:os.close(directory_fd)
            result={} if row['result'] is None else strict_json_loads(row['result'])
            result['recording']={'status':'available','sha256':sha,'bytes':len(bundle),'schema':'cartpole-run-bundle-v1',
                                 'dropped_records':manifest['recording']['dropped_records'],'error':manifest['recording']['error']}
            db.execute('UPDATE jobs SET result=? WHERE id=?',(canonical_json(result),identifier))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
    def recording(self,identifier,owner=None):
        from .queue import QueueError
        job=self.get(identifier,owner)
        try:
            directory=os.open(self.root/'recordings',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                fd=os.open(identifier+'.zip',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                with os.fdopen(fd,'rb') as stream:bundle=stream.read(MAX_BUNDLE_BYTES+1)
            finally:os.close(directory)
        except OSError as exc:raise QueueError('recording unavailable') from exc
        metadata=(job.get('result') or {}).get('recording') or {}
        if len(bundle)>MAX_BUNDLE_BYTES or metadata.get('sha256')!=hashlib.sha256(bundle).hexdigest():raise QueueError('recording integrity failure')
        return bundle
