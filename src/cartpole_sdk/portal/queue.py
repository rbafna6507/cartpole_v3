"""Durable owner-bound jobs and one transactional lease per physical rig."""
from contextlib import contextmanager, closing
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sqlite3
import time
import uuid
from cartpole_sdk.recording import canonical_json, strict_json_loads
from .budget import execution_wall_seconds
from .artifacts import inspect_artifact, ArtifactError, MAX_UPLOAD_BYTES
from .reservations import ReservationStoreMixin
from .run_recording import RecordingStoreMixin

class QueueError(ValueError):
    pass

class RigBlocked(QueueError):
    """No claim authority until explicit operator recovery."""

_TERMINAL=('succeeded','failed','cancelled')
MAX_OWNER_ARTIFACT_BYTES=256*1024*1024
MAX_OWNER_JOBS=512
MAX_STORE_ARTIFACT_BYTES=2*1024*1024*1024
MAX_STORE_ARTIFACTS=8192
MIN_FREE_DISK_BYTES=256*1024*1024

def name(value, label='identifier'):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}',value):
        raise QueueError(f'invalid {label}')
    return value

class Store(ReservationStoreMixin, RecordingStoreMixin):
    claim_lease_seconds=2
    def __init__(self, root, *, reservations_enabled=False):
        self.reservations_enabled=bool(reservations_enabled)
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.blobs=self.root/'artifacts'
        if self.blobs.is_symlink():raise QueueError('artifact directory cannot be a symlink')
        self.blobs.mkdir(mode=0o700,exist_ok=True)
        self.database=self.root/'queue.sqlite3'
        if self.database.is_symlink():raise QueueError('database cannot be a symlink')
        with closing(self._connect()) as db:
            db.executescript('''
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, key TEXT NOT NULL,
              artifact TEXT NOT NULL, filename TEXT NOT NULL, source TEXT NOT NULL,
              duration REAL NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
              updated REAL NOT NULL, worker TEXT, rig TEXT, lease TEXT,
              lease_until REAL, deadline REAL, cancel_requested INTEGER NOT NULL DEFAULT 0,
              result TEXT, UNIQUE(owner,key));
            CREATE TABLE IF NOT EXISTS rigs (
              id TEXT PRIMARY KEY, job TEXT, blocked INTEGER NOT NULL DEFAULT 0, reason TEXT);
            ''')
        # Serialize additive migration across concurrently starting workers.
        with self._tx() as db:
            columns={row['name'] for row in db.execute('PRAGMA table_info(jobs)')}
            for column,kind in (('recording',"TEXT NOT NULL DEFAULT 'standard'"),('approved_by','TEXT'),('approved_at','REAL'),('reservation_id','TEXT'),('artifact_bytes','INTEGER NOT NULL DEFAULT 0')):
                if column not in columns:db.execute(f'ALTER TABLE jobs ADD COLUMN {column} {kind}')
            for row in db.execute('SELECT DISTINCT artifact FROM jobs WHERE artifact_bytes=0').fetchall():
                blob=self.blobs/(row['artifact']+'.py')
                try:
                    metadata=blob.lstat()
                    if not stat.S_ISREG(metadata.st_mode):raise QueueError('unsafe existing artifact')
                except OSError as exc:raise QueueError('existing artifact unavailable during quota migration') from exc
                db.execute('UPDATE jobs SET artifact_bytes=? WHERE artifact=?',(metadata.st_size,row['artifact']))
            db.execute("UPDATE jobs SET status='awaiting_approval' WHERE source='magnetic-step' AND status='pending' AND approved_by IS NULL")
        os.chmod(self.database,0o600)
        self._init_reservations()
    def _connect(self):
        db=sqlite3.connect(self.database,timeout=3)
        db.row_factory=sqlite3.Row
        return db
    @contextmanager
    def _tx(self,*,durable=True):
        db=self._connect()
        try:
            db.execute('PRAGMA synchronous='+('FULL' if durable else 'NORMAL'))
            if not durable:db.execute('PRAGMA wal_autocheckpoint=0')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback();raise
        finally:db.close()
    def _public(self,row, *, worker=False):
        if row is None:raise QueueError('job unavailable')
        result=dict(row)
        result['cancel_requested']=bool(result['cancel_requested'])
        result['result']=None if result['result'] is None else strict_json_loads(result['result'])
        if not worker:
            for field in ('lease','lease_until','worker'):
                result.pop(field,None)
        return result
    def _write_blob(self,source):
        sha=hashlib.sha256(source).hexdigest()
        directory=os.open(self.blobs,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        fd=None
        try:
            try:os.stat(f'{sha}.py',dir_fd=directory,follow_symlinks=False)
            except FileNotFoundError:
                total=count=0
                with os.scandir(self.blobs) as entries:
                    for entry in entries:
                        metadata=entry.stat(follow_symlinks=False)
                        if not stat.S_ISREG(metadata.st_mode):raise QueueError('unsafe artifact storage entry')
                        count+=1;total+=metadata.st_size
                        if count>=MAX_STORE_ARTIFACTS or total+len(source)>MAX_STORE_ARTIFACT_BYTES:raise QueueError('artifact storage quota exceeded')
                if len(source)>MAX_STORE_ARTIFACT_BYTES:raise QueueError('artifact storage quota exceeded')
                if shutil.disk_usage(self.root).free-len(source)<MIN_FREE_DISK_BYTES:raise QueueError('insufficient free disk reserve')
            try:
                fd=os.open(f'{sha}.py',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400,dir_fd=directory)
                with os.fdopen(fd,'wb') as stream:
                    fd=None;stream.write(source);stream.flush();os.fsync(stream.fileno())
            except FileExistsError:
                try:
                    existing=os.open(f'{sha}.py',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
                    with os.fdopen(existing,'rb') as stream:
                        if stream.read(MAX_UPLOAD_BYTES+1)!=source:raise QueueError('artifact integrity failure')
                except OSError as exc:raise QueueError('unsafe artifact path') from exc
            os.fsync(directory)
        except OSError as exc:raise QueueError('artifact storage failed') from exc
        finally:
            if fd is not None:os.close(fd)
            os.close(directory)
        return sha
    def artifact(self,sha):
        if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha):raise QueueError('invalid artifact digest')
        directory=os.open(self.blobs,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            fd=os.open(f'{sha}.py',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
            with os.fdopen(fd,'rb') as stream:source=stream.read(MAX_UPLOAD_BYTES+1)
        except OSError as exc:raise QueueError('artifact unavailable') from exc
        finally:os.close(directory)
        if len(source)>MAX_UPLOAD_BYTES or hashlib.sha256(source).hexdigest()!=sha:raise QueueError('artifact integrity failure')
        return source
    def submit(self,owner,key,source,filename,sensor_source,duration,*,now=None,reservation_id=None,recording='standard'):
        name(owner,'owner');name(key,'idempotency key')
        if recording not in ('standard','detailed'):raise QueueError('recording must be standard or detailed')
        try:inspect_artifact(source,filename)
        except ArtifactError as exc:raise QueueError(str(exc)) from exc
        if sensor_source not in ('magnetic-step','simulation'):raise QueueError('unsupported explicit source selection')
        if type(duration) not in (int,float) or not .05<=duration<=60:raise QueueError('duration must be 0.05..60 seconds')
        now=time.time() if now is None else now
        sha=hashlib.sha256(source).hexdigest()
        with self._tx() as db:
            prior=db.execute('SELECT * FROM jobs WHERE owner=? AND key=?',(owner,key)).fetchone()
            if prior is not None:
                if (prior['artifact'],prior['filename'],prior['source'],prior['duration'],prior['reservation_id'],prior['recording'])!=(sha,filename,sensor_source,duration,reservation_id,recording):raise QueueError('idempotency key already binds a different submission')
                return self._public(prior)
            if db.execute("SELECT count(*) FROM jobs WHERE owner=? AND status IN ('awaiting_approval','pending','claimed','running')",(owner,)).fetchone()[0]>=32:raise QueueError('owner queue limit reached')
            if db.execute('SELECT count(*) FROM jobs WHERE owner=?',(owner,)).fetchone()[0]>=MAX_OWNER_JOBS:raise QueueError('owner lifetime job quota exceeded')
            owned=db.execute('SELECT artifact,MAX(artifact_bytes) AS size FROM jobs WHERE owner=? GROUP BY artifact',(owner,)).fetchall()
            if sha not in {row['artifact'] for row in owned} and sum(row['size'] for row in owned)+len(source)>MAX_OWNER_ARTIFACT_BYTES:
                raise QueueError('owner artifact quota exceeded')
            self._validate_reservation(db,reservation_id,owner,sensor_source,now)
            self._write_blob(source)
            identifier=uuid.uuid4().hex
            db.execute('INSERT INTO jobs (id,owner,key,artifact,filename,source,duration,status,created,updated,reservation_id,artifact_bytes,recording) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (identifier,owner,key,sha,filename,sensor_source,duration,'awaiting_approval' if sensor_source=='magnetic-step' else 'pending',now,now,reservation_id,len(source),recording))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
    def approve(self,identifier,operator,*,artifact,sensor_source,duration):
        """Trusted operator decision bound to the exact immutable job reviewed.

        HTTP role enforcement belongs to the authenticated server; direct Store
        access is trusted local administration, like rig recovery.
        """
        name(identifier);name(operator,'operator')
        if type(duration) not in (int,float):raise QueueError('invalid review duration')
        with self._tx() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
            if row is None:raise QueueError('job unavailable')
            if (artifact,sensor_source,duration)!=(row['artifact'],row['source'],row['duration']):
                raise QueueError('review does not match immutable job')
            if row['source']!='magnetic-step' or row['status'] not in ('awaiting_approval','pending'):
                raise QueueError('job is not awaiting physical execution')
            if row['approved_by'] is None:
                now=time.time()
                db.execute("UPDATE jobs SET status='pending',approved_by=?,approved_at=?,updated=? WHERE id=?",(operator,now,now,identifier))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())

    def get(self,identifier,owner=None):
        name(identifier)
        with closing(self._connect()) as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        if row is None or owner is not None and row['owner']!=owner:raise QueueError('job unavailable')
        return self._public(row)
    def list(self,owner=None):
        with closing(self._connect()) as db:
            rows=db.execute('SELECT * FROM jobs '+('WHERE owner=? ' if owner is not None else '')+'ORDER BY created DESC LIMIT 100',() if owner is None else (owner,)).fetchall()
        return [self._public(row) for row in rows]
    def cancel(self,identifier,owner):
        with self._tx() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=? AND owner=?',(identifier,owner)).fetchone()
            if row is None:raise QueueError('job unavailable')
            if row['status'] in ('awaiting_approval','pending'):db.execute("UPDATE jobs SET status='cancelled',updated=? WHERE id=?",(time.time(),identifier))
            elif row['status'] in ('claimed','running'):db.execute('UPDATE jobs SET cancel_requested=1,updated=? WHERE id=?',(time.time(),identifier))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
    def _recover(self,db,now):
        rows=db.execute("SELECT * FROM jobs WHERE status='running' OR (status='claimed' AND (lease_until<=? OR deadline<=?))",(now,now)).fetchall()
        recovered=0
        for row in rows:
            row=self._effective_lease_row(row,now)
            if now<row["lease_until"] and now<row["deadline"]:continue
            recovered+=1
            uncertain=row['status']=='running'
            result=canonical_json({'error':'worker lease expired','stopped':False if uncertain else True})
            db.execute("UPDATE jobs SET status='failed',result=?,updated=? WHERE id=?",(result,now,row['id']))
            db.execute('UPDATE rigs SET job=NULL,blocked=?,reason=? WHERE id=?',(int(uncertain),'expired active lease; physical stop must be confirmed' if uncertain else None,row['rig']))
        return recovered
    def recover(self,*,now=None):
        with self._tx() as db:return self._recover(db,time.time() if now is None else now)
    def claim(self,worker,rig,*,now=None,sensor_source=None):
        name(worker,'worker');name(rig,'rig')
        if sensor_source is not None and sensor_source not in ('simulation','magnetic-step'):raise QueueError('unsupported explicit source selection')
        now=time.time() if now is None else now
        with self._tx() as db:
            self._recover(db,now)
            db.execute('INSERT OR IGNORE INTO rigs(id) VALUES(?)',(rig,))
            status=db.execute('SELECT * FROM rigs WHERE id=?',(rig,)).fetchone()
            if status['blocked']:raise RigBlocked('rig blocked pending operator recovery')
            if status['job'] is not None:return None
            candidates=db.execute("SELECT * FROM jobs WHERE status='pending' AND (source='simulation' OR approved_by IS NOT NULL)"+(" AND source=?" if sensor_source is not None else "")+" ORDER BY created,id",(sensor_source,) if sensor_source is not None else ())
            row=next((candidate for candidate in candidates if self._reservation_allows(db,candidate,rig,now)),None)
            if row is None:return None
            token=secrets.token_hex(32)
            db.execute("UPDATE jobs SET status='claimed',worker=?,rig=?,lease=?,lease_until=?,deadline=?,updated=? WHERE id=?",(worker,rig,token,now+self.claim_lease_seconds,now+35,now,row['id']))
            db.execute('UPDATE rigs SET job=? WHERE id=?',(row['id'],rig))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(row['id'],)).fetchone(),worker=True)
    def _effective_lease_row(self,row,now):
        return row
    def _lease(self,db,identifier,worker,token,now):
        row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        row=self._effective_lease_row(row,now)
        if row is None or row['status'] not in ('claimed','running') or row['worker']!=worker or not isinstance(token,str) or not hmac.compare_digest(row['lease'],token) or now>=row['lease_until'] or now>=row['deadline']:
            raise QueueError('inactive or invalid worker lease')
        return row
    def start(self,identifier,worker,token,*,now=None):
        now=time.time() if now is None else now
        with self._tx() as db:
            row=self._lease(db,identifier,worker,token,now)
            if row['status']!='claimed' or row['cancel_requested']:raise QueueError('job cannot start')
            if not self._reservation_allows(db,row,row['rig'],now,starting=True):raise QueueError('outside reservation window')
            deadline=now+execution_wall_seconds(row['source'],row['duration'])+1
            db.execute("UPDATE jobs SET status='running',deadline=?,lease_until=?,updated=? WHERE id=?",(deadline,min(now+2,deadline),now,identifier))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone(),worker=True)
    def heartbeat(self,identifier,worker,token,*,now=None):
        now=time.time() if now is None else now
        # Renewal only extends an existing authority. A power-loss rollback of
        # this update expires it earlier; start/approval/finish stay FULL durable.
        # Avoid making every 250ms renewal wait for flash synchronization.
        # Size-based checkpoint work stays on durable authority transitions.
        with self._tx(durable=False) as db:
            row=self._lease(db,identifier,worker,token,now)
            db.execute('UPDATE jobs SET lease_until=?,updated=? WHERE id=?',(min(now+2,row['deadline']),now,identifier))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone(),worker=True)
    def finish(self,identifier,worker,token,status,result,*,now=None):
        if status not in _TERMINAL:raise QueueError('invalid terminal status')
        if not isinstance(result,dict):raise QueueError('result must be an object')
        try:encoded=canonical_json(result)
        except (ValueError,TypeError) as exc:raise QueueError('invalid result') from exc
        if len(encoded.encode())>65536:raise QueueError('result too large')
        now=time.time() if now is None else now
        with self._tx() as db:
            row=self._lease(db,identifier,worker,token,now)
            stopped=result.get('stopped') is True
            board=result.get('board_state')
            hardware_fault=row['source']=='magnetic-step' and (
                result.get('error') in ('hardware_executor_fault','stop_unconfirmed') or
                (isinstance(board,dict) and board.get('fault',0)!=0))
            blocked=not stopped or hardware_fault
            # This acknowledgement is owned by the durable queue, not supplied
            # by a worker. A confirmed STOP alone cannot clear a hardware fault.
            result=dict(result)
            result.pop('operator_recovery_required',None)
            if blocked:result['operator_recovery_required']=True
            encoded=canonical_json(result)
            if len(encoded.encode())>65536:raise QueueError('result too large')
            if hardware_fault:status='failed'
            elif row['cancel_requested'] and stopped:status='cancelled'
            db.execute('UPDATE jobs SET status=?,result=?,updated=? WHERE id=?',(status if stopped else 'failed',encoded,now,identifier))
            reason=('worker cannot confirm physical stop' if not stopped else
                    'hardware fault; operator recovery required' if hardware_fault else None)
            db.execute('UPDATE rigs SET job=NULL,blocked=?,reason=? WHERE id=?',(int(blocked),reason,row['rig']))
            return self._public(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
    def release_block(self,rig,*,stopped_confirmed):
        name(rig,'rig')
        if stopped_confirmed is not True:raise QueueError('operator must confirm stopped rig')
        with self._tx() as db:
            row=db.execute('SELECT * FROM rigs WHERE id=?',(rig,)).fetchone()
            if row is None or row['job'] is not None:raise QueueError('rig unavailable for recovery')
            db.execute('UPDATE rigs SET blocked=0,reason=NULL WHERE id=?',(rig,))
