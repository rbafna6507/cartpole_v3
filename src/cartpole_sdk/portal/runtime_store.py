"""Single-process runtime authority, independent of durable queue I/O.

Only the API owning this Store may manage its active jobs. Durable running rows
are recovery markers, never sufficient authority after an API restart. Runtime
renewal checks identity, cancellation, monotonic freshness and a fixed deadline
without opening SQLite. Losing this process therefore revokes every live run.
"""
import fcntl
import os
import weakref
import sqlite3
from pathlib import Path
import hmac
import threading
import time
from .queue import Store, QueueError
from .budget import execution_wall_seconds

class RuntimeStore(Store):
    claim_lease_seconds=35
    def __init__(self,root,**kwargs):
        self._closed=False
        directory=Path(root).resolve();directory.mkdir(parents=True,exist_ok=True)
        fd=os.open(directory/'.runtime-owner.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            os.close(fd);raise QueueError('runtime authority already owned')
        self._release_owner=weakref.finalize(self,os.close,fd)
        self._runtime={};self._cancel_pending={}
        self._runtime_lock=threading.Lock()
        self._drained=threading.Condition(self._runtime_lock);self._connections=0
        try:super().__init__(root,**kwargs)
        except BaseException:self.close();raise

    def close(self):
        with self._drained:
            self._closed=True;self._runtime.clear();self._cancel_pending.clear()
            while self._connections:self._drained.wait()
        self._release_owner()

    def _connect(self):
        with self._drained:
            if self._closed:raise QueueError('runtime authority closed')
            self._connections+=1
        store=self
        class Connection(sqlite3.Connection):
            released=False
            def close(connection):
                if connection.released:return
                try:super().close()
                finally:
                    connection.released=True
                    with store._drained:
                        store._connections-=1;store._drained.notify_all()
        try:
            db=sqlite3.connect(self.database,timeout=3,factory=Connection)
            db.row_factory=sqlite3.Row
            return db
        except BaseException:
            with self._drained:self._connections-=1;self._drained.notify_all()
            raise

    def _public(self,row,*,worker=False):
        value=super()._public(row,worker=worker)
        if worker and value['status']=='claimed':value['setup_heartbeat_required']=False
        return value

    def _effective_lease_row(self,row,now):
        if row is None or row['status']!='running':return row
        with self._runtime_lock:
            entry=self._runtime.get(row['id'])
            if entry is not None and time.monotonic()<min(entry['expires'],entry['end']):
                # Return a copy: database transactions never hold the runtime lock.
                mono=time.monotonic()
                return dict(entry['row'],lease_until=now+entry['expires']-mono,deadline=now+entry['end']-mono)
            self._runtime.pop(row['id'],None)
        return dict(row,lease_until=0)

    def start(self,identifier,worker,token,*,now=None):
        now=time.time() if now is None else now
        if self._closed:raise QueueError('runtime authority closed')
        installed=False
        preparation_began=time.monotonic()
        try:
            with self._tx() as db:
                row=self._lease(db,identifier,worker,token,now)
                if row['status']!='claimed' or row['cancel_requested']:raise QueueError('job cannot start')
                if not self._reservation_allows(db,row,row['rig'],now,starting=True):raise QueueError('outside reservation window')
                preparation_seconds=min(35,max(0,row['deadline']-now))
                reservation_end=None
                if row['source']=='magnetic-step':
                    if row['reservation_id']:
                        reservation_end=db.execute('SELECT ends_at FROM reservations WHERE id=?',(row['reservation_id'],)).fetchone()[0]
                    else:
                        upcoming=db.execute('SELECT MIN(starts_at) FROM reservations WHERE rig=? AND cancelled=0 AND starts_at>=?',(row['rig'],now)).fetchone()[0]
                        reservation_end=upcoming
                reservation_bound=None if reservation_end is None else preparation_began+max(0,reservation_end-now)
                # The durable marker conservatively reserves the longest possible
                # preparation+run horizon; actual authority uses monotonic time.
                deadline=now+preparation_seconds+execution_wall_seconds(row['source'],row['duration'])+1
                db.execute("UPDATE jobs SET status='running',deadline=?,lease_until=?,updated=? WHERE id=?",(deadline,min(now+2,deadline),now,identifier))
                current=dict(db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone())
                preparation_end=preparation_began+preparation_seconds
                with self._runtime_lock:
                    if self._closed:raise QueueError('runtime authority closed')
                    if self._cancel_pending.get(identifier)==current['owner']:current['cancel_requested']=1
                    self._runtime[identifier]={'row':current,'expires':preparation_end,'end':preparation_end,'ready':False}
                    installed=True
            # Never publish authority until the running marker is committed.
            with self._runtime_lock:
                if self._closed:raise QueueError('runtime authority closed')
                entry=self._runtime[identifier]
                if time.monotonic()>=min(entry['expires'],entry['end']):raise QueueError('start commit exceeded lease deadline')
                mono=time.monotonic()
                entry['expires']=mono+2
                entry['end']=mono+execution_wall_seconds(current['source'],current['duration'])+1
                if reservation_bound is not None:entry['end']=min(entry['end'],reservation_bound)
                if entry['end']<=mono:raise QueueError('reservation elapsed during start commit')
                entry['expires']=min(entry['expires'],entry['end'])
                entry['ready']=True
                return self._public(dict(entry['row']),worker=True)
        except BaseException:
            if installed:
                with self._runtime_lock:self._runtime.pop(identifier,None)
            raise

    def heartbeat(self,identifier,worker,token,*,now=None):
        now=time.time() if now is None else now
        with self._runtime_lock:
            entry=self._runtime.get(identifier)
            if entry is None or not entry['ready']:raise QueueError('runtime authority unavailable')
            row=entry['row'];mono=time.monotonic()
            if mono>=min(entry['expires'],entry['end']):
                self._runtime.pop(identifier,None)
                raise QueueError('runtime authority expired')
            if (row['worker']!=worker or not isinstance(token,str) or not hmac.compare_digest(row['lease'],token)):
                raise QueueError('inactive or invalid worker lease')
            entry['expires']=min(mono+2,entry['end'])
            row['lease_until']=now+entry['expires']-mono;row['deadline']=now+entry['end']-mono;row['updated']=now
            return self._public(dict(row),worker=True)|{'runtime_remaining_s':entry['end']-mono}

    def cancel(self,identifier,owner):
        # Cancellation is sticky even if persisting it stalls or fails. Only the
        # authenticated owner can revoke this runtime; unknown jobs use Store.
        with self._runtime_lock:
            entry=self._runtime.get(identifier)
            if entry is not None:
                if entry['row']['owner']!=owner:raise QueueError('job unavailable')
                entry['row']['cancel_requested']=1
        if entry is None:
            # Authenticate against a WAL read before registering a bounded intent
            # for an existing job. It may be between claimed and runtime publish.
            existing=self.get(identifier,owner)
            if existing['status'] in ('claimed','running'):
                with self._runtime_lock:
                    self._cancel_pending[identifier]=owner
                    entry=self._runtime.get(identifier)
                    if entry is not None:entry['row']['cancel_requested']=1
        try:return super().cancel(identifier,owner)
        finally:
            # start may have published while cancel waited for its transaction.
            with self._runtime_lock:
                entry=self._runtime.get(identifier)
                if entry is not None and entry['row']['owner']==owner:
                    entry['row']['cancel_requested']=1

    def finish(self,identifier,worker,token,status,result):
        finished=super().finish(identifier,worker,token,status,result)
        with self._runtime_lock:
            self._runtime.pop(identifier,None);self._cancel_pending.pop(identifier,None)
        return finished

    def _recover(self,db,now):
        count=super()._recover(db,now)
        active={row[0] for row in db.execute("SELECT id FROM jobs WHERE status IN ('claimed','running')")}
        with self._runtime_lock:
            for identifier in set(self._runtime)|set(self._cancel_pending):
                if identifier not in active:
                    self._runtime.pop(identifier,None);self._cancel_pending.pop(identifier,None)
        return count
