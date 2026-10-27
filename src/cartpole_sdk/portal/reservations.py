"""Optional exclusive experiment windows; never a motion or approval grant."""
from contextlib import closing
import math
import time
import uuid


class ReservationStoreMixin:
    def _init_reservations(self):
        with self._tx() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS reservations (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, rig TEXT NOT NULL,
                starts_at REAL NOT NULL, ends_at REAL NOT NULL,
                cancelled INTEGER NOT NULL DEFAULT 0)''')

    def reserve(self,owner,rig,starts_at,duration_minutes,*,now=None):
        from .queue import QueueError, name
        if not self.reservations_enabled:raise QueueError('reservations disabled')
        name(owner,'owner');name(rig,'rig')
        now=time.time() if now is None else now
        if (type(starts_at) not in (int,float) or not math.isfinite(starts_at)
                or not now-5<=starts_at<=now+7*86400):
            raise QueueError('reservation start must be now or within seven days')
        if type(duration_minutes) not in (int,float) or not math.isfinite(duration_minutes) or not 1<=duration_minutes<=30:
            raise QueueError('reservation duration must be 1..30 minutes')
        starts_at=max(starts_at,now);ends_at=starts_at+duration_minutes*60
        with self._tx() as db:
            if db.execute('SELECT 1 FROM reservations WHERE rig=? AND cancelled=0 AND starts_at<? AND ends_at>?',(rig,ends_at,starts_at)).fetchone():
                raise QueueError('reservation overlaps another window')
            # A reservation cannot displace an already claimed/running experiment.
            if db.execute("SELECT 1 FROM jobs WHERE rig=? AND status IN ('claimed','running') AND deadline>?",(rig,starts_at)).fetchone():
                raise QueueError('rig has an active experiment')
            if db.execute('SELECT count(*) FROM reservations WHERE owner=? AND cancelled=0 AND ends_at>?',(owner,now)).fetchone()[0]>=8:
                raise QueueError('reservation limit reached')
            identifier=uuid.uuid4().hex
            db.execute('INSERT INTO reservations(id,owner,rig,starts_at,ends_at) VALUES(?,?,?,?,?)',(identifier,owner,rig,starts_at,ends_at))
            return self._reservation_public(db.execute('SELECT * FROM reservations WHERE id=?',(identifier,)).fetchone(),now)

    @staticmethod
    def _reservation_public(row,now):
        result=dict(row)
        result['status']='cancelled' if result.pop('cancelled') else 'expired' if now>=result['ends_at'] else 'active' if now>=result['starts_at'] else 'scheduled'
        return result

    def reservations(self,owner=None,*,now=None):
        now=time.time() if now is None else now
        with closing(self._connect()) as db:
            rows=db.execute('SELECT * FROM reservations '+('WHERE owner=? ' if owner is not None else '')+'ORDER BY starts_at DESC LIMIT 100',() if owner is None else (owner,)).fetchall()
        return [self._reservation_public(row,now) for row in rows]

    def cancel_reservation(self,identifier,owner):
        from .queue import QueueError, name
        name(identifier);name(owner,'owner')
        with self._tx() as db:
            row=db.execute('SELECT * FROM reservations WHERE id=? AND owner=?',(identifier,owner)).fetchone()
            if row is None:raise QueueError('reservation unavailable')
            if db.execute("SELECT 1 FROM jobs WHERE reservation_id=? AND status IN ('claimed','running')",(identifier,)).fetchone():
                raise QueueError('reservation has an active experiment')
            db.execute('UPDATE reservations SET cancelled=1 WHERE id=?',(identifier,))
            return self._reservation_public(db.execute('SELECT * FROM reservations WHERE id=?',(identifier,)).fetchone(),time.time())

    def _validate_reservation(self,db,identifier,owner,source,now):
        from .queue import QueueError, name
        if identifier is None:return
        name(identifier,'reservation')
        if not self.reservations_enabled:raise QueueError('reservations disabled')
        row=db.execute('SELECT * FROM reservations WHERE id=? AND owner=? AND cancelled=0 AND ends_at>?',(identifier,owner,now)).fetchone()
        if row is None or source!='magnetic-step':raise QueueError('reservation unavailable for submission')

    def _reservation_allows(self,db,row,rig,now,*,starting=False):
        if row['source']=='simulation':return True
        # Reserve setup, duration and stop overhead; start rechecks the same gate.
        end=now+row['duration']+(1 if starting else 6)
        if row['reservation_id']:
            if not self.reservations_enabled:return False
            r=db.execute('SELECT * FROM reservations WHERE id=?',(row['reservation_id'],)).fetchone()
            return bool(r and not r['cancelled'] and r['owner']==row['owner'] and r['rig']==rig and r['starts_at']<=now and end<=r['ends_at'])
        return db.execute('SELECT 1 FROM reservations WHERE rig=? AND cancelled=0 AND starts_at<? AND ends_at>?',(rig,end,now)).fetchone() is None
