"""Keep encoder/fusion work in its own process; camera submissions never wait."""
import atexit
import multiprocessing as mp
import queue
import threading
import time
from .encoder_feed import Comparison, EncoderReceiver


def _worker(connection,visions,directory,sign,delay,port):
    comparison=Comparison(directory,sign,delay);receiver=None;generation=0
    try:
        if port:
            try:receiver=EncoderReceiver(comparison,port)
            except OSError as error:comparison.receiver_error=str(error)
        connection.send({'ready':True,'error':comparison.receiver_error})
        while True:
            if connection.poll(.002):
                method,kwargs=connection.recv()
                if method=='close':
                    connection.send(None);break
                if method in ('reset','configure'):
                    generation=kwargs.pop('_generation')
                try:
                    result=getattr(comparison,method)(**kwargs)
                    connection.send((True,result))
                except Exception as error:connection.send((False,str(error)))
            for _ in range(8):
                try:sample_generation,state=visions.get_nowait()
                except queue.Empty:break
                # Queue feeder and RPC pipe are independent: draining is not a barrier.
                if sample_generation==generation:comparison.vision(state)
    except (EOFError,BrokenPipeError):pass
    finally:
        if receiver:receiver.close()
        comparison.close();connection.close()


class ComparisonProcess:
    def __init__(self,directory=None,encoder_sign=1,vision_delay_ms=0,port=3096):
        ctx=mp.get_context('spawn')
        parent,child=ctx.Pipe()
        self.connection=parent;self.queue=ctx.Queue(maxsize=8)
        self.lock=threading.RLock()
        self.encoder_sign=encoder_sign;self.vision_delay=vision_delay_ms/1000
        self.dropped=0;self.closed=False;self.error=None;self.generation=0
        self.unavailable=Comparison(None,encoder_sign,vision_delay_ms)
        self.process=ctx.Process(target=_worker,args=(child,self.queue,directory,encoder_sign,vision_delay_ms,port),
                                 name='vision-comparison',daemon=True)
        self.process.start();child.close()
        if not parent.poll(10):
            self.close();raise RuntimeError('Comparison process did not start')
        ready=parent.recv()
        if not ready.get('ready'):self.close();raise RuntimeError('Comparison process failed')
        atexit.register(self.close)

    def _call(self,method,**kwargs):
        with self.lock:
            if self.closed or not self.process.is_alive():raise RuntimeError('Comparison process unavailable')
            self.connection.send((method,kwargs))
            if not self.connection.poll(2):raise RuntimeError('Comparison process timeout')
            ok,result=self.connection.recv()
            if not ok:raise RuntimeError(result)
            return result

    def _failed(self,error):
        # Kill/discard the pipe after any transport failure: a late response must
        # never be consumed as the answer to a different RPC.
        self.error=self.error or str(error)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
        self.close()

    def vision(self,state,now=None):
        if not state.get('angle_valid'):return False
        if not self.lock.acquire(blocking=False):
            self.dropped+=1;return False
        try:
            if self.closed or not self.process.is_alive():return False
            sample={k:state.get(k) for k in ('t','theta_rad','angle_valid','confidence')}
            try:self.queue.put_nowait((self.generation,sample));return True
            except (queue.Full,ValueError,OSError,EOFError):
                self.dropped+=1;return False
        finally:self.lock.release()

    def snapshot(self,now=None,after=None,epoch=None,include_history=True):
        with self.lock:
            try:
                result=self._call('snapshot',now=now,after=after,epoch=epoch,include_history=include_history)
            except (RuntimeError,EOFError,OSError) as error:
                self._failed(error)
                result=self.unavailable.snapshot(now=now,after=after,epoch=epoch,include_history=include_history)
                result['status'].update(source='unavailable',receiver_error=self.error)
            result['status']['fusion_queue_drops']=self.dropped
            return result

    def packet(self,now=None,after_seq=None,session=None):
        with self.lock:
            try:return self._call('packet',now=now,after_seq=after_seq,session=session)
            except (RuntimeError,EOFError,OSError) as error:
                self._failed(error)
                return dict(packet=None,packets=[],session=None,oldest_seq=None,latest_seq=None,history_truncated=False,reset=True,received_t=None,age_s=None,error=self.error)

    def reset(self):
        with self.lock:
            self.generation+=1;self.unavailable.reset()
            try:self._call('reset',_generation=self.generation)
            except (RuntimeError,EOFError,OSError) as error:self._failed(error)

    def configure(self,encoder_sign,vision_delay_ms):
        with self.lock:
            self.generation+=1
            self.unavailable.configure(encoder_sign,vision_delay_ms)
            try:self._call('configure',encoder_sign=encoder_sign,vision_delay_ms=vision_delay_ms,_generation=self.generation)
            except (RuntimeError,EOFError,OSError) as error:self._failed(error)
            self.encoder_sign=encoder_sign;self.vision_delay=vision_delay_ms/1000

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            try:
                if self.process.is_alive():
                    self.connection.send(('close',{}))
                    if self.connection.poll(2):self.connection.recv()
            except (OSError,EOFError):pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=1)
            self.connection.close();self.queue.close()
            self.queue.cancel_join_thread()
