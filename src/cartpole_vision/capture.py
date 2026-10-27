"""Latest-frame camera capture and honest host-delivery diagnostics."""
from collections import deque
import json
import math
import shutil
from pathlib import Path
import sys
import threading
import time
import cv2
import numpy as np


class ArrivalStats:
    def __init__(self, target_fps):
        self.target_fps=target_fps
        self.times=deque(maxlen=max(240,int(target_fps*3)))

    def observe(self,t):
        self.times.append(t)

    def report(self):
        times=np.asarray(self.times)
        dt=np.diff(times)
        fps=float((len(times)-1)/(times[-1]-times[0])) if len(times)>1 and times[-1]>times[0] else None
        return dict(delivered_fps=fps,arrival_interval_p95_ms=float(np.percentile(dt,95)*1000) if len(dt) else None,
                    target_rate_met=bool(fps is not None and len(times)>=self.target_fps and fps>=self.target_fps*.95),
                    timing_note='Host read completion timestamps; not sensor timestamps or end-to-end latency')


class Camera:
    def __init__(self, source='0', width=960, height=600, fps=80, rotate=180,
                 exposure_raw=None, auto_exposure_raw=None, gain_raw=None):
        if min(width,height,fps)<=0 or rotate not in (0,90,180,270):
            raise ValueError('Invalid camera dimensions, rate or rotation')
        backend=cv2.CAP_AVFOUNDATION if sys.platform=='darwin' else cv2.CAP_V4L2
        src=int(source) if str(source).isdigit() else str(source)
        self.cap=cv2.VideoCapture(src,backend)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError('Cannot open camera. Close competing camera apps and allow camera access for this terminal/application.')
        self.lock=threading.Lock();self.stop=threading.Event();self.latest=None;self.seq=0
        self.rotation=rotate;self.stats=ArrivalStats(fps);self.read_failures=0;self.error=None
        self.settings=dict(requested_width=width,requested_height=height,requested_fps=fps,
                           backend=self.cap.getBackendName(),rotation=rotate,controls={})
        props=[('width',cv2.CAP_PROP_FRAME_WIDTH,width),('height',cv2.CAP_PROP_FRAME_HEIGHT,height),
               ('fps',cv2.CAP_PROP_FPS,fps),('buffer',cv2.CAP_PROP_BUFFERSIZE,1)]
        # This AR0234 family advertises uncompressed YUY2, not MJPEG.
        if sys.platform!='darwin':
            props.insert(0,('fourcc',cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'YUYV')))
        for name,prop,value in props:
            self.settings['controls'][name]={'requested':value,'accepted':bool(self.cap.set(prop,value)),
                                             'readback':self._get(prop)}
        for name,prop,value in [('auto_exposure_raw',cv2.CAP_PROP_AUTO_EXPOSURE,auto_exposure_raw),
                                ('exposure_raw',cv2.CAP_PROP_EXPOSURE,exposure_raw),
                                ('gain_raw',cv2.CAP_PROP_GAIN,gain_raw)]:
            if value is not None:
                self.settings['controls'][name]={'requested':value,'accepted':bool(self.cap.set(prop,value)),
                                                 'readback':self._get(prop)}
        self.settings.update(driver_fps=self._get(cv2.CAP_PROP_FPS),
                             driver_auto_exposure_raw=self._get(cv2.CAP_PROP_AUTO_EXPOSURE),
                             driver_exposure_raw=self._get(cv2.CAP_PROP_EXPOSURE),
                             driver_gain_raw=self._get(cv2.CAP_PROP_GAIN))
        self.thread=threading.Thread(target=self._run,name='camera-reader',daemon=True)
        self.thread.start()

    def _get(self,prop):
        value=float(self.cap.get(prop))
        return value if math.isfinite(value) else None

    def _run(self):
        try:
            while not self.stop.is_set():
                ok,frame=self.cap.read();t=time.monotonic()
                if not ok:
                    with self.lock:self.read_failures+=1
                    self.stop.wait(.01);continue
                if self.rotation:
                    frame=cv2.rotate(frame,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[self.rotation])
                with self.lock:
                    self.seq+=1;self.latest=(self.seq,t,frame)
                    self.stats.observe(t)
                    self.settings['actual_frame_size']=[frame.shape[1],frame.shape[0]]
        except Exception as error:
            with self.lock:self.error=str(error)
        finally:
            self.cap.release()

    def get(self):
        with self.lock:return self.latest

    def report(self):
        with self.lock:
            report=self.stats.report()
            report.update(settings=self.settings.copy(),read_failures=self.read_failures,error=self.error,
                          frames_delivered=self.seq,
                          latest_age_s=None if self.latest is None else time.monotonic()-self.latest[1])
        if report['latest_age_s'] is None or report['latest_age_s']>.15:report['target_rate_met']=False
        return report

    def close(self):
        self.stop.set();self.thread.join(timeout=3)


class RawRecorder:
    """Records processed raw frames, not a promise to save every sensor frame.

AVI is constant-rate for convenience. raw_times.jsonl preserves actual times
and camera sequence numbers and must be passed to replay for velocity analysis.
"""
    def __init__(self, directory, fps, max_bytes=512 * 1024 * 1024, min_free_bytes=256 * 1024 * 1024):
        self.max_bytes=max_bytes;self.min_free_bytes=min_free_bytes
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.fps=fps;self.writer=None
        self.timestamps=(self.directory/'raw_times.jsonl').open('w')
        self.shape=None;self.count=0

    def write(self,frame,seq,t):
        # Reserve a conservative uncompressed frame plus container overhead;
        # stop explicit captures at a budget rather than filling unattended disks.
        allowance=frame.nbytes+65536
        video=self.directory/'raw.avi'
        size=(video.stat().st_size if video.exists() else 0)+self.timestamps.tell()
        if size+allowance>self.max_bytes:
            raise OSError('Raw recording size limit reached')
        if shutil.disk_usage(self.directory).free<self.min_free_bytes+allowance:
            raise OSError('Raw recording paused: free space below reserve')
        if self.writer is None:
            self.shape=frame.shape
            h,w=frame.shape[:2]
            self.writer=cv2.VideoWriter(str(self.directory/'raw.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,(w,h))
            if not self.writer.isOpened():raise RuntimeError('Unable to open raw recorder')
        if frame.shape!=self.shape:raise RuntimeError('Camera dimensions changed during recording')
        self.writer.write(frame)
        self.timestamps.write(json.dumps(dict(frame=self.count,seq=seq,t=t))+'\n')
        self.count+=1
        if self.count%80==0:self.timestamps.flush()

    def close(self):
        if self.writer is not None:self.writer.release()
        self.timestamps.close()


class BufferedRecorder:
    """A bounded queue isolates encoding/disk latency from pose processing."""
    def __init__(self,directory,fps,capacity=32):
        import queue
        self.queue=queue.Queue(maxsize=capacity)
        self.recorder=None;self.thread=None
        self.dropped=0;self.error=None;self.closed=False
        try:self.recorder=RawRecorder(directory,fps)
        except (OSError,RuntimeError) as error:
            self.error=str(error);return
        self.thread=threading.Thread(target=self._run,name='video-recorder',daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                self.recorder.write(*item)
        except Exception as error:
            self.error=str(error)
        finally:
            try:self.recorder.close()
            except (OSError,RuntimeError) as error:self.error=self.error or str(error)

    def write(self,frame,seq,t):
        import queue
        if self.closed:raise RuntimeError('Recorder is closed')
        if self.error:self.dropped+=1;return
        try:self.queue.put_nowait((frame,seq,t))
        except queue.Full:self.dropped+=1

    def close(self):
        if self.closed:return
        self.closed=True
        # Never block forever if the writer failed while the queue was full.
        import queue
        if self.thread is None:return
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.2);break
            except queue.Full:continue
        self.thread.join(timeout=5)
        if self.thread.is_alive():self.error='Recorder did not finish within 5 seconds'
