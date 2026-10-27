"""Low-rate JPEG fallback, independent of measurement processing."""
import threading
import cv2
from .replay import annotate


class PreviewWorker:
    def __init__(self,engine,fps=1):
        self.engine=engine;self.period=1/fps;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True,name='jpeg-preview')
        self.thread.start()

    def run(self):
        last=None
        while not self.stop.is_set():
            item=self.engine.video_frame
            if item is not None and item[0]!=last:
                seq,t,frame,pose,status,mask=item
                try:
                    ok,jpg=cv2.imencode('.jpg',annotate(frame,pose,status),[cv2.IMWRITE_JPEG_QUALITY,80])
                    mask_ok,mask_jpg=cv2.imencode('.jpg',mask)
                    with self.engine.lock:
                        if ok:self.engine.jpg=jpg.tobytes();self.engine.preview_seq=seq
                        if mask_ok:self.engine.mask_jpg=mask_jpg.tobytes()
                    last=seq
                except Exception as error:
                    self.engine.preview_error=str(error)
            self.stop.wait(self.period)

    def close(self):
        self.stop.set();self.thread.join(timeout=3)
