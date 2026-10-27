"""Paced 80 FPS synthetic source through the real engine/optional H.264 worker.

No camera, network listener, power change or hardware control. Reports local
service/encode latency only; rendering the input cache is outside the trial.
"""
import argparse
import json
import math
from pathlib import Path
import queue
import threading
import time
import cv2
from .app import Engine
from .evaluate import render_arm,accuracy
from .metrics import summarize
from .stream import VideoStream


def run(output,seconds=8,video_fps=0,width=640,mode='tracked'):
    if seconds<2:raise ValueError('Use at least two seconds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2)
    images=[render_arm(2*math.pi*i/80) for i in range(80)]
    class Source:
        latest=None
        def get(self):return self.latest
    source=Source();engine=Engine(80,detector_width=width)
    engine.estimator.home_angle=0;engine.estimator.home_length=180
    stop=threading.Event();video=None;viewer=None;rows=[];truth=[]
    consumer=None;encoded_packets=[]
    if video_fps:
        video=VideoStream(source,engine,960,600,video_fps,4000)
        video.set_mode(mode);viewer=video.hub.subscribe()
        def consume():
            while not stop.is_set():
                try:item=viewer.get(timeout=.1)
                except queue.Empty:continue
                if item is None:break
                encoded_packets.append(len(item))
        consumer=threading.Thread(target=consume,daemon=True);consumer.start()
    start=time.monotonic()
    def produce():
        last=-1
        while not stop.is_set():
            now=time.monotonic();i=int((now-start)*80)
            if i!=last:
                source.latest=(i+1,now,images[i%80]);last=i
                truth.append(dict(t=now,image_angle_rad=2*math.pi*i/80))
            stop.wait(.0005)
    producer=threading.Thread(target=produce,daemon=True);producer.start()
    try:
        last=None
        while time.monotonic()-start<seconds:
            item=source.get()
            if item is None or item[0]==last:stop.wait(.0005);continue
            seq,t,frame=item;last=seq;rows.append(engine.process(frame,t,seq))
    finally:
        stop.set();producer.join(timeout=2)
        if consumer:consumer.join(timeout=2)
        report_video=video.report() if video else None
        if video:video.close()
    steady=[r for r in rows if r['t']>=start+1]
    report=dict(kind='paced synthetic source; no physical camera',seconds=seconds,detector_width=width,
                video_fps=video_fps,video_mode=mode,video=report_video,packets_consumed=len(encoded_packets),
                all=summarize(rows),after_one_second_warmup=summarize(steady),
                synthetic_angle_accuracy=accuracy(rows,truth)['angle_error_deg'],
                limitations=['Includes engine and optional software H.264 contention; excludes USB/camera capture, network, and browser.',
                             'Input cache is rendered before timing. Other existing Jetson processes are not stopped.',
                             'No velocity accuracy claim: discrete cached angles have jittery host-publish timestamps.'])
    (output/'states.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--video-fps',type=int,default=0);p.add_argument('--detector-width',type=int,default=640)
    p.add_argument('--mode',choices=('camera','tracked','mask'),default='tracked')
    a=p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds<2 or a.video_fps<0 or a.detector_width<=0:p.error('Invalid duration/rate/width')
    print(json.dumps(run(a.out,a.seconds,a.video_fps,a.detector_width,a.mode),indent=2,allow_nan=False))


if __name__=='__main__':main()
