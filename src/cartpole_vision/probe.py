"""Measure camera delivery and tracking throughput without running a web server."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .capture import Camera, ArrivalStats
from .detector import ArmDetector
from .replay import annotate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera',default='0');p.add_argument('--fps',type=int,default=80)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=600)
    p.add_argument('--seconds',type=float,default=8);p.add_argument('--rotate',type=int,choices=(0,90,180,270),default=180)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not np.isfinite(a.seconds) or a.seconds<=0:p.error('seconds must be positive')
    a.out.mkdir(parents=True,exist_ok=False)
    cam=Camera(a.camera,a.width,a.height,a.fps,a.rotate);detector=ArmDetector()
    rows=[];last=0;stats=ArrivalStats(a.fps);started=time.monotonic();sampled=False
    try:
        while time.monotonic()-started<a.seconds:
            item=cam.get()
            if item is None or item[0]==last:
                time.sleep(.001);continue
            seq,t,f=item;skipped=seq-last-1 if last else 0;last=seq
            tick=time.perf_counter();result=detector.detect(f);ms=(time.perf_counter()-tick)*1000
            stats.observe(t);rows.append(dict(seq=seq,t=t,processing_ms=ms,detected=result.pose is not None,skipped=skipped))
            if not sampled and time.monotonic()-started>2:
                cv2.imwrite(str(a.out/'raw.png'),f)
                cv2.imwrite(str(a.out/'overlay.jpg'),annotate(f,result.pose,result.reason));sampled=True
        report=dict(camera=cam.report(),tracking=stats.report(),processed_frames=len(rows),
                    detected_frames=sum(r['detected'] for r in rows),skipped_frames=sum(r['skipped'] for r in rows),
                    processing_ms_median=float(np.median([r['processing_ms'] for r in rows])) if rows else None,
                    processing_ms_p95=float(np.percentile([r['processing_ms'] for r in rows],95)) if rows else None)
        (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        (a.out/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        print(json.dumps(report,indent=2))
    finally:cam.close()

if __name__=='__main__':main()
