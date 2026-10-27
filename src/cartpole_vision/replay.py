"""Deterministic offline analysis; input PTS or raw timestamp sidecar is authoritative."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .detector import ArmDetector
from .state import StateEstimator


def annotate(frame, pose, label):
    display=frame.copy()
    if pose is not None:
        p,q=tuple(map(round,pose.pivot)),tuple(map(round,pose.bob))
        cv2.line(display,p,q,(0,240,255),2)
        cv2.circle(display,p,8,(50,60,255),2)
        cv2.circle(display,q,8,(255,160,40),2)
    cv2.rectangle(display,(0,0),(display.shape[1],50),(18,25,30),-1)
    cv2.putText(display,label,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,240,240),2)
    return display


def replay(source, output, home_angle=None, timestamps=None, rotate=0):
    source=Path(source);output=Path(output)
    if not source.is_file():
        raise ValueError(f'Video not found: {source}')
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot decode video: {source}')
    output.mkdir(parents=True,exist_ok=False)
    detector=ArmDetector();estimator=StateEstimator();estimator.home_angle=home_angle
    supplied=None
    if timestamps is not None:
        supplied=[json.loads(line)['t'] for line in Path(timestamps).read_text().splitlines()]
        if not supplied or not np.all(np.isfinite(supplied)) or np.any(np.diff(supplied)<=0):
            cap.release();raise ValueError('Timestamp sidecar must be nonempty, finite, strictly increasing')
    rate=cap.get(cv2.CAP_PROP_FPS)
    rate=rate if np.isfinite(rate) and rate>0 else 30.
    expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows=[];tiles=[];writer=None;last_t=None;fallbacks=0
    try:
        with (output/'states.jsonl').open('w') as log:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if rotate:frame=cv2.rotate(frame,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate])
                i=len(rows)
                if supplied is not None:
                    if i>=len(supplied):raise ValueError('Timestamp sidecar has fewer entries than the video')
                    t=supplied[i]-supplied[0]
                else:
                    t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if not np.isfinite(t) or (last_t is not None and t<=last_t):
                        t=(last_t+1/rate) if last_t is not None else 0.
                        fallbacks+=1
                last_t=t
                start=time.perf_counter();detection=detector.detect(frame)
                state=estimator.update(detection.pose,t)
                state.update(frame=i,detected=detection.pose is not None,
                             detection_status=detection.reason,processing_ms=(time.perf_counter()-start)*1000,
                             timestamp_kind='recorded host arrival' if supplied is not None else 'video presentation time')
                log.write(json.dumps(state,allow_nan=False)+'\n');rows.append(state)
                display=annotate(frame,detection.pose,f'{t:.3f}s | {detection.reason}')
                if writer is None:
                    h,w=display.shape[:2]
                    writer=cv2.VideoWriter(str(output/'overlay.avi'),cv2.VideoWriter_fourcc(*'MJPG'),rate,(w,h))
                    if not writer.isOpened():raise RuntimeError('Cannot create replay overlay video')
                writer.write(display)
                if i%max(1,expected//12)==0 and len(tiles)<12:
                    tiles.append(cv2.resize(display,(480,round(display.shape[0]*480/display.shape[1]))))
        if not rows:raise ValueError('Video contains no decoded frames')
        if supplied is not None and len(supplied)!=len(rows):raise ValueError('Timestamp sidecar has more entries than the video')
    finally:
        cap.release()
        if writer is not None:writer.release()
    counts=Counter(r['detection_status'] for r in rows)
    duration=rows[-1]['t']-rows[0]['t']
    summary=dict(source=str(source.resolve()),frames=len(rows),detected_frames=sum(r['detected'] for r in rows),
                 detection_fraction=float(np.mean([r['detected'] for r in rows])),duration_s=duration,
                 processing_ms_median=float(np.median([r['processing_ms'] for r in rows])),
                 processing_ms_p95=float(np.percentile([r['processing_ms'] for r in rows],95)),
                 reasons=dict(counts),timestamp_fallbacks=fallbacks,
                 rejected_times_s=[r['t'] for r in rows if not r['detected']],
                 limitations=['Detection coverage is not ground-truth accuracy.',
                              'Screen recordings include preview resampling and cannot establish sensor FPS, exposure or latency.',
                              'Overlay AVI has constant-rate playback; source timestamps are printed and saved in states.jsonl.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    if tiles:
        while len(tiles)%3:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(output/'contact.jpg'),np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)]))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--home-angle',type=float,help='Known hanging image angle in radians; otherwise report image angle only')
    parser.add_argument('--timestamps',type=Path,help='raw_times.jsonl from a live recording')
    parser.add_argument('--rotate',type=int,choices=(0,90,180,270),default=0)
    args=parser.parse_args()
    if args.home_angle is not None and not np.isfinite(args.home_angle):parser.error('home angle must be finite')
    try:print(json.dumps(replay(args.source,args.out,args.home_angle,args.timestamps,args.rotate),indent=2))
    except (ValueError,RuntimeError,FileExistsError) as error:parser.exit(1,f'{error}\n')

if __name__=='__main__':main()
