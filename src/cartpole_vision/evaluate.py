"""Offline, reproducible motion evaluation. Never opens a camera."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from .metrics import summarize, percentiles


def error_stats(values):
    values=np.asarray(values,float)
    return dict(samples=len(values),rmse=float(np.sqrt(np.mean(values**2))) if len(values) else None,
                bias=float(np.mean(values)) if len(values) else None,
                abs_p95=float(np.percentile(abs(values),95)) if len(values) else None,
                abs_max=float(np.max(abs(values))) if len(values) else None)


def accuracy(rows,reference,max_gap=.1):
    """Reference image angle is positive image-right from down, in radians.

    Optional image_omega_rad_s has that same sign (opposite dashboard theta).
    Times must share the states' time origin. Never extrapolate reference data.
    """
    times=np.array([r['t'] for r in reference],float)
    if not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('Reference times must be finite and strictly increasing')
    angles=np.unwrap([r['image_angle_rad'] for r in reference])
    speeds=np.array([r.get('image_omega_rad_s',np.nan) for r in reference],float)
    if not np.all(np.isfinite(angles)):raise ValueError('Reference angles must be finite')
    def interpolate(t,values):
        if t is None or not np.isfinite(t) or t<times[0] or t>times[-1]:return None
        i=int(np.searchsorted(times,t))
        if i<len(times) and abs(times[i]-t)<1e-9:
            return float(values[i]) if np.isfinite(values[i]) else None
        if i==0 or i==len(times) or times[i]-times[i-1]>max_gap:return None
        if not np.isfinite(values[i-1:i+1]).all():return None
        return float(np.interp(t,times,values))
    angle_errors=[];speed_errors=[];current_errors=[]
    for row in rows:
        expected=interpolate(row['t'],angles)
        if expected is not None and row.get('image_angle_rad') is not None:
            delta=row['image_angle_rad']-expected
            angle_errors.append(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
        if row.get('omega_rad_s') is not None:
            expected=interpolate(row.get('omega_effective_t'),speeds)
            if expected is not None:speed_errors.append(-row['omega_rad_s']-expected)
            expected=interpolate(row['t'],speeds)
            if expected is not None:current_errors.append(-row['omega_rad_s']-expected)
    return dict(angle_error_deg=error_stats(angle_errors),omega_effective_error_rad_s=error_stats(speed_errors),
                omega_current_frame_error_rad_s=error_stats(current_errors),
                reference_samples=len(reference),reference_max_interpolation_gap_s=max_gap)


def render_arm(angle,pivot=(480,280),length=180,blur_radians=0):
    """Known subpixel geometry, antialiased at 2x. No image used for tuning."""
    frames=[]
    for a in np.linspace(angle-blur_radians/2,angle+blur_radians/2,5) if blur_radians else [angle]:
        image=np.full((1200,1920,3),35,np.uint8)
        cv2.rectangle(image,(60,60),(220,230),(80,190,0),-1)
        p=np.array(pivot,float)*2;bob=p+2*length*np.array([math.sin(a),math.cos(a)])
        point=lambda xy:tuple(np.rint(xy*256).astype(int))
        cv2.line(image,point(p),point(bob),(90,200,10),44,cv2.LINE_AA,8)
        cv2.circle(image,point(bob),58*256,(90,200,10),-1,cv2.LINE_AA,8)
        cv2.circle(image,point(p),12*256,(25,25,25),-1,cv2.LINE_AA,8)
        for h in np.arange(0,2*math.pi,math.pi/3):
            xy=bob+38*np.array([math.cos(h),math.sin(h)])
            cv2.circle(image,point(xy),6*256,(25,25,25),-1,cv2.LINE_AA,8)
        frames.append(cv2.resize(image,(960,600),interpolation=cv2.INTER_AREA))
    return np.mean(frames,axis=0).astype(np.uint8) if len(frames)>1 else frames[0]


def load_package(path,name):
    path=Path(path)/'pendulum_vision'
    spec=importlib.util.spec_from_file_location(name,path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return importlib.import_module(name+'.detector').ArmDetector,importlib.import_module(name+'.state').StateEstimator


def benchmark(output,baseline=None,frames=320,width=640,source=None,timestamps=None,threads=2):
    from .detector import ArmDetector
    from .state import StateEstimator
    cv2.setNumThreads(threads)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    engines=[('vision3',ArmDetector(processing_width=width),StateEstimator())]
    if baseline:
        detector,estimator=load_package(baseline,'baseline_vision')
        engines.insert(0,('vision2',detector(),estimator()))
    logs={name:[] for name,_,_ in engines};reference=[];differences=[]
    for _,_,estimator in engines:estimator.home_angle=0
    cap=cv2.VideoCapture(str(source)) if source else None
    if cap is not None and not cap.isOpened():raise ValueError('Cannot open saved video')
    supplied_rows=[json.loads(s) for s in Path(timestamps).read_text().splitlines()] if timestamps else None
    supplied=[r['t'] for r in supplied_rows] if supplied_rows is not None else None
    if supplied is not None and (not supplied or not np.isfinite(supplied).all() or np.any(np.diff(supplied)<=0)):
        raise ValueError('Timestamp sidecar must be finite and strictly increasing')
    rate=cap.get(cv2.CAP_PROP_FPS) if cap else 80
    rate=rate if math.isfinite(rate) and rate>0 else 80
    started=time.time()
    try:
        for i in range(frames):
            t=i/80;seq=i
            if cap:
                ok,frame=cap.read()
                if not ok:break
                if supplied is not None:
                    if i>=len(supplied):raise ValueError('Timestamp sidecar shorter than video')
                    t=supplied[i]-supplied[0]
                    seq=supplied_rows[i].get('seq',i)
                else:t=i/rate
            else:
                # Continuously rotating arm: 4 rad/s with acceleration and an
                # exposure-sized blur arc. Known truth for angle and velocity.
                angle=4*t+.35*math.sin(2*math.pi*.7*t)
                omega=4+.35*2*math.pi*.7*math.cos(2*math.pi*.7*t)
                frame=render_arm(angle,blur_radians=omega*.002)
                reference.append(dict(t=t,image_angle_rad=angle,image_omega_rad_s=omega))
            poses={}
            # Alternate order to reduce thermal/scheduling bias.
            for name,detector,estimator in (engines if i%2 else engines[::-1]):
                tick=time.perf_counter();detection=detector.detect(frame)
                state=estimator.update(detection.pose,t)
                elapsed=(time.perf_counter()-tick)*1000
                state.update(seq=seq,detected=detection.pose is not None,detection_status=detection.reason,
                             processing_ms=elapsed)
                if state.get('omega_effective_t') is not None:state['omega_delay_s']=t-state['omega_effective_t']
                logs[name].append(state);poses[name]=detection.pose
            if len(poses)==2 and all(p is not None for p in poses.values()):
                d=poses['vision3'].angle-poses['vision2'].angle
                differences.append(math.degrees(math.atan2(math.sin(d),math.cos(d))))
    finally:
        if cap:cap.release()
    reports={}
    for name,rows in logs.items():
        (output/f'{name}-states.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
        reports[name]=summarize(rows,80)
        for field in ('processed_fps','angle_fps','velocity_fps'):
            reports[name]['source_timeline_'+field]=reports[name].pop(field)
        reports[name]['serial_compute_capacity_fps']=1000/np.mean([r['processing_ms'] for r in rows]) if rows else 0
        if reference:reports[name]['synthetic_accuracy']=accuracy(rows,reference)
    result=dict(kind='saved video' if source else 'synthetic rotating arm with 2 ms simulated exposure',
                source=str(source) if source else None,started_unix=started,opencv_threads=threads,
                time_basis='unpaced source timeline; use service time or paced benchmark for throughput',
                detector_width=width,results=reports,paired_angle_difference_deg=error_stats(differences),
                limitations=['Offline service time excludes camera capture, video encoding, network and browser.',
                             'Saved-video detection coverage is not physical accuracy.',
                             'Synthetic accuracy is against rendered geometry, not an independent physical reference.'])
    if reference:
        with (output/'truth.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=reference[0].keys());writer.writeheader();writer.writerows(reference)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    bench=sub.add_parser('benchmark');bench.add_argument('--out',type=Path,required=True)
    bench.add_argument('--baseline',type=Path);bench.add_argument('--frames',type=int,default=320)
    bench.add_argument('--detector-width',type=int,default=640);bench.add_argument('--threads',type=int,default=2)
    bench.add_argument('--source',type=Path);bench.add_argument('--timestamps',type=Path)
    log=sub.add_parser('log');log.add_argument('states',type=Path);log.add_argument('--out',type=Path,required=True)
    log.add_argument('--reference',type=Path);log.add_argument('--target-fps',type=float,default=80)
    a=parser.parse_args()
    if a.command=='benchmark':
        if min(a.frames,a.detector_width,a.threads)<=0:parser.error('Counts and dimensions must be positive')
        result=benchmark(a.out,a.baseline,a.frames,a.detector_width,a.source,a.timestamps,a.threads)
    else:
        if not math.isfinite(a.target_fps) or a.target_fps<=0:parser.error('target FPS must be positive')
        rows=[json.loads(line) for line in a.states.read_text().splitlines() if line.strip()]
        result=summarize(rows,a.target_fps)
        if a.reference:
            with a.reference.open() as f:
                reference=[{k:float(v) for k,v in row.items() if v!=''} for row in csv.DictReader(f)]
            result['reference_accuracy']=accuracy(rows,reference)
        a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
