"""Measurement coverage and timing, independent of UI/stream throughput."""
from collections import Counter, deque
import math
import numpy as np


def percentiles(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    if not values: return dict(p50=None,p95=None,p99=None,max=None)
    return dict(zip(('p50','p95','p99','max'),map(float,np.percentile(values,[50,95,99,100]))))


def counts(rows):
    result=Counter(processed_frames=len(rows),skipped_camera_frames=0,detected_frames=0,
                   angle_frames=0,velocity_frames=0,calibrated_camera_frames=0)
    previous=None
    for row in rows:
        skipped=max(0,row['seq']-previous-1) if previous is not None else 0
        previous=row['seq']
        result['skipped_camera_frames']+=skipped
        result['detected_frames']+=bool(row.get('detected'))
        result['angle_frames']+=bool(row.get('angle_valid'))
        result['velocity_frames']+=row.get('omega_rad_s') is not None
        if row.get('calibrated'): result['calibrated_camera_frames']+=1+skipped
    return result


def coverage(result):
    result=dict(result)
    total=result['processed_frames']+result['skipped_camera_frames']
    calibrated=result['calibrated_camera_frames']
    result.update(camera_frames_in_span=total,
                  geometry_coverage=result['detected_frames']/total if total else None,
                  angle_coverage_calibrated=result['angle_frames']/calibrated if calibrated else None,
                  velocity_coverage_calibrated=result['velocity_frames']/calibrated if calibrated else None)
    return result


def summarize(rows, target_fps=80):
    rows=list(rows); result=coverage(counts(rows))
    duration=rows[-1]['t']-rows[0]['t'] if len(rows)>1 else 0
    result.update(duration_s=duration,processed_fps=(len(rows)-1)/duration if duration>0 else 0,
                  angle_fps=sum(bool(r.get('angle_valid')) for r in rows[1:])/duration if duration>0 else 0,
                  velocity_fps=sum(r.get('omega_rad_s') is not None for r in rows[1:])/duration if duration>0 else 0,
                  detection_reasons=dict(Counter(r.get('detection_status','unknown') for r in rows)),
                  processing_ms=percentiles(r.get('processing_ms') for r in rows),
                  host_read_to_result_ms=percentiles(r.get('host_read_to_result_ms') for r in rows),
                  velocity_delay_ms=percentiles(1000*r['omega_delay_s'] for r in rows if r.get('omega_delay_s') is not None),
                  processing_deadline_misses=sum(r.get('processing_ms',0)>1000/target_fps for r in rows))
    consecutive=longest=0
    for row in rows:
        consecutive=0 if row.get('detected') else consecutive+1
        longest=max(longest,consecutive)
    result['longest_rejected_run_frames']=longest
    return result


class MeasurementMetrics:
    def __init__(self,target_fps=80):
        self.target_fps=target_fps; self.rows=deque(maxlen=800)
        self.totals=counts([]); self.last_seq=None

    def observe(self,state):
        # Small immutable record; never retain image arrays or calibration lists.
        keys=('t','seq','detected','calibrated','angle_valid','omega_rad_s','processing_ms',
              'host_read_to_result_ms','omega_delay_s','detection_status')
        row={k:state[k] for k in keys if k in state}
        delta=counts([row])
        skipped=max(0,row['seq']-self.last_seq-1) if self.last_seq is not None else 0
        delta['skipped_camera_frames']+=skipped
        if row.get('calibrated'):delta['calibrated_camera_frames']+=skipped
        self.totals.update(delta);self.last_seq=row['seq'];self.rows.append(row)

    def report(self,now):
        recent=[r for r in self.rows if r['t']>=now-2]
        result=summarize(recent,self.target_fps)
        if not recent or now-recent[-1]['t']>.15:
            result.update(processed_fps=0,angle_fps=0,velocity_fps=0)
        return dict(recent=result,totals=coverage(self.totals),window_s=2,
                    timing_note='Host read completion to result; excludes sensor exposure/readout and browser display.')
