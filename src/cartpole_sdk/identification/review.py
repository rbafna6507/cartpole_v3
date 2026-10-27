"""Plot admitted raw recordings before choosing intervals or fitting any model."""
import math
from pathlib import Path
import numpy as np
from .workflow import load_records,write_json,provenance


def review_recording(source,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    records,identity=load_records(source)
    states=[r['state'] for r in records if r.get('type')=='state']
    if len(states)<2:raise ValueError('at least two raw states required')
    t=np.asarray([s['unwrapped_us'] for s in states],float)/1e6;t-=t[0]
    if not np.isfinite(t).all() or np.any(np.diff(t)<0):raise ValueError('finite non-reversed board timestamps required')
    q=np.unwrap(np.asarray([s['theta'] for s in states],float))
    if not np.isfinite(q).all():raise ValueError('finite raw encoder angles required')
    # Plotting-only integer-turn branch; no fitted zero/scale or interval selection.
    branch=math.pi+2*math.pi*round((float(np.median(q))-math.pi)/(2*math.pi))
    baseline=identity.get('passive_baseline')
    zero=branch if baseline is None else baseline['encoder']['raw_angle_rad']
    zero+=2*math.pi*round((float(q[0])-zero)/(2*math.pi))
    protocol=identity.get('collection_protocol')
    report=dict(schema='cartpole-recording-review-v1',source_sha256=identity['sha256'],source_identity=identity,
        state_count=len(states),duration_s=float(t[-1]),max_snapshot_gap_ms=float(np.max(np.diff(t))*1000),
        raw_angle_span_rad=float(np.ptp(q)),collection_protocol=protocol,model_fitted=False,
        intervals_selected=False,physical_model_validated=False,plot_integer_branch_rad=branch,
        plot_zero_rad=zero,hanging_zero_measured=baseline is not None,measured_baseline=baseline,
        note='Data admission/visual review only. Requested conditions are nominal; raw theta is not globally calibrated.',provenance=provenance())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
    axes[0].plot(t,np.degrees(q-zero),lw=.8);axes[0].set_ylabel('Angle from measured hanging (°)' if baseline else 'Raw angle − hanging branch (°)')
    for key,label in (('x','Counted cart position (m)'),('velocity','Commanded velocity (m/s)')):
        values=np.asarray([s.get(key,np.nan) for s in states],float)
        if np.isfinite(values).any():axes[1].plot(t,values,label=label,lw=.8)
    if axes[1].lines:axes[1].legend(loc='upper right')
    axes[1].set_ylabel('Cart channels (SI)')
    for key,label in (('demand','Requested acceleration'),('acceleration','MCU governed acceleration')):
        values=np.asarray([s.get(key,np.nan) for s in states],float)
        if np.isfinite(values).any():axes[2].plot(t,values,label=label,lw=.8)
    if axes[2].lines:axes[2].legend(loc='upper right')
    axes[2].set_ylabel('Acceleration (m/s²)');axes[2].set_xlabel('Seconds from first recorded STATE')
    for ax in axes:ax.grid(alpha=.2)
    label='Unlabelled historical recording' if protocol is None else protocol['dataset_role']+' / '+protocol['run_label']
    fig.suptitle('Raw observations — '+label+'\nNo fitted model; enter free-interval times in annotation JSON before scoring')
    output.mkdir(parents=True,exist_ok=False)
    try:
        fig.savefig(output/'raw-observations.png',dpi=160);fig.savefig(output/'raw-observations.pdf')
        write_json(output/'report.json',report)
    finally:plt.close(fig)
    return report
