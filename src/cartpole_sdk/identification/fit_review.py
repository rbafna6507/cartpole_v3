"""Review an existing passive fit against its exact annotated validation source.

No parameter fit, model export, new acceptance decision, device access or
interval selection occurs here. Predictions use the pinned passive ODE method,
not MuJoCo. Driven reports need their own input-aware review and are rejected.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np
from . import passive
from .workflow import (MAX_REPORT_BYTES, digest, passive_episodes, read_json,
                       write_json)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(errors):
    values=np.asarray(errors,float)
    return dict(prediction_samples=len(values),
        prediction_rmse_rad=float(np.sqrt(np.mean(values**2))) if len(values) else None,
        prediction_p95_rad=float(np.quantile(np.abs(values),.95)) if len(values) else None)


def _errors(predicted,recorded):
    d=np.asarray(predicted)-np.asarray(recorded)
    return np.arctan2(np.sin(d),np.cos(d))


def forecast_segment(t,q,coefficients,config):
    """Causal frozen rollouts on one contiguous segment, never across gaps."""
    t,q=np.asarray(t,float),np.asarray(q,float)
    if (t.ndim!=1 or q.shape!=t.shape or len(t)<8 or not np.isfinite(t).all()
        or not np.isfinite(q).all() or np.any(np.diff(t)<=0)
        or np.max(np.diff(t))>config['max_gap_s']):
        raise ValueError('finite ordered contiguous segment required')
    start=max(10,int(np.searchsorted(t,t[0]+.1)))
    def rollout(i,end):
        omega=float(np.polyfit(t[i-10:i+1]-t[i],q[i-10:i+1],2)[-2])
        initial=dict(t_s=float(t[i]),theta_rad=float(q[i]),omega_rad_s=omega,
            support_start_s=float(t[i-10]),support_end_s=float(t[i]),support_samples=11)
        angle=float(q[i]);predicted=[]
        for j in range(i+1,end):
            delta=t[j]-t[j-1];steps=max(1,math.ceil(delta/.001));h=delta/steps
            for _ in range(steps):angle,omega=passive.advance(angle,omega,h,coefficients)
            if not math.isfinite(angle) or not math.isfinite(omega):
                raise ValueError('saved coefficients produce nonfinite ODE trajectory')
            predicted.append(angle)
        return dict(initial=initial,t_s=t[i+1:end].tolist(),
            recorded_theta_rad=q[i+1:end].tolist(),predicted_theta_rad=predicted,
            **_metrics(_errors(predicted,q[i+1:end])))
    short=[];i=start
    while i+8<len(t):
        end=int(np.searchsorted(t,t[i]+config['prediction_horizon_s'],side='right'))
        if t[end-1]-t[i]<.9*config['prediction_horizon_s']:break
        short.append(rollout(i,end));i=end
    continuous=rollout(start,len(t)) if start+1<len(t) else dict(initial=None,
        t_s=[],recorded_theta_rad=[],predicted_theta_rad=[],**_metrics([]))
    return dict(t_s=t.tolist(),recorded_theta_rad=q.tolist(),short_windows=short,continuous=continuous)


def _bound_inputs(fit_path,validation,annotations):
    fit_path=Path(fit_path)
    if fit_path.stat().st_size>MAX_REPORT_BYTES:raise ValueError('fit report byte limit exceeded')
    fit=read_json(fit_path)
    if not isinstance(fit,dict) or fit.get('schema')!='cartpole-id-fit-v1' or fit.get('kind')!='passive':
        raise ValueError('review-fit currently requires a packaged passive fit; driven ODE review is unsupported')
    if fit.get('model') not in ('viscous','viscous_coulomb') or type(fit.get('accepted')) is not bool:
        raise ValueError('invalid saved passive fit model or admission')
    provenance=fit['provenance'];cfg=provenance['config']
    if cfg!=dict(passive.DEFAULT_GATES,window_s=.35):
        raise ValueError('saved fit gates differ from the supported fixed passive method')
    if provenance.get('method_sha256')!=_sha(passive.__file__):
        raise ValueError('saved passive method hash differs; exact frozen solver is unavailable')
    c=fit.get('parameters',{}).get('linear_coefficients')
    count=3 if fit['model']=='viscous' else 4
    if (not isinstance(c,list) or len(c)!=count or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in c)):
        raise ValueError('saved fit has no finite identifiable coefficient vector to review')
    episodes,identity=passive_episodes(validation,annotations)
    saved=provenance['validation']
    for key in ('sha256','annotations_sha256','session','reference','selection',
                'operator_cart_restraint','selection_method'):
        if key not in saved or digest(saved[key])!=digest(identity[key]):
            raise ValueError('validation provenance mismatch: '+key)
    for key in ('collection_protocol_sha256',):
        if saved.get(key)!=identity.get(key):raise ValueError('validation provenance mismatch: '+key)
    for key in ('sha256','status','completed'):
        if saved.get('collection_report',{}).get(key)!=identity.get('collection_report',{}).get(key):
            raise ValueError('validation collection report identity mismatch: '+key)
    if (provenance.get('validation_sha256')!=digest(episodes) or
            provenance.get('validation_run_ids')!=sorted({e['run_id'] for e in episodes})):
        raise ValueError('canonical validation episodes differ from saved fit')
    training=provenance['training']
    if (training['sha256']==identity['sha256'] or
        training['session']['session_id']==identity['session']['session_id'] or
        training['session']['clock_id']==identity['session']['clock_id']):
        raise ValueError('saved training/validation identities are not independent')
    if len(episodes)>24:raise ValueError('review supports at most24 annotated validation episodes')
    return fit,episodes,identity


def _assert_saved_metrics(actual,saved):
    for key in ('prediction_windows','prediction_samples','prediction_rmse_rad',
                'prediction_p95_rad','integral_nrmse','weak_window_count','angle_span_rad'):
        a,b=actual.get(key),saved.get(key)
        if a is None or b is None:
            if a!=b:raise ValueError('saved validation metric mismatch: '+key)
        elif not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('saved validation metric mismatch: '+key)


def _draw(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    episodes=report['episodes']
    fig,axes=plt.subplots(len(episodes),2,figsize=(13,max(3,3*len(episodes))),squeeze=False,layout='constrained')
    for row,episode in enumerate(episodes):
        all_q=np.concatenate([s['recorded_theta_rad'] for s in episode['segments']])
        branch=math.pi+2*math.pi*round((float(np.median(all_q))-math.pi)/(2*math.pi))
        for index,segment in enumerate(episode['segments']):
            t=np.asarray(segment['t_s']);q=np.asarray(segment['recorded_theta_rad'])
            for ax in axes[row]:
                ax.plot(t,np.degrees(q-branch),color='#1766a0',lw=1.15,label='Recorded raw angle' if index==0 else None)
            for window in segment['short_windows']:
                axes[row,0].plot(window['t_s'],np.degrees(np.asarray(window['predicted_theta_rad'])-branch),color='#d56720',lw=.9)
            continuous=segment['continuous']
            axes[row,1].plot(continuous['t_s'],np.degrees(np.asarray(continuous['predicted_theta_rad'])-branch),color='#d56720',lw=1.1,label='Frozen ODE prediction' if index==0 else None)
        short=episode['short_forecasts'];continuous=episode['continuous']
        fmt=lambda value:'unavailable' if value is None else f'{math.degrees(value):.2f}°'
        axes[row,0].set_title(f"{episode['episode_id']}: {report['fixed_config']['prediction_horizon_s']:g}s forecasts; RMS {fmt(short['prediction_rmse_rad'])}")
        axes[row,1].set_title(f"Continuous within each segment; RMS {fmt(continuous['prediction_rmse_rad'])}")
        for ax in axes[row]:
            ax.grid(alpha=.2);ax.set_ylabel('Raw angle − integer hanging branch (°)');ax.set_xlabel('Seconds from first recorded STATE')
    axes[0,0].plot([],[],color='#d56720',label='Frozen ODE forecasts')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    admission='accepted within saved scope' if report['saved_admission']['accepted'] else 'rejected'
    label=(report['data_label']+'\n') if report.get('data_label') else ''
    fig.suptitle(label+'Recorded versus saved passive ODE — not MuJoCo\nSaved admission: '+admission+'; no refit or model promotion',fontsize=13)
    try:
        fig.savefig(output/'recorded-vs-frozen-ode.png',dpi=150)
        fig.savefig(output/'recorded-vs-frozen-ode.pdf')
    finally:plt.close(fig)


def review_fit(fit_path,validation,annotations,output,*,data_label=None):
    """Create immutable plots/metrics for the fit's exact validation selection."""
    if data_label is not None and (not isinstance(data_label,str) or not data_label.strip() or len(data_label)>128):
        raise ValueError('data label must be a nonempty string of at most128 characters')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    fit,episodes,identity=_bound_inputs(fit_path,validation,annotations)
    c=fit['parameters']['linear_coefficients'];cfg=fit['provenance']['config']
    parts=passive.segments(episodes,cfg);x,y=passive.matrix(parts,cfg,fit['model'])
    short=passive.predict(parts,c,cfg)
    short.update(integral_nrmse=passive.nrmse(x,y,c),weak_window_count=len(y),
        angle_span_rad=max((float(np.ptp(q)) for t,q,identity_ in parts),default=0.))
    _assert_saved_metrics(short,fit['validation'])
    results=[]
    for episode in episodes:
        selected=passive.segments([episode],cfg)
        segments=[forecast_segment(t,q,c,cfg) for t,q,_ in selected]
        errors=[]
        for segment in segments:
            trace=segment['continuous'];errors.extend(_errors(trace['predicted_theta_rad'],trace['recorded_theta_rad']))
        short_ep=passive.predict(selected,c,cfg)
        results.append(dict(episode_id=episode['episode_id'],run_id=episode['run_id'],
            segment_count=len(segments),continuous_reinitializations=len(segments),
            short_forecasts=short_ep,continuous=_metrics(errors),segments=segments))
    admission_keys=('accepted','statistical_checks_pass','training_checks_pass','reasons',
        'acceptance_scope','physical_cart_stationarity_certified','recommended_for_hardware_rl')
    report=dict(schema='cartpole-fit-review-v1',solver='frozen passive ODE; not MuJoCo',data_label=data_label,
        parameters_refitted=False,model_exported=False,recommended_for_hardware_rl=False,
        independent_acceptance_test=False,saved_admission={k:deepcopy(fit.get(k)) for k in admission_keys},
        parameters=deepcopy(fit['parameters']),fixed_config=deepcopy(cfg),
        saved_validation=deepcopy(fit['validation']),saved_per_episode_validation=deepcopy(fit.get('per_episode_validation',[])),
        short_forecasts=short,episodes=results,validation_identity=identity,
        provenance=dict(fit_path=str(Path(fit_path).resolve()),fit_sha256=_sha(fit_path),
            passive_method_sha256=_sha(passive.__file__),review_method_sha256=_sha(__file__),
            canonical_validation_sha256=digest(episodes)),
        limitations=['ODE replay of saved coefficients; no MuJoCo simulation is performed.',
            'Each forecast uses only the preceding11 raw angle samples and its initial angle.',
            'Continuous predictions initialize once per contiguous segment; gaps never bridge.',
            'All original admission fields and fixed gates are preserved; plots cannot promote a rejected fit.',
            'Existing validation data are replayed, not a new independent acceptance test.',
            'The fitted local phase and physically fixed cart remain assumptions, not global calibration.'])
    # Render before publishing the requested directory; failed plots cannot
    # masquerade as a completed immutable review. Existing output is never used.
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fit-review-',dir=output.parent) as staging:
        staging=Path(staging);_draw(report,staging);write_json(staging/'report.json',report)
        output.mkdir(exist_ok=False)
        for name in ('recorded-vs-frozen-ode.png','recorded-vs-frozen-ode.pdf','report.json'):
            (staging/name).rename(output/name)
    return report
