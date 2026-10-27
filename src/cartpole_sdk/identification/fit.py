"""Weak-form fit for theta'' = alpha sin(theta)-gamma theta'-beta a cos(theta).

Compact test functions vanish together with their first derivative at window
ends. Twice integrating by parts avoids differentiating observed angle. Delay
selection and uncertainty use training only; held-out data assess prediction.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class FitConfig:
    gravity_m_s2: float = 9.81
    length_bounds_m: tuple[float, float] = (.05, 1.5)
    gain_bounds: tuple[float, float] = (.05, 3.)
    max_damping_per_s: float = 5.
    angle_offset_rad: float = 0.
    max_angle_zero_bias_rad: float = .02
    max_gap_s: float = .05
    window_s: float = .35
    delay_candidates_s: tuple[float, ...] = tuple(i*.005 for i in range(21))
    min_windows: int = 20
    max_condition: float = 1000.
    min_angle_span_rad: float = .15
    min_acceleration_rms_m_s2: float = .05
    prediction_horizon_s: float = .5
    max_prediction_rmse_rad: float = .08
    max_prediction_p95_rad: float = .15
    max_integral_nrmse: float = .15

    def __post_init__(self):
        for name in ('gravity_m_s2','max_damping_per_s','max_gap_s','window_s','max_condition',
                     'min_angle_span_rad','min_acceleration_rms_m_s2','prediction_horizon_s',
                     'max_prediction_rmse_rad','max_prediction_p95_rad','max_integral_nrmse','max_angle_zero_bias_rad'):
            x=getattr(self,name)
            if type(x) not in (int,float) or not math.isfinite(x) or x<=0:
                raise ValueError(f'{name} must be finite and positive')
        if type(self.angle_offset_rad) not in (int,float) or not math.isfinite(self.angle_offset_rad):
            raise ValueError('angle_offset_rad must be finite')
        for bounds in (self.length_bounds_m, self.gain_bounds):
            if len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or not 0<bounds[0]<bounds[1]:
                raise ValueError('parameter bounds must be ordered positive finite pairs')
        if type(self.min_windows) is not int or self.min_windows<4:
            raise ValueError('min_windows must be at least four')
        if not self.delay_candidates_s or any(not math.isfinite(x) or x<0 for x in self.delay_candidates_s):
            raise ValueError('delay candidates must be finite nonnegative seconds')
        if tuple(sorted(set(self.delay_candidates_s)))!=self.delay_candidates_s:
            raise ValueError('delay candidates must strictly increase')

@dataclass(frozen=True)
class FitReport:
    accepted: bool
    parameters: dict
    uncertainty: dict
    training: dict
    validation: dict
    reasons: tuple[str, ...]
    provenance: dict
    assumptions: tuple[str, ...]
    unidentifiable: tuple[str, ...] = ('cart_mass_kg','pole_mass_kg','force_gain_N_per_command','hinge_damping_N_m_s')
    recommended_model: dict | None = None

    def to_dict(self):
        return asdict(self)


def _hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _segments(rows, source, acceleration_kind, cfg):
    segments=[]; active=[]; previous=None; identity=None
    if len(rows)<3: raise ValueError('need at least three timestamped rows')
    for row in rows:
        for key in ('t_s','theta_rad','cart_acceleration_m_s2'):
            value=row.get(key)
            if type(value) not in (int,float) or not math.isfinite(value):
                # Invalid records can explicitly omit unavailable channels.
                if row.get('valid',True) is False and key!='t_s' and value is None: continue
                raise ValueError(f'{key} must be finite')
        t=row['t_s']
        if previous is not None and t<=previous: raise ValueError('timestamps must strictly increase')
        for key,expected in (('source',source),('acceleration_kind',acceleration_kind)):
            if key in row and row[key]!=expected: raise ValueError(f'{key} does not match declared provenance')
        if 'valid' in row and type(row['valid']) is not bool: raise ValueError('valid must be bool')
        if row.get('omega_rad_s') is not None and (type(row['omega_rad_s']) not in (float,int) or not math.isfinite(row['omega_rad_s'])):
            raise ValueError('omega_rad_s must be finite')
        if row.get('omega_rad_s') is not None and row.get('omega_effective_t_s') is not None:
            effective=row['omega_effective_t_s']
            if type(effective) not in (int,float) or not math.isfinite(effective) or abs(effective-t)>1e-8:
                raise ValueError('omega derivative effective time differs from row time; align explicitly or omit omega')
        ident=(row.get('run_id'),row.get('calibration_id'),row.get('clock_id'),row.get('frame_id'))
        split=(previous is not None and t-previous>cfg.max_gap_s) or (identity is not None and ident!=identity) or not row.get('valid',True)
        if split and active: segments.append(active); active=[]
        if row.get('valid',True): active.append(row)
        previous=t;identity=ident
    if active: segments.append(active)
    result=[]
    for segment in segments:
        if len(segment)<8: continue
        t=np.array([r['t_s'] for r in segment], float)
        q=np.unwrap([r['theta_rad'] for r in segment])-cfg.angle_offset_rad
        a=np.array([r['cart_acceleration_m_s2'] for r in segment], float)
        # Unwrapping is valid only when less than pi physical motion occurs
        # between samples. This remains an explicitly declared assumption.
        result.append((t,q,a,segment))
    return result


def _integral(values,t):
    # Explicit trapezoid rule supports the documented NumPy >=1.26 extra.
    return np.sum((values[1:]+values[:-1])*np.diff(t)/2)


def _windows(segments,delay,cfg,*,include_offset=False):
    xs=[];ys=[]
    for t,q,a,_ in segments:
        start=t[0]+max(cfg.delay_candidates_s)
        while start+cfg.window_s<=t[-1]+1e-10:
            indices=np.flatnonzero((t>=start-1e-10)&(t<=start+cfg.window_s+1e-10))
            if len(indices)>=8:
                tw=t[indices]; qw=q[indices]
                duration=tw[-1]-tw[0]; s=(tw-tw[0])/duration
                phi=s*s*(1-s)**2
                dp=(2*s-6*s*s+4*s**3)/duration
                ddp=(2-12*s+12*s*s)/(duration**2)
                aw=np.interp(tw-delay,t,a)
                # Constant/linear terms integrate exactly to zero against phi'';
                # remove them analytically before quadrature, avoiding offset
                # bias on irregular grids. Integrate the linear damping term
                # analytically too: integral(t*phi') = -integral(phi).
                slope=(qw[-1]-qw[0])/duration
                detrended=qw-qw[0]-slope*(tw-tw[0])
                damping_feature=_integral(detrended*dp,tw)-slope*_integral(phi,tw)
                features=[_integral(np.sin(qw)*phi,tw),damping_feature,-_integral(aw*np.cos(qw)*phi,tw)]
                if include_offset:
                    features.extend([_integral(np.cos(qw)*phi,tw),-_integral(aw*np.sin(qw)*phi,tw)])
                xs.append(features)
                ys.append(_integral(detrended*ddp,tw))
            start+=cfg.window_s
    return np.asarray(xs).reshape((-1,5 if include_offset else 3)),np.asarray(ys)


def _solve(x,y):
    scale=np.linalg.norm(x,axis=0)
    if np.any(scale<1e-10): return None, None, 0, None
    normalized=x/scale
    coef,_,rank,singular=np.linalg.lstsq(normalized,y,rcond=None)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
    return coef/scale,scale,int(rank),condition


def _predict(segments,coef,delay,cfg):
    alpha,gamma,beta=coef; errors=[]; window_count=0
    def rhs(t,y,ts,acc):
        q,w=y
        return np.array([w,alpha*math.sin(q)-gamma*w-beta*float(np.interp(t-delay,ts,acc))*math.cos(q)])
    for ts,q,acc,rows in segments:
        i=int(np.searchsorted(ts,ts[0]+max(cfg.delay_candidates_s)))
        while i+8<len(ts):
            end=int(np.searchsorted(ts,ts[i]+cfg.prediction_horizon_s,side='right'))
            if end>len(ts) or ts[end-1]-ts[i]<cfg.prediction_horizon_s*.9: break
            if rows[i].get('omega_rad_s') is not None:
                omega=float(rows[i]['omega_rad_s'])
            else:
                # Estimate initialization using only samples before forecast.
                if i<6: i=6; continue
                omega=float(np.polyfit(ts[i-6:i+1]-ts[i],q[i-6:i+1],2)[-2])
            state=np.array([q[i],omega])
            for j in range(i+1,end):
                h=ts[j]-ts[j-1]; substeps=max(1,math.ceil(h/.005)); h/=substeps
                t=ts[j-1]
                for _ in range(substeps):
                    k1=rhs(t,state,ts,acc);k2=rhs(t+h/2,state+h*k1/2,ts,acc)
                    k3=rhs(t+h/2,state+h*k2/2,ts,acc);k4=rhs(t+h,state+h*k3,ts,acc)
                    state+=h*(k1+2*k2+2*k3+k4)/6;t+=h
                errors.append(math.atan2(math.sin(state[0]-q[j]),math.cos(state[0]-q[j])))
            window_count+=1;i=end
    if not errors: return dict(prediction_windows=0,prediction_rmse_rad=None,prediction_p95_rad=None)
    values=np.asarray(errors)
    return dict(prediction_windows=window_count,prediction_samples=len(errors),prediction_rmse_rad=float(np.sqrt(np.mean(values**2))),
                prediction_p95_rad=float(np.quantile(np.abs(values),.95)), horizon_s=cfg.prediction_horizon_s)


def identify_pendulum(training: Sequence[Mapping], validation: Sequence[Mapping], *, source='magnetic',
                      acceleration_kind='commanded', config: FitConfig | None=None, provenance=None) -> FitReport:
    """Fit training only; evaluate independent contiguous validation run segments.

    Input theta is SI, upright zero and positive in the model's cart +x rotation.
    No timestamp interpolation, unwrap, regression or rollout crosses a gap,
    invalid record, run, clock, frame or calibration change.
    """
    cfg=config or FitConfig()
    if source not in ('magnetic','vision','simulation','fused'): raise ValueError('unknown angle source')
    if acceleration_kind not in ('commanded','measured'): raise ValueError('acceleration_kind must be commanded or measured')
    training=[dict(r) for r in training];validation=[dict(r) for r in validation]
    train=_segments(training,source,acceleration_kind,cfg);valid=_segments(validation,source,acceleration_kind,cfg)
    train_ids={r['run_id'] for r in training if r.get('run_id') is not None}
    valid_ids={r['run_id'] for r in validation if r.get('run_id') is not None}
    if train_ids & valid_ids or _hash(training)==_hash(validation): raise ValueError('training and validation must be independent runs without overlap')
    # Without run identities, detect exact reuse of recorded timestamp/value tuples.
    if not train_ids or not valid_ids:
        tuples=lambda rows:{(r['t_s'],r['theta_rad'],r['cart_acceleration_m_s2']) for r in rows if r.get('valid',True)}
        if tuples(training)&tuples(validation): raise ValueError('training/validation records overlap; provide independent run_id metadata')
    info=dict(source=source,acceleration_kind=acceleration_kind,training_sha256=_hash(training),validation_sha256=_hash(validation),
              training_run_ids=sorted(train_ids),validation_run_ids=sorted(valid_ids),config=asdict(cfg),supplied=provenance or {})
    assumptions=('upright theta=0; positive theta follows cart +x rotation',
                 'rigid planar pendulum; constant effective length; linear viscous angular damping',
                 'sampling captures less than pi angular motion per valid interval',
                 'acceleration and angle share a calibrated monotonic clock',
                 'commanded input gain/delay include actuator, estimator and clock effects' if acceleration_kind=='commanded' else 'measured cart acceleration is independently calibrated',
                 'uncertainty is conditional regression uncertainty; excludes calibration, correlated sensor noise and model bias')
    measurement_keys=('t_s','theta_rad','cart_acceleration_m_s2','omega_rad_s','valid')
    measurement_hash=lambda rows:_hash([{key:r.get(key) for key in measurement_keys} for r in rows])
    reasons=[]
    if measurement_hash(training)==measurement_hash(validation):
        reasons.append('training and held-out measurements are identical despite different run identities')
    profile=[];best=None
    for delay in cfg.delay_candidates_s:
        x,y=_windows(train,delay,cfg)
        if len(y)<cfg.min_windows: continue
        coef,scale,rank,condition=_solve(x,y)
        if coef is None or rank<3: continue
        loss=float(np.mean((y-x@coef)**2))
        profile.append(dict(delay_s=delay,mean_squared_integral_residual=loss))
        if best is None or loss<best[0]:best=(loss,delay,coef,x,y,rank,condition)
    stats=dict(segment_count=len(train),sample_count=sum(len(s[0]) for s in train),delay_profile=profile)
    if best is None:
        return FitReport(False,{}, {},stats,{},('insufficient windows, rank or excitation',),info,assumptions)
    loss,delay,coef,x,y,rank,condition=best;alpha,gamma,beta=coef
    # A calibration offset produces coupled cos(theta) gravity and sin(theta)
    # actuation terms. Fit both solely as a nuisance diagnostic, never silently
    # use them to recommend a plant parameter or alter live calibration.
    dx,dy=_windows(train,delay,cfg,include_offset=True)
    dc,_,dr,dk=_solve(dx,dy)
    diagnostic={'rank':dr,'condition':dk,'identifiable':dr==5 and dk is not None and dk<=cfg.max_condition}
    if diagnostic['identifiable']:
        gravity_offset=math.atan2(-float(dc[3]),float(dc[0]))
        input_offset=math.atan2(float(dc[4]),float(dc[2]))
        diagnostic.update(gravity_offset_rad=gravity_offset,input_offset_rad=input_offset)
        if abs(gravity_offset)>cfg.max_angle_zero_bias_rad or abs(input_offset)>cfg.max_angle_zero_bias_rad:
            reasons.append('angle zero bias diagnostic exceeds threshold; independently establish calibration/offset')
    else:
        reasons.append('angle zero bias cannot be identified with supplied excitation/calibration')
    stats['angle_zero_diagnostic']=diagnostic
    nrmse=float(np.sqrt(loss)/max(float(np.std(y)),1e-12))
    span=max((float(np.ptp(s[1])) for s in train),default=0.)
    ar=np.concatenate([s[2] for s in train]); a_rms=float(np.std(ar))
    stats.update(window_count=len(y),rank=rank,condition=condition,integral_nrmse=nrmse,angle_span_rad=span,acceleration_std_m_s2=a_rms)
    params={}
    if alpha>0:
        length=cfg.gravity_m_s2/float(alpha)
        params=dict(effective_length_m=length,input_gain=float(beta)*length,angular_damping_per_s=float(gamma),
                    damping_ratio=float(gamma)/(2*math.sqrt(float(alpha))),delay_s=delay)
        if not cfg.length_bounds_m[0]<=length<=cfg.length_bounds_m[1]: reasons.append('effective length outside prior bounds')
        if not cfg.gain_bounds[0]<=params['input_gain']<=cfg.gain_bounds[1]: reasons.append('input gain outside prior bounds')
    else:reasons.append('gravity coefficient is nonpositive')
    if not 0<=gamma<=cfg.max_damping_per_s:reasons.append('angular damping outside nonnegative bounds')
    if condition is None or condition>cfg.max_condition:reasons.append('rank/conditioning inadequate')
    if span<cfg.min_angle_span_rad or a_rms<cfg.min_acceleration_rms_m_s2: reasons.append('insufficient angle/input excitation')
    if nrmse>cfg.max_integral_nrmse:reasons.append('training integral residual exceeds threshold')
    if len(profile)>1 and delay==cfg.delay_candidates_s[-1]:reasons.append('delay hits maximum search boundary')
    residual=y-x@coef
    covariance=np.linalg.pinv(x.T@x)*float(residual@residual)/max(1,len(y)-3)
    se=np.sqrt(np.maximum(np.diag(covariance),0.))
    near=[p['delay_s'] for p in profile if p['mean_squared_integral_residual']<=loss*1.05+1e-12]
    uncertainty=dict(coefficient_standard_errors=dict(alpha_per_s2=float(se[0]),gamma_per_s=float(se[1]),beta_per_m=float(se[2])),
                     delay_profile_near_optimal_s=near,method='OLS conditional covariance on nonoverlapping weak-form windows',
                     effective_length_standard_error_m=float(cfg.gravity_m_s2*se[0]/alpha**2) if alpha>0 else None)
    check=_predict(valid,coef,delay,cfg)
    vx,vy=_windows(valid,delay,cfg)
    check['segment_count']=len(valid)
    check['integral_nrmse']=float(np.sqrt(np.mean((vy-vx@coef)**2))/max(float(np.std(vy)),1e-12)) if len(vy) else None
    if (check['prediction_windows']<3 or check['prediction_rmse_rad'] is None or check['prediction_rmse_rad']>cfg.max_prediction_rmse_rad
        or check['prediction_p95_rad']>cfg.max_prediction_p95_rad or check['integral_nrmse'] is None or check['integral_nrmse']>cfg.max_integral_nrmse):
        reasons.append('held-out prediction or integral residual exceeds threshold/coverage')
    accepted=not reasons
    recommendation={'effective_length_m':params['effective_length_m']} if accepted else None
    return FitReport(accepted,params,uncertainty,stats,check,tuple(reasons),info,assumptions,recommended_model=recommendation)
