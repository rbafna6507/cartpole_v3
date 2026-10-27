"""Offline passive weak-form identification, with frozen independent validation.

No hardware, network, calibration or simulator mutation. Callers must establish
free-release boundaries and fixed-cart/angle validity independently.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

DEFAULT_GATES={'min_windows': 20, 'min_angle_span_rad': 0.15, 'max_condition': 1000, 'length_bounds_m': [0.05, 1.5], 'angular_damping_bounds_per_s': [0, 5], 'coulomb_bounds_rad_per_s2': [0, 20], 'max_fitted_offset_abs_rad': 0.35, 'max_gap_s': 0.05, 'max_integral_nrmse': 0.15, 'prediction_horizon_s': 0.5, 'max_prediction_rmse_rad': 0.08, 'max_prediction_p95_rad': 0.15, 'min_prediction_windows': 3}
FROZEN_METHOD_ORIGIN="system-id-20261016 passive weak form; packaged without evidence imports"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def segments(episodes,cfg):
    result=[];identities=set()
    for episode in episodes:
        identity=(episode['run_id'],episode['episode_id'])
        if identity in identities:raise ValueError('duplicate episode identity')
        identities.add(identity)
        t=np.asarray(episode['t_s'],float);q=np.asarray(episode['theta_rad'],float)
        if t.ndim!=1 or q.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(q)):
            raise ValueError('need finite matching angle/time vectors with at least eight samples')
        if np.any(np.diff(t)<=0):raise ValueError('timestamps must strictly increase within each episode')
        for indices in np.split(np.arange(len(t)),np.flatnonzero(np.diff(t)>cfg['max_gap_s'])+1):
            if len(indices)>=8:result.append((t[indices],np.unwrap(q[indices]),identity))
    return result


def integral(y,t):
    return float(np.sum((y[1:]+y[:-1])*np.diff(t)/2))


def matrix(parts,cfg,model):
    features=[];targets=[]
    for t,q,identity in parts:
        velocity=np.full(len(t),np.nan)
        if model=='viscous_coulomb':
            for i in range(len(t)):
                lo=np.searchsorted(t,t[i]-.024-1e-10);hi=np.searchsorted(t,t[i]+.024+1e-10,side='right')
                if hi-lo>=2:
                    velocity[i]=np.polyfit(t[lo:hi]-t[i],q[lo:hi],min(2,hi-lo-1))[-2]
        start=t[0]+.1
        while start+cfg['window_s']<=t[-1]+1e-10:
            ix=np.flatnonzero((t>=start-1e-10)&(t<=start+cfg['window_s']+1e-10))
            if len(ix)>=8 and (model!='viscous_coulomb' or np.all(np.isfinite(velocity[ix]))):
                tw=t[ix];qw=q[ix];duration=tw[-1]-tw[0];s=(tw-tw[0])/duration
                phi=s*s*(1-s)**2;dp=(2*s-6*s*s+4*s**3)/duration;ddp=(2-12*s+12*s*s)/duration**2
                slope=(qw[-1]-qw[0])/duration;detrended=qw-qw[0]-slope*(tw-tw[0])
                row=[integral(np.sin(qw)*phi,tw),integral(detrended*dp,tw)-slope*integral(phi,tw),integral(np.cos(qw)*phi,tw)]
                if model=='viscous_coulomb':
                    sign=np.where(np.abs(velocity[ix])>.02,np.sign(velocity[ix]),0.)
                    row.append(-integral(sign*phi,tw))
                features.append(row);targets.append(integral(detrended*ddp,tw))
            start+=cfg['window_s']
    return np.asarray(features,float).reshape((-1,4 if model=='viscous_coulomb' else 3)),np.asarray(targets,float)


def solve(x,y):
    scales=np.linalg.norm(x,axis=0)
    if not len(y) or np.any(scales<1e-10):return None,0,None,None
    normalized=x/scales;c,unused,rank,singular=np.linalg.lstsq(normalized,y,rcond=None)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None;c=c/scales
    residual=y-x@c;cov=np.linalg.pinv(x.T@x)*float(residual@residual)/max(1,len(y)-len(c))
    return c,int(rank),condition,cov


def nrmse(x,y,c):
    return float(np.sqrt(np.mean((y-x@c)**2))/max(float(np.std(y)),1e-12)) if len(y) else None


def advance(q,w,h,c):
    a,gamma,b=c[:3];friction=c[3] if len(c)>3 else 0.
    def gravity(q):return a*math.sin(q)+b*math.cos(q)
    if abs(w)<1e-8 and abs(gravity(q))<=friction:return q,0.
    def rhs(q,w):
        force=gravity(q)
        sign=1. if w>0 else -1. if w<0 else (1. if force>0 else -1.)
        return w,force-gamma*w-friction*sign
    q1,w1=rhs(q,w);q2,w2=rhs(q+h*q1/2,w+h*w1/2)
    q3,w3=rhs(q+h*q2/2,w+h*w2/2);q4,w4=rhs(q+h*q3,w+h*w3)
    nq=q+h*(q1+2*q2+2*q3+q4)/6;nw=w+h*(w1+2*w2+2*w3+w4)/6
    if w*nw<0 and abs(gravity(nq))<=friction:nw=0.
    return nq,nw


def predict(parts,c,cfg):
    errors=[];count=0
    for t,q,identity in parts:
        i=max(10,int(np.searchsorted(t,t[0]+.1)))
        while i+8<len(t):
            end=int(np.searchsorted(t,t[i]+cfg['prediction_horizon_s'],side='right'))
            if t[end-1]-t[i]<.9*cfg['prediction_horizon_s']:break
            w=float(np.polyfit(t[i-10:i+1]-t[i],q[i-10:i+1],2)[-2]);pred=float(q[i])
            for j in range(i+1,end):
                delta=t[j]-t[j-1];steps=max(1,math.ceil(delta/.001));h=delta/steps
                for unused in range(steps):pred,w=advance(pred,w,h,c)
                errors.append(math.atan2(math.sin(pred-q[j]),math.cos(pred-q[j])))
            count+=1;i=end
    if not errors:return dict(prediction_windows=0,prediction_samples=0,prediction_rmse_rad=None,prediction_p95_rad=None)
    e=np.asarray(errors)
    return dict(prediction_windows=count,prediction_samples=len(e),prediction_rmse_rad=float(np.sqrt(np.mean(e*e))),prediction_p95_rad=float(np.quantile(np.abs(e),.95)))


def camera_stationarity(rows):
    """Raw image-plane pivot stability only; calibrated/fused fields ignored."""
    points=[];times=[];unique=0;duplicates=0;reversed_time=0;previous=None
    for row in rows:
        v=row.get('vision',{});t=v.get('t')
        if type(t) not in (float,int) or not math.isfinite(t):continue
        if previous is not None:
            if t==previous:duplicates+=1;continue
            if t<previous:reversed_time+=1
        previous=t;unique+=1
        p=v.get('pivot_px');confidence=v.get('confidence',0.)
        if (type(confidence) not in (float,int) or not math.isfinite(confidence) or confidence<.5 or
            not isinstance(p,(list,tuple)) or len(p)!=2 or
            any(type(z) not in (float,int) or not math.isfinite(z) for z in p)):continue
        points.append(p);times.append(t)
    result=dict(unique_camera_frames=unique,valid_pivot_frames=len(points),duplicate_frames=duplicates,
                reversed_timestamps=reversed_time,valid_geometry_fraction=len(points)/max(1,unique),
                metric_stationarity_established=False,independent_angle_calibration=False,
                pixel_stationarity_check_pass=False,source='raw pivot_px with confidence >=0.5',
                caveat='Image-plane stability cannot establish metric cart motion or hidden out-of-plane motion; old calibration is ignored.')
    if not points:return result
    p=np.asarray(points,float);center=np.median(p,axis=0);distance=np.linalg.norm(p-center,axis=1)
    result.update(median_pivot_px=center.tolist(),radial_p95_px=float(np.quantile(distance,.95)),
                  radial_max_px=float(np.max(distance)),max_camera_gap_s=float(np.max(np.diff(times))) if len(times)>1 else None)
    result['pixel_stationarity_check_pass']=(len(points)>=80 and result['valid_geometry_fraction']>=.9 and
        reversed_time==0 and result['radial_p95_px']<=2 and result['radial_max_px']<=4 and result['max_camera_gap_s']<=.1)
    return result


def fit_passive(training,validation=None,*,model='viscous_coulomb'):
    if model not in ('viscous','viscous_coulomb'):raise ValueError('unknown declared passive model')
    if not training or validation==[]:raise ValueError('nonempty training and, if provided, independent validation episodes required')
    if validation is not None and {x['run_id'] for x in training}&{x['run_id'] for x in validation}:raise ValueError('training and heldout recordings must be independent')
    measurements=lambda rows:[{'t_s':x['t_s'],'theta_rad':x['theta_rad']} for x in rows]
    if validation is not None and digest(measurements(training))==digest(measurements(validation)):raise ValueError('identical measurements cannot be heldout under renamed identities')
    cfg=dict(DEFAULT_GATES);cfg['window_s']=.35
    train=segments(training,cfg);valid=segments(validation,cfg) if validation is not None else []
    x,y=matrix(train,cfg,model);vx,vy=matrix(valid,cfg,model)
    c,rank,condition,cov=solve(x,y)
    span=max((float(np.ptp(q)) for t,q,identity in train),default=0.)
    stats=dict(window_count=len(y),segment_count=len(train),sample_count=sum(len(t) for t,q,identity in train),rank=rank,condition=condition,angle_span_rad=span)
    result=dict(model=model,training_checks_pass=False,statistical_checks_pass=False,recommended_model=None,parameters={},training=stats,validation={'status':'not provided'} if validation is None else {},reasons=[],
                provenance=dict(training_sha256=digest(training),validation_sha256=digest(validation) if validation is not None else None,method_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),config=cfg,training_run_ids=sorted({x['run_id'] for x in training}),validation_run_ids=sorted({x['run_id'] for x in validation}) if validation is not None else []),
                assumptions=['Cart is physically fixed and all selected episodes are free releases.','Raw magnetic angle has constant bias, unit scale and no nonlinear mapping distortion.','Friction is viscous plus symmetric Coulomb/stiction; no input gain is estimated.','Fitted bias is training-only model inference, not independent global angle calibration.','Conditional regression uncertainty omits calibration, correlated sensor noise, release annotation and model bias.'],
                unidentifiable=['input_gain','physical_actuator_delay','cart_mass','pole_mass','hinge_torque_damping','motor_force_or_friction'])
    reasons=result['reasons']
    if len(y)<cfg['min_windows']:reasons.append('insufficient training weak windows')
    if span<cfg['min_angle_span_rad']:reasons.append('insufficient training angle excitation')
    if c is None or rank<x.shape[1] or condition is None or condition>cfg['max_condition']:
        reasons.append('training rank/conditioning inadequate');return result
    a,gamma,b=c[:3];gravity=float(math.hypot(a,b))
    if gravity<=1e-12:
        reasons.append('gravity magnitude is zero; no physical pendulum length');return result
    length=9.81/gravity;bias=math.atan2(-b,a);friction=float(c[3]) if len(c)>3 else 0.
    result['parameters']=dict(effective_length_m=length,gravity_coefficient_per_s2=gravity,angular_damping_per_s=float(gamma),coulomb_rad_per_s2=friction,angle_offset_rad=bias,linear_coefficients=c.tolist())
    result['uncertainty']=dict(coefficient_standard_errors=np.sqrt(np.maximum(np.diag(cov),0)).tolist(),method='conditional OLS weak-window covariance')
    for value,key,label in [(length,'length_bounds_m','effective length'),(gamma,'angular_damping_bounds_per_s','viscous damping'),(friction,'coulomb_bounds_rad_per_s2','Coulomb friction')]:
        if not cfg[key][0]<=value<=cfg[key][1]:reasons.append(label+' outside frozen physical bounds')
    if abs(bias)>cfg['max_fitted_offset_abs_rad']:reasons.append('training angle offset outside frozen prior')
    stats['integral_nrmse']=nrmse(x,y,c)
    if stats['integral_nrmse']>cfg['max_integral_nrmse']:reasons.append('training integral residual exceeds threshold')
    result['training_checks_pass']=not reasons
    if validation is None:
        reasons.append('independent heldout validation not provided');return result
    prediction=predict(valid,c,cfg);prediction['integral_nrmse']=nrmse(vx,vy,c);prediction['weak_window_count']=len(vy)
    prediction['angle_span_rad']=max((float(np.ptp(q)) for t,q,identity in valid),default=0.)
    result['validation']=prediction
    if prediction['angle_span_rad']<cfg['min_angle_span_rad']:reasons.append('insufficient heldout angle excitation')
    if (prediction['prediction_windows']<cfg['min_prediction_windows'] or prediction['prediction_rmse_rad'] is None or
        prediction['prediction_rmse_rad']>cfg['max_prediction_rmse_rad'] or prediction['prediction_p95_rad']>cfg['max_prediction_p95_rad'] or
        prediction['integral_nrmse'] is None or prediction['integral_nrmse']>cfg['max_integral_nrmse']):
        reasons.append('heldout prediction or integral residual exceeds threshold/coverage')
    result['statistical_checks_pass']=not reasons
    return result
