"""Nominal operator conditions; never substitute requested angles for readings."""
from copy import deepcopy
import math

SCHEMA='cartpole-collection-protocol-v1'
ROLES=('training','validation','development')

def make_protocol(kind,*,dataset_role='development',run_label='unspecified',angles_deg=None,
                  directions=('left','right'),repetitions=1,cart_restraint=''):
    if kind not in ('passive','pulse','inspect'):raise ValueError('unknown collection procedure')
    if type(repetitions) is not int or not 1<=repetitions<=4:raise ValueError('repetitions must be1..4')
    angles=[] if angles_deg is None else list(angles_deg)
    if kind!='passive' and angles:raise ValueError('release angles apply only to passive collection')
    if (not isinstance(directions,(list,tuple)) or not directions or len(set(directions))!=len(directions)
        or any(d not in ('left','right') for d in directions)):raise ValueError('unique left/right directions required')
    if any(type(a) not in (int,float) or not math.isfinite(a) or not 0<a<=180 for a in angles):
        raise ValueError('requested angles must be finite degrees in(0,180] from hanging')
    if angles and (not isinstance(cart_restraint,str) or not cart_restraint.strip()):
        raise ValueError('describe how the cart will be restrained for requested passive releases')
    releases=[]
    for repeat in range(1,repetitions+1):
        for angle in angles:
            for direction in directions:
                releases.append(dict(release_id=f'release-{len(releases)+1:02d}',angle_from_hanging_deg=float(angle),
                                     direction=direction,repetition=repeat))
    p=dict(schema=SCHEMA,procedure=kind,dataset_role=dataset_role,run_label=run_label,
        operator_cart_restraint=cart_restraint,nominal_angles_only=True,requested_releases=releases)
    return validate_protocol(p,kind)

def validate_protocol(value,kind):
    fields={'schema','procedure','dataset_role','run_label','operator_cart_restraint','nominal_angles_only','requested_releases'}
    if (not isinstance(value,dict) or set(value)!=fields or value['schema']!=SCHEMA or
        value['procedure']!=kind or kind not in ('passive','pulse','inspect') or value['dataset_role'] not in ROLES or
        value['nominal_angles_only'] is not True):raise ValueError('invalid collection protocol schema, role or procedure')
    for key in ('run_label','operator_cart_restraint'):
        v=value[key]
        if not isinstance(v,str) or len(v)>1024 or (key=='run_label' and not v.strip()):raise ValueError('invalid protocol '+key)
    releases=value['requested_releases']
    if not isinstance(releases,list) or len(releases)>12:raise ValueError('protocol supports at most12 requested releases per recording')
    if releases and (kind!='passive' or not value['operator_cart_restraint'].strip()):raise ValueError('passive releases require stated cart restraint')
    names=set()
    for r in releases:
        if not isinstance(r,dict) or set(r)!={'release_id','angle_from_hanging_deg','direction','repetition'}:raise ValueError('invalid protocol release')
        a=r['angle_from_hanging_deg'];name=r['release_id']
        if (not isinstance(name,str) or not name or len(name)>128 or name in names or r['direction'] not in ('left','right')
            or type(r['repetition']) is not int or not 1<=r['repetition']<=4 or type(a) not in (int,float)
            or not math.isfinite(a) or not 0<a<=180):raise ValueError('invalid or duplicate protocol release conditions')
        names.add(name)
    return deepcopy(value)

def instructions(protocol):
    return [f"{r['release_id']}: {r['angle_from_hanging_deg']:g} degrees {r['direction'].upper()} of hanging in the camera view; hold still2s, release without pushing, let settle."
            for r in protocol['requested_releases']]
