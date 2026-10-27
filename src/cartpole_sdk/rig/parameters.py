"""Live tuning names, numeric limits and explicit host profile files.

This module never loads profiles implicitly or opens a rig transport. The
operator console selects its startup profile separately in rig.startup.
"""
from dataclasses import fields
import json
import math
from pathlib import Path
import struct
import tempfile
import os

from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.control import ControllerParameters
from cartpole_sdk.control.parameters import PULSE_SPEED_MAX, validate_coupled

PARAMETER_NAMES = tuple(field.name for field in fields(ControllerParameters))
LEGACY_PARAMETER_NAMES = PARAMETER_NAMES[:30]
TUNABLE_FIRMWARE = 'executor-v6-measuredrail'
VERTICAL_FIRMWARE = 'executor-v7-vertical'
TUNABLE_FIRMWARES = (TUNABLE_FIRMWARE, VERTICAL_FIRMWARE)
GEOMETRY_PARAMETERS = frozenset(('rail','rail_fraction','rail_margin','rail_fault_margin','rail_hysteresis'))
COUPLED_PARAMETERS = GEOMETRY_PARAMETERS | {'vman','manual_speed_max','spin_trip_rad_s','spin_resume_rad_s'}
# Lower bound, upper bound, whether the lower bound is inclusive.
_BOUNDS = {
    'leff':(.02,1,True), 'pw':(.1,30,True), 'bal_pw':(1,20,True),
    'pz':(.05,5,True), 'pc1':(-50,-.1,True), 'pc2':(-50,-.1,True),
    'vmax':(0,PULSE_SPEED_MAX,False), 'vmax_s':(0,PULSE_SPEED_MAX,False), 'vmax_b':(0,PULSE_SPEED_MAX,False),
    'approach_v':(0,PULSE_SPEED_MAX,False), 'catch_v':(0,PULSE_SPEED_MAX,False),
    'amax_s':(0,200,False), 'amax_b':(0,200,False), 'jmax':(0,10000,False),
    'amax_m':(0,200,False), 'jmax_m':(0,10000,False), 'vman':(0,PULSE_SPEED_MAX,True),
    'ke':(0,100,True), 'kpx':(0,1000,True), 'kdx':(0,100,True),
    'phase_soft':(0,50,False), 'catch_a':(.05,1.2,True),
    'catch_r':(.1,30,True), 'giveup':(.1,2.5,True), 'bw':(.1,100,True),
    'approach_angle':(.65,1.5,True), 'catch_da':(0,200,False),
    'spin_trip_rad_s':(0,150,False), 'spin_resume_rad_s':(0,150,False),
    'rail':(.02,.375,False),
    'manual_speed_max':(0,PULSE_SPEED_MAX,False), 'rail_fraction':(.5,.99,True),
    'rail_margin':(.002,.05,True), 'rail_fault_margin':(.001,.049,True),
    'rail_hysteresis':(.001,.05,True), 'rail_recover_v':(0,PULSE_SPEED_MAX,False),
    'rail_recover_a':(0,200,False), 'rail_recover_s':(.1,60,True),
    'rehome_v':(0,PULSE_SPEED_MAX,False), 'rehome_a':(0,200,False),
    'rehome_j':(0,10000,False),
}


def parameter_name(name):
    if not isinstance(name,str) or name not in PARAMETER_NAMES:
        raise ValidationError(f'unknown controller parameter: {name!r}')
    return name


def float32(value):
    return struct.unpack('f',struct.pack('f',value))[0]


def parameter_value(name,value):
    parameter_name(name)
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValidationError(f'{name} must be finite numeric, not boolean')
    try:
        rounded=float32(value)
    except (OverflowError,struct.error):
        raise ValidationError(f'{name} must fit a finite float32') from None
    if not math.isfinite(rounded):
        raise ValidationError(f'{name} must be finite numeric')
    if value != 0 and (rounded == 0 or abs(rounded) < 2**-126):
        raise ValidationError(f'{name} must not underflow float32')
    lower,upper,inclusive=_BOUNDS[name]
    lower,upper=float32(lower),float32(upper)
    if rounded>upper or (rounded<lower if inclusive else rounded<=lower):
        relation='<=' if inclusive else '<'
        raise ValidationError(f'{name} requires {lower:g} {relation} value <= {upper:g}')
    return rounded


def parameter_map(values):
    if not isinstance(values,dict) or not values:
        raise ValidationError('profile parameters must be a nonempty object')
    return {parameter_name(name):parameter_value(name,value) for name,value in values.items()}


def validate_parameters(values):
    try:validate_coupled(values)
    except ValueError as exc:raise ValidationError(str(exc)) from exc


def validate_spin(values):
    if not 0 < values['spin_resume_rad_s'] < values['spin_trip_rad_s']:
        raise ValidationError('spin limits require 0 < spin_resume_rad_s < spin_trip_rad_s')


def read_profile(path):
    def unique(pairs):
        result={}
        for name,value in pairs:
            if name in result:raise ValidationError(f'duplicate profile key: {name}')
            result[name]=value
        return result
    try:
        document=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique)
    except (ValueError,UnicodeError) as exc:
        raise ValidationError(f'invalid parameter profile: {exc}') from exc
    if not isinstance(document,dict) or type(document.get('schema_version')) is not int or document['schema_version']!=1:
        raise ValidationError('parameter profile requires schema_version 1')
    if set(document)-{'schema_version','parameters','source_firmware'}:
        raise ValidationError('unknown parameter profile fields')
    if 'source_firmware' in document and not isinstance(document['source_firmware'],str):
        raise ValidationError('source_firmware must be a string when supplied')
    return parameter_map(document.get('parameters'))


def write_profile(path,values,firmware=None):
    document={'schema_version':1,'parameters':values}
    if firmware is not None:document['source_firmware']=firmware
    destination=Path(path)
    pending=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=destination.parent,
                                         prefix=destination.name+'.',delete=False) as stream:
            pending=Path(stream.name)
            json.dump(document,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
        os.replace(pending,destination)
    finally:
        if pending is not None and pending.exists():pending.unlink()
    return document
