"""Saved candidate controller settings, independent of rig calibration."""
from dataclasses import asdict, dataclass, fields
from importlib.resources import files
import json
import math
from .motion import _Float32

# Pulse-generation ceiling, not measured motor tracking capability.
PULSE_SPEED_MAX = (1_000_000 / 7 * .5) * 120 / 3_200_000


@dataclass(frozen=True)
class ControllerParameters:
    leff: float = .124
    pw: float = 4.5
    pz: float = .85
    bal_pw: float = 4.0
    pc1: float = -3.0
    pc2: float = -4.0
    vmax: float = 1.5
    vmax_s: float = 1.5
    vmax_b: float = 1.5
    approach_v: float = .3
    catch_v: float = .3
    approach_angle: float = .8
    catch_da: float = 4.0
    amax_s: float = 25.0
    amax_b: float = 25.0
    jmax: float = 150.0
    amax_m: float = .5
    jmax_m: float = 10.0
    ke: float = 4.0
    kpx: float = 40.0
    kdx: float = 2.0
    phase_soft: float = 2.0
    catch_a: float = .6
    catch_r: float = 3.0
    giveup: float = .8
    spin_trip_rad_s: float = 150.0
    spin_resume_rad_s: float = 10.0
    rail: float = .26
    bw: float = 30.0
    vman: float = .05
    manual_speed_max: float = .15
    rail_fraction: float = .95
    rail_margin: float = .015
    rail_fault_margin: float = .010
    rail_hysteresis: float = .005
    rail_recover_v: float = .15
    rail_recover_a: float = 1.5
    rail_recover_s: float = 8.0
    rehome_v: float = .08
    rehome_a: float = .5
    rehome_j: float = 10.0

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{f.name} must be finite numeric')
        positive = ('leff','pw','bal_pw','pz','vmax','vmax_s','vmax_b','approach_v','catch_v','approach_angle','catch_da','amax_s','amax_b','jmax','amax_m','jmax_m','phase_soft','catch_a','catch_r','giveup')
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError('physical/controller limits must be positive')
        if self.rail <= .02 or not .1 <= self.bw <= 100:
            raise ValueError('rail or estimator bandwidth outside supported range')
        if not 0 < self.spin_resume_rad_s < self.spin_trip_rad_s:
            raise ValueError('spin limits require 0 < resume < trip')
        for name in ('vmax','vmax_s','vmax_b','approach_v','catch_v','manual_speed_max','rail_recover_v','rehome_v'):
            if not 0 < getattr(self,name) <= PULSE_SPEED_MAX + 1e-7:
                raise ValueError(f'{name} exceeds pulse speed envelope')
        for name in ('amax_s','amax_b','amax_m','rail_recover_a','rehome_a','catch_da'):
            if not 0 < getattr(self,name) <= 200:
                raise ValueError(f'{name} exceeds acceleration envelope')
        for name in ('jmax','jmax_m','rehome_j'):
            if not 0 < getattr(self,name) <= 10000:
                raise ValueError(f'{name} exceeds jerk envelope')
        for name,lower,upper in (('rail_fraction',.5,.99),('rail_margin',.002,.05),
                                 ('rail_fault_margin',.001,.049),('rail_hysteresis',.001,.05),
                                 ('rail_recover_s',.1,60)):
            if not lower <= getattr(self,name) <= upper:
                raise ValueError(f'{name} outside supported range')
        validate_coupled(asdict(self))

def validate_coupled(values):
    """Validate a complete supported catalog; never fill absent legacy fields."""
    try:values={name:_Float32(value) for name,value in values.items()}
    except OverflowError as exc:raise ValueError('parameters must fit finite float32') from exc
    if not 0 < values['spin_resume_rad_s'] < values['spin_trip_rad_s']:
        raise ValueError('spin limits require 0 < spin_resume_rad_s < spin_trip_rad_s')
    if 'manual_speed_max' in values:
        if not 0 <= values['vman'] <= values['manual_speed_max']:
            raise ValueError('vman must be between zero and manual_speed_max')
        if values['rail_margin'] - values['rail_fault_margin'] < _Float32(.001) - _Float32(1e-8):
            raise ValueError('rail_margin must exceed rail_fault_margin by at least 0.001 m')
        if values['rail'] - values['rail_margin'] - values['rail_hysteresis'] < _Float32(1) / _Float32(26666.6666667):
            raise ValueError('rail must exceed rail_margin + rail_hysteresis by at least one step')


@dataclass(frozen=True)
class Gains:
    k1: float
    k2: float
    k3: float
    k4: float

def load_tuned_parameters() -> ControllerParameters:
    preset = json.loads(files(__package__).joinpath('tuned_parameters.json').read_text())
    # Historical offline preset predates motion tunables. Explicit class defaults
    # preserve its governor behavior; live profile loading never uses this path.
    return ControllerParameters(**{f.name: preset['settings'][f.name] for f in fields(ControllerParameters)
                                   if f.name in preset['settings']})

def gains(p: ControllerParameters, upright: bool = False) -> Gains:
    pw = p.bal_pw if upright else p.pw
    b1, b0 = 2*p.pz*pw, pw*pw
    c1, c0 = -(p.pc1+p.pc2), p.pc1*p.pc2
    k3 = -b0*c0*p.leff/9.81
    k4 = -(b1*c0+b0*c1)*p.leff/9.81
    k1 = p.leff*k3-9.81-p.leff*(b0+b1*c1+c0)
    k2 = p.leff*(k4-b1-c1)
    return Gains(k1,k2,k3,k4)
