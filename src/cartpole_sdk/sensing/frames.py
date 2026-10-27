"""Explicit angular coordinate transforms; no inference from a frame label.

An anchor records the original source calibration and a measured zero in that
source's coordinates. A direct upright camera anchor must be supplied by the
operator; a magnetic half-turn assumption is not a camera calibration.
"""
from dataclasses import asdict, dataclass, replace
import math
from .vision import calibration_fingerprint


def wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


@dataclass(frozen=True)
class AngularAlignment:
    source: str
    source_id: str
    source_frame_id: str
    source_calibration_id: str
    target_frame_id: str
    reference_id: str
    zero_rad: float
    sign: int

    def __post_init__(self):
        if self.source not in ('magnetic','vision'):
            raise ValueError('alignment source must be magnetic or vision')
        for name in ('source_id','source_frame_id','source_calibration_id','target_frame_id','reference_id'):
            value=getattr(self,name)
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f'explicit {name} required')
        if type(self.zero_rad) not in (int,float) or not math.isfinite(self.zero_rad):
            raise ValueError('finite zero_rad required')
        if type(self.sign) is not int or self.sign not in (-1,1):
            raise ValueError('explicit angular sign must be -1 or 1')

    @property
    def calibration_id(self):
        return calibration_fingerprint(asdict(self))

    def apply(self, observation, *, source_id, source_calibration_id):
        if (source_id,source_calibration_id)!=(self.source_id,self.source_calibration_id):
            raise ValueError('angular alignment source/calibration identity changed')
        samples=[]
        found=False
        for sample in observation.samples:
            if sample.source==self.source and sample.quantity in ('angle','angular_velocity'):
                found=True
                if sample.frame_id!=self.source_frame_id:
                    raise ValueError('angular alignment source frame mismatch')
                value=sample.value
                if value is not None:
                    value=wrap(self.sign*(value-self.zero_rad)) if sample.quantity=='angle' else self.sign*value
                sample=replace(sample,value=value,frame_id=self.target_frame_id,
                    method=sample.method+'; explicit angular anchor '+self.calibration_id)
            samples.append(sample)
        if not found:
            raise ValueError('angular alignment source channel missing')
        return replace(observation,samples=tuple(samples))
