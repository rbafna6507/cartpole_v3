"""Causal, shadow-only complementary angle correction.

Encoder observations supply fast relative motion. Camera updates compare against
wrapped interpolation of encoder history at the camera effective timestamp, never
against the newest encoder angle. A slow rate-limited correction is carried to the
current encoder angle. This is not a model of nonlinear/hysteretic magnetic error,
not an exposure-time estimator, and not a commissioned control observer. Rates
remain source-specific; silently averaging differently delayed rates is avoided.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
import math
from ..contracts import Observation, Sample, validate_observation
from .frames import wrap
from .vision import calibration_fingerprint


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    calibration_id: str
    clock_mapping_id: str

    def __post_init__(self):
        for value in asdict(self).values():
            if not isinstance(value,str) or not value.strip():
                raise ValueError('explicit source, calibration and clock mapping identities required')


@dataclass(frozen=True)
class FusionConfig:
    bias_time_constant_s: float = 2.
    max_bias_rate_rad_s: float = .1
    max_bias_rad: float = .35
    max_innovation_rad: float = .35
    history_capacity: int = 1024
    history_age_ns: int = 2_000_000_000
    max_interpolation_gap_ns: int = 30_000_000
    max_encoder_age_ns: int = 100_000_000
    max_vision_age_ns: int = 200_000_000
    initial_update_s: float = .01

    def __post_init__(self):
        for name in ('bias_time_constant_s','max_bias_rate_rad_s','max_bias_rad','max_innovation_rad','initial_update_s'):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('history_capacity','history_age_ns','max_interpolation_gap_ns','max_encoder_age_ns','max_vision_age_ns'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:
                raise ValueError(f'{name} must be a positive integer')
        if self.history_capacity<2 or self.max_bias_rad>=math.pi or self.max_innovation_rad>=math.pi:
            raise ValueError('history needs two samples; bias and innovation limits must be below pi')


class AngleFusion:
    """Explicit source identities and an already aligned common frame are required.

    Calls must be made serially in receipt order. Reconstruct after calibration,
    clock, session, or source changes. No method can send a rig command. Missing
    vision/encoder makes the fused channel invalid, without implicit fallback.
    """
    def __init__(self, session, *, frame_id, encoder_identity, vision_identity, config=None):
        if not isinstance(frame_id,str) or not frame_id.strip():raise ValueError('explicit common frame required')
        if not isinstance(encoder_identity,SourceIdentity) or not isinstance(vision_identity,SourceIdentity):
            raise ValueError('explicit source identities required')
        self.session=session;self.frame_id=frame_id
        self.identities={'magnetic':encoder_identity,'vision':vision_identity}
        self.config=FusionConfig() if config is None else config
        if not isinstance(self.config,FusionConfig):raise ValueError('FusionConfig required')
        self.calibration_id=calibration_fingerprint(dict(frame_id=frame_id,
            identities={k:asdict(v) for k,v in self.identities.items()},config=asdict(self.config)))
        self.history=deque(maxlen=self.config.history_capacity)
        self.latest={};self.last_effective={};self.last_available=None;self.last_output_ns=None
        self.last_vision_effective_ns=None;self.last_vision_received_ns=None;self.last_vision_available_ns=None
        self.bias=0.;self.innovation=None;self.sequence=0
        self.status='warming_up';self.reason='aligned encoder and vision required'
        self.reset_reason=None;self.accepted_vision=0;self.rejected_vision=0

    @property
    def diagnostics(self):
        return dict(status=self.status,reason=self.reason,bias_rad=self.bias,
            innovation_rad=self.innovation,last_vision_effective_ns=self.last_vision_effective_ns,
            last_vision_received_ns=self.last_vision_received_ns,last_vision_available_ns=self.last_vision_available_ns,
            history_size=len(self.history),
            accepted_vision=self.accepted_vision,rejected_vision=self.rejected_vision,
            calibration_id=self.calibration_id,frame_id=self.frame_id,
            sources={k:asdict(v) for k,v in self.identities.items()},commissioned=False,
            control_eligible=False,timing_basis='explicit clock map; camera host read is an acquisition proxy')

    def _fatal(self, reason):
        self.reset_reason=reason;self.status='reset_required';self.reason=reason
        self.history.clear();self.bias=0.;self.last_vision_effective_ns=None
        raise ValueError(reason+'; reset by constructing a new fusion session')

    def _input(self, observation, source, identity):
        if self.reset_reason:self._fatal(self.reset_reason)
        try:validate_observation(self.session,observation)
        except ValueError as exc:self._fatal(str(exc))
        if identity!=self.identities[source]:self._fatal('source, calibration/reference or clock mapping identity changed')
        candidates=[s for s in observation.samples if s.source==source and s.quantity=='angle' and s.entity==self.session.topology.joints[0]]
        if len(candidates)!=1:self._fatal('exactly one explicit angle source required')
        sample=candidates[0]
        if sample.frame_id!=self.frame_id:self._fatal('common angular frame mismatch')
        last_seen=max(t for t in (self.last_available,self.last_output_ns,0) if t is not None)
        if observation.available_ns<last_seen:
            self._fatal('receipt clock rollback')
        self.last_available=observation.available_ns
        previous=self.last_effective.get(source)
        if sample.effective_ns is not None:
            if sample.effective_ns>observation.available_ns:self._fatal('future effective measurement time')
            if previous is not None and sample.effective_ns<previous:self._fatal(source+' clock rollback')
            if previous==sample.effective_ns and sample.validity=='valid':
                self.reason=source+' repeated effective time';return None
            self.last_effective[source]=sample.effective_ns
        self.latest[source]=sample
        return sample

    def update_encoder(self, observation, *, identity):
        sample=self._input(observation,'magnetic',identity)
        if sample is None:return False
        # A history batch can arrive later than its source's actual serial receipt.
        # Retain healthy historical samples for alignment; output freshness still
        # uses now - effective_ns, so delayed history cannot refresh live state.
        if (sample.validity!='valid' or sample.effective_ns is None or sample.received_ns is None or
            sample.received_ns-sample.effective_ns>self.config.max_encoder_age_ns):
            self.history.clear();self.bias=0.;self.last_vision_effective_ns=None
            self.status='unavailable';self.reason='encoder invalid or stale';return False
        if self.history and sample.effective_ns-self.history[-1].effective_ns>self.config.max_encoder_age_ns:
            self.history.clear();self.bias=0.;self.last_vision_effective_ns=None
        self.history.append(sample)
        while self.history and sample.effective_ns-self.history[0].effective_ns>self.config.history_age_ns:
            self.history.popleft()
        return True

    def _aligned_encoder(self, timestamp):
        if not self.history:return None,'encoder history unavailable'
        if timestamp<self.history[0].effective_ns or timestamp>self.history[-1].effective_ns:
            return None,'vision effective time outside encoder history; no extrapolation'
        previous=None
        for sample in self.history:
            if sample.effective_ns==timestamp:return sample.value,None
            if sample.effective_ns>timestamp:
                gap=sample.effective_ns-previous.effective_ns
                if gap>self.config.max_interpolation_gap_ns:return None,'encoder interpolation gap too large'
                fraction=(timestamp-previous.effective_ns)/gap
                return wrap(previous.value+fraction*wrap(sample.value-previous.value)),None
            previous=sample
        return None,'encoder history unavailable'

    def _reject(self, reason):
        self.rejected_vision+=1;self.status='vision_rejected';self.reason=reason
        return False

    def update_vision(self, observation, *, identity):
        sample=self._input(observation,'vision',identity)
        if sample is None:return self._reject(self.reason)
        if (sample.validity!='valid' or sample.effective_ns is None or
            observation.available_ns-sample.effective_ns>self.config.max_vision_age_ns):
            return self._reject('vision invalid or stale')
        aligned,reason=self._aligned_encoder(sample.effective_ns)
        if reason:return self._reject(reason)
        self.innovation=wrap(sample.value-wrap(aligned+self.bias))
        if abs(self.innovation)>self.config.max_innovation_rad:
            return self._reject('wrapped vision innovation exceeds outlier gate')
        dt=(self.config.initial_update_s if self.last_vision_effective_ns is None else
            (sample.effective_ns-self.last_vision_effective_ns)/1e9)
        # A long dropout cannot produce a one-frame large correction on recovery.
        dt=min(dt,self.config.max_vision_age_ns/1e9)
        alpha=-math.expm1(-dt/self.config.bias_time_constant_s)
        limit=self.config.max_bias_rate_rad_s*dt
        correction=max(-limit,min(limit,alpha*self.innovation))
        candidate=self.bias+correction
        if abs(candidate)>self.config.max_bias_rad:return self._reject('bias exceeds configured bound; recalibration required')
        self.bias=candidate;self.last_vision_effective_ns=sample.effective_ns
        self.last_vision_received_ns=sample.received_ns;self.last_vision_available_ns=observation.available_ns
        self.accepted_vision+=1
        self.status='tracking';self.reason=None
        return True

    def observation(self, *, now_ns):
        if type(now_ns) is not int or now_ns<0:raise ValueError('nonnegative integer now_ns required')
        last_seen=max(t for t in (self.last_available,self.last_output_ns,0) if t is not None)
        if now_ns<last_seen:self._fatal('output clock rollback')
        self.last_output_ns=now_ns
        encoder=self.latest.get('magnetic');reason=self.reset_reason
        if reason is None:
            if (encoder is None or encoder.validity!='valid' or encoder.effective_ns is None or
                now_ns-encoder.effective_ns>self.config.max_encoder_age_ns):reason='encoder unavailable or stale'
            elif self.last_vision_effective_ns is None:reason='aligned vision correction unavailable'
            elif now_ns-self.last_vision_effective_ns>self.config.max_vision_age_ns:reason='vision correction stale or dropout'
        samples=[]
        for source in ('magnetic','vision'):
            sample=self.latest.get(source)
            if sample is not None:
                max_age=self.config.max_encoder_age_ns if source=='magnetic' else self.config.max_vision_age_ns
                if sample.effective_ns is None or now_ns-sample.effective_ns>max_age:
                    sample=replace(sample,value=None,validity='invalid',reason=source+' effective time stale')
                samples.append(sample)
        timestamp=None if encoder is None else encoder.effective_ns
        samples.append(Sample(self.session.topology.joints[0],'angle','fused',
            None if reason else wrap(encoder.value+self.bias),'rad',self.frame_id,
            'shadow complementary bias; delayed wrapped encoder interpolation; '+self.calibration_id,
            'invalid' if reason else 'valid',reason,'estimated',
            None if encoder is None else encoder.acquired_ns,timestamp,now_ns))
        if reason and not self.reset_reason:self.status='unavailable';self.reason=reason
        result=Observation(self.session.session_id,self.session.clock_id,self.sequence,now_ns,
            'shadow observer receipt; source effective times retained',tuple(samples))
        self.sequence+=1
        return result
