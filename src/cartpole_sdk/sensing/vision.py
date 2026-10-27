"""Read-only vision3 snapshots converted into source-explicit SDK observations.

vision3 t is host read completion, an acquisition proxy rather than a sensor
exposure timestamp. Display/comparison fusion is deliberately never consumed.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from urllib.parse import urlsplit
from urllib.request import urlopen
from ..contracts import Observation, Sample, Session, validate_observation
from ..recording import strict_json_loads


def _finite(value, label):
    if type(value) not in (int,float) or not math.isfinite(value): raise ValueError(f'{label} must be finite')
    return float(value)


def _integer(value,label):
    if type(value) is not int or value<0:raise ValueError(f'{label} must be nonnegative integer')


def calibration_fingerprint(calibration):
    """Content address the full caller-approved calibration, not a stale flag."""
    if not isinstance(calibration,dict):raise ValueError('calibration dictionary required')
    return hashlib.sha256(json.dumps(calibration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ClockMapping:
    source_clock_id: str
    session_clock_id: str
    offset_ns: int

    def __post_init__(self):
        if (not isinstance(self.source_clock_id,str) or not self.source_clock_id.strip()
            or not isinstance(self.session_clock_id,str) or not self.session_clock_id.strip() or type(self.offset_ns) is not int):
            raise ValueError('explicit source/session clock identities and integer offset_ns required')

    def map_seconds(self,value):
        t=_finite(value,'source timestamp')
        result=int(t*1e9)+self.offset_ns
        if result<0:raise ValueError('mapped timestamp precedes session clock origin')
        return result


class VisionProvider:
    def __init__(self,session:Session,*,frame_id:str,clock_mapping:ClockMapping,calibration:dict,max_age_ns=150_000_000):
        if not isinstance(clock_mapping,ClockMapping) or clock_mapping.session_clock_id!=session.clock_id:
            raise ValueError('explicit calibrated clock mapping must match session clock')
        if not isinstance(frame_id,str) or not frame_id:raise ValueError('explicit calibrated frame_id required')
        _integer(max_age_ns,'max_age_ns')
        self.session=session;self.frame_id=frame_id;self.mapping=clock_mapping;self.max_age_ns=max_age_ns
        self.calibration_id=calibration_fingerprint(calibration)
        self.sequence=0;self.last_acquired_ns=None;self.last_timing={}

    def observation(self,snapshot,*,received_ns:int,source_clock_id:str):
        """Convert a new snapshot; caller timestamps local receipt explicitly.

        A calibration change needs a new provider/session or an explicit caller
        reset by reconstruction. Never retains derivatives across calibration.
        """
        _integer(received_ns,'received_ns')
        if source_clock_id!=self.mapping.source_clock_id:raise ValueError('source clock does not match calibrated mapping')
        if calibration_fingerprint(snapshot.get('calibration'))!=self.calibration_id:
            raise ValueError('vision calibration changed; reset provider/session and explicitly approve new calibration/frame')
        acquired=self.mapping.map_seconds(snapshot['t'])
        if acquired>received_ns:raise ValueError('future host-read acquisition proxy; clock mapping invalid')
        if self.last_acquired_ns is not None and acquired<=self.last_acquired_ns:
            raise ValueError('vision host-read timestamps must increase; repeated/stale frame')
        result_ms=snapshot.get('host_read_to_result_ms')
        if result_ms is not None:
            result_ms=_finite(result_ms,'host_read_to_result_ms')
            if result_ms<0 or acquired+int(result_ms*1e6)>received_ns:raise ValueError('result timestamp is in future; clock mapping/receipt invalid')
        stale=snapshot.get('stale',False) is True or received_ns-acquired>self.max_age_ns
        calibration=snapshot['calibration']
        home=(calibration.get('home_angle_rad') is not None and calibration.get('home_length_px') is not None)
        if home:
            _finite(calibration['home_angle_rad'],'home_angle_rad')
            home=_finite(calibration['home_length_px'],'home_length_px')>0
        scale=calibration.get('cart_span_mm')
        cart_scale=scale is not None and _finite(scale,'cart_span_mm')>0
        for key in ('cart_a_px','cart_b_px'):
            point=calibration.get(key)
            cart_scale=cart_scale and isinstance(point,(list,tuple)) and len(point)==2
            if point is not None:
                for number in point:_finite(number,key)
        if cart_scale and calibration['cart_a_px']==calibration['cart_b_px']:cart_scale=False
        angle_ok=snapshot.get('angle_valid') is True and snapshot.get('calibrated') is True and home
        cart_ok=snapshot.get('cart_valid') is True and cart_scale
        def sample(entity,quantity,field,unit,ok,effective_field,method,kind):
            effective=acquired if effective_field is None else (None if snapshot.get(effective_field) is None else self.mapping.map_seconds(snapshot[effective_field]))
            if effective is not None and effective>acquired:raise ValueError('future effective measurement timestamp')
            value=snapshot.get(field)
            reason=None
            if stale:reason='camera stale by mapped acquisition time'
            elif not ok:reason='angle invalid or hanging calibration unavailable' if quantity in ('angle','angular_velocity') else 'cart invalid or metric cart scale unavailable'
            elif value is None or effective is None:reason='measurement/derivative warming up or effective time unknown'
            elif received_ns-effective>self.max_age_ns:reason='effective measurement time stale'
            if reason is None:value=_finite(value,field)
            else:value=None
            return Sample(entity,quantity,'vision',value,unit,self.frame_id,method,'valid' if reason is None else 'invalid',reason,kind,acquired,effective,received_ns)
        samples=(sample(self.session.topology.joints[0],'angle','theta_rad','rad',angle_ok,None,'vision3 arm angle; host-read acquisition proxy','measured'),
                 sample(self.session.topology.joints[0],'angular_velocity','omega_rad_s','rad/s',angle_ok,'omega_effective_t','vision3 causal local slope','estimated'),
                 sample(self.session.topology.cart,'position','cart_x_m','m',cart_ok,None,'vision3 calibrated cart '+str(snapshot.get('cart_source') or 'unknown'),'measured'),
                 sample(self.session.topology.cart,'velocity','cart_v_m_s','m/s',cart_ok,'cart_velocity_effective_t','vision3 causal cart local slope','estimated'))
        observation=Observation(self.session.session_id,self.session.clock_id,self.sequence,received_ns,'caller receipt; host-read is acquisition proxy',samples)
        validate_observation(self.session,observation)
        self.last_acquired_ns=acquired;self.sequence+=1
        self.last_timing=dict(host_read_source_t_s=snapshot['t'],host_read_to_result_ms=result_ms,calibration_sha256=self.calibration_id,
                              source_clock_id=source_clock_id,mapping_offset_ns=self.mapping.offset_ns)
        return observation


class VisionHTTPClient:
    """Optional read-only HTTP fetch isolated from observation conversion."""
    def __init__(self,url,*,timeout_s=2.,max_bytes=2_000_000):
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc:raise ValueError('HTTP client requires http/https URL')
        if timeout_s<=0 or not math.isfinite(timeout_s):raise ValueError('positive finite HTTP timeout required')
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive integer max_bytes required')
        self.url=url;self.timeout_s=timeout_s;self.max_bytes=max_bytes

    def fetch(self):
        with urlopen(self.url,timeout=self.timeout_s) as response:
            payload=response.read(self.max_bytes+1)
        if len(payload)>self.max_bytes:raise ValueError('vision snapshot exceeds byte limit')
        snapshot=strict_json_loads(payload.decode('utf-8'))
        if not isinstance(snapshot,dict):raise ValueError('vision snapshot must be JSON object')
        return snapshot
