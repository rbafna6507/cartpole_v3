"""Configured read-only bridge from SDK mirror/camera snapshots to shadow fusion.

A receipt-anchor clock map is an explicitly chosen approximation. Its transport
latency, uncertainty, residual budget and lifetime are required configuration,
not measurements of exposure time. Changing board session or calibration latches
an error and needs a new observer. This module never opens serial or commands Rig.
"""
from collections import deque
from dataclasses import asdict, replace
from ..contracts import Observation, Sample, Session
from .frames import AngularAlignment
from .fusion import AngleFusion, FusionConfig, SourceIdentity
from .vision import ClockMapping, VisionProvider, calibration_fingerprint


class ShadowObserver:
    def __init__(self, config):
        if config.get('schema_version')!=1:raise ValueError('shadow config schema_version must be 1')
        self.config=config
        self.session=Session(config['session_id'],config['clock_id'])
        self.encoder_alignment=AngularAlignment(**config['encoder_alignment'])
        self.vision_alignment=AngularAlignment(**config['vision_alignment'])
        if (self.encoder_alignment.source!='magnetic' or self.vision_alignment.source!='vision' or
            self.encoder_alignment.target_frame_id!=self.vision_alignment.target_frame_id or
            self.encoder_alignment.reference_id!=self.vision_alignment.reference_id):
            raise ValueError('explicit magnetic/vision alignments must share one target frame and physical reference')
        self.reference=config['encoder_reference']
        if (type(self.reference.get('upright_raw')) is not int or not 0<=self.reference['upright_raw']<4096 or
            not isinstance(self.reference.get('firmware'),str) or not self.reference['firmware']):
            raise ValueError('explicit encoder reference raw count and firmware required')
        if calibration_fingerprint(self.reference)!=self.encoder_alignment.source_calibration_id:
            raise ValueError('encoder alignment reference fingerprint mismatch')
        if calibration_fingerprint(config['vision_calibration'])!=self.vision_alignment.source_calibration_id:
            raise ValueError('camera alignment calibration fingerprint mismatch')
        self.encoder_clock=config['encoder_clock']
        if self.encoder_clock.get('strategy')!='receipt-anchor':
            raise ValueError('explicit encoder receipt-anchor clock strategy required')
        for key in ('receipt_delay_ns','uncertainty_ns','max_receipt_residual_ns','valid_for_ns'):
            value=self.encoder_clock.get(key)
            if type(value) is not int or value<0 or (key=='valid_for_ns' and not value):
                raise ValueError(f'encoder clock {key} must be an explicit nonnegative integer')
        self.vision_mapping=ClockMapping(**config['vision_clock'])
        self.vision_provider=VisionProvider(self.session,frame_id=self.vision_alignment.source_frame_id,
            clock_mapping=self.vision_mapping,calibration=config['vision_calibration'])
        self.fusion_config=FusionConfig(**config.get('fusion',{}))
        self.fusion=None;self.mapping=None;self.encoder_identity=None;self.vision_identity=None
        self.last_encoder_sequence=None;self.last_vision_t=None;self.encoder_generation=None;self.encoder_session=None
        self.error=None;self.sequence=0
        self.pending_vision=deque(maxlen=32);self.pending_vision_dropped=0
        self.history_truncations=0

    def _fail(self, reason):
        self.error=reason
        raise ValueError(reason+'; construct a new shadow observer after checking calibration and clocks')

    def encoder(self, packet, *, received_ns):
        if self.error:self._fail(self.error)
        try:
            return self._encoder(packet,received_ns=received_ns)
        except (KeyError,TypeError,ValueError) as exc:
            self._fail(str(exc))

    def _encoder(self, packet, *, received_ns):
        if packet.get('v')!=1 or type(packet.get('seq')) is not int:raise ValueError('SDK encoder packet required')
        state=packet['sdk_state'];info=packet['sdk_info']
        if packet.get('host_clock_id')!=self.session.clock_id:
            raise ValueError('encoder receipt host clock differs from explicitly selected shadow clock')
        if not isinstance(info,dict):
            # A missing reference cannot retain previously usable fusion.
            if self.fusion is not None:raise ValueError('encoder INFO reference became unavailable')
            return False
        actual_reference=dict(upright_raw=info['encoder_upright_raw'],firmware=info['firmware'])
        if actual_reference!=self.reference:raise ValueError('encoder calibration/reference changed')
        board_clock_id=packet['board_clock_id']
        if board_clock_id!=f"teensy:{state['sid']}" or info['sid']!=state['sid'] or packet['session']!=str(state['sid']):
            raise ValueError('encoder board clock/session identity mismatch')
        for key in ('unwrapped_us','received_ns'):
            if type(state[key]) is not int or state[key]<0:raise ValueError('invalid board/receipt clock')
        serial_receipt=state['received_ns']
        if type(received_ns) is not int or received_ns<serial_receipt:raise ValueError('future serial receipt')
        if self.mapping is None:
            offset=serial_receipt-state['unwrapped_us']*1000-self.encoder_clock['receipt_delay_ns']
            self.mapping=dict(self.encoder_clock,source_clock_id=board_clock_id,
                session_clock_id=self.session.clock_id,offset_ns=offset,anchor_received_ns=serial_receipt)
            self.encoder_generation=packet['generation'];self.encoder_session=packet['session']
            self.encoder_identity=SourceIdentity(self.encoder_alignment.source_id,
                self.encoder_alignment.calibration_id,calibration_fingerprint(self.mapping))
            self.vision_identity=SourceIdentity(self.vision_alignment.source_id,
                self.vision_alignment.calibration_id,calibration_fingerprint(asdict(self.vision_mapping)))
            self.fusion=AngleFusion(self.session,frame_id=self.encoder_alignment.target_frame_id,
                encoder_identity=self.encoder_identity,vision_identity=self.vision_identity,config=self.fusion_config)
        if (board_clock_id!=self.mapping['source_clock_id'] or packet['generation']!=self.encoder_generation):
            raise ValueError('encoder session or reference generation changed')
        if self.last_encoder_sequence is not None:
            if packet['seq']<self.last_encoder_sequence:raise ValueError('encoder sequence rollback')
            if packet['seq']==self.last_encoder_sequence:return False
        self.last_encoder_sequence=packet['seq']
        if received_ns-self.mapping['anchor_received_ns']>self.encoder_clock['valid_for_ns']:
            raise ValueError('encoder clock mapping expired; re-establish bounded clock mapping')
        timestamp=state['unwrapped_us']*1000+self.mapping['offset_ns']
        residual=serial_receipt-timestamp-self.encoder_clock['receipt_delay_ns']
        if timestamp<0 or timestamp>serial_receipt or abs(residual)>self.encoder_clock['max_receipt_residual_ns']:
            raise ValueError('encoder receipt clock mapping residual exceeded')
        valid=(packet['valid'] is True and bool(state['flags'] & 1) and
               not state['flags'] & 256 and state['fault']==0)
        sample=Sample(self.session.topology.joints[0],'angle','magnetic',state['theta'] if valid else None,
            'rad',self.encoder_alignment.source_frame_id,'magnetic decoded; explicit receipt-anchor clock',
            'valid' if valid else 'invalid',None if valid else 'current executor encoder health invalid',
            'measured',timestamp,timestamp,serial_receipt)
        observation=Observation(self.session.session_id,self.session.clock_id,packet['seq'],received_ns,
            'shadow HTTP/record receipt; board timestamp explicitly mapped',(sample,))
        observation=self.encoder_alignment.apply(observation,source_id=self.encoder_alignment.source_id,
            source_calibration_id=calibration_fingerprint(actual_reference))
        return self.fusion.update_encoder(observation,identity=self.encoder_identity)

    def encoder_batch(self, response, *, received_ns):
        """Ingest bounded source history, retaining original serial receipts.

        HTTP receipt is availability to this observer; it does not replace the
        serial owner's earlier receipt or downsample real acquisition history.
        """
        packets=response.get('packets')
        if not isinstance(packets,list) or len(packets)>2048:
            self._fail('bounded encoder packet history required; update camera service')
        if response.get('history_truncated') is True:self.history_truncations+=1
        return sum(self.encoder(packet,received_ns=received_ns) for packet in packets)

    def _vision_observation(self, snapshot, received_ns):
        if self.error:self._fail(self.error)
        try:
            if calibration_fingerprint(snapshot.get('calibration'))!=self.vision_alignment.source_calibration_id:
                raise ValueError('vision calibration changed')
            # A restored calibration can precede the first camera frame. This
            # explicit unavailable heartbeat carries no acquisition to enqueue.
            # Validate calibration above even when no measurement exists.
            if snapshot.get('t') is None and snapshot.get('stale') is True and snapshot.get('angle_valid') is False:
                return None
            if self.last_vision_t==snapshot['t']:return None
            observation=self.vision_provider.observation(snapshot,received_ns=received_ns,
                source_clock_id=self.vision_mapping.source_clock_id)
            self.last_vision_t=snapshot['t']
            return self.vision_alignment.apply(observation,source_id=self.vision_alignment.source_id,
                source_calibration_id=calibration_fingerprint(snapshot['calibration']))
        except (KeyError,TypeError,ValueError) as exc:self._fail(str(exc))

    def vision(self, snapshot, *, received_ns):
        if self.error:self._fail(self.error)
        if self.fusion is None:return False
        observation=self._vision_observation(snapshot,received_ns)
        return False if observation is None else self.fusion.update_vision(observation,identity=self.vision_identity)

    def enqueue_vision(self, snapshot, *, received_ns):
        """Validate now; defer correction until actual encoder history brackets it.

        Original camera receipt/acquisition timestamps are immutable. A deferred
        correction becomes available only at its eventual processing time.
        """
        observation=self._vision_observation(snapshot,received_ns)
        if observation is None:return False
        if len(self.pending_vision)==self.pending_vision.maxlen:self.pending_vision_dropped+=1
        self.pending_vision.append(observation)
        return True

    def flush_pending(self, *, now_ns):
        if self.error:self._fail(self.error)
        accepted=0
        while self.pending_vision:
            observation=self.pending_vision[0]
            if now_ns<observation.available_ns:self._fail('pending camera receipt clock rollback')
            sample=next(s for s in observation.samples if s.source=='vision' and s.quantity=='angle')
            expired=(sample.effective_ns is None or now_ns-sample.effective_ns>self.fusion_config.max_vision_age_ns)
            if self.fusion is None:
                if not expired:break
                self.pending_vision.popleft();self.pending_vision_dropped+=1
                continue
            latest=self.fusion.history[-1].effective_ns if self.fusion.history else None
            if sample.validity=='valid' and not expired and (latest is None or sample.effective_ns>latest):
                break  # A future encoder bracket may arrive on the next HTTP poll.
            self.pending_vision.popleft()
            observation=replace(observation,available_ns=now_ns,
                availability_basis='deferred camera correction; original receipt retained; actual encoder bracket available now')
            accepted+=self.fusion.update_vision(observation,identity=self.vision_identity)
        return accepted

    def packet(self, *, now_ns):
        if self.fusion is None:
            reason=self.error or 'waiting for encoder packet with explicit matching INFO reference'
            observation=Observation(self.session.session_id,self.session.clock_id,self.sequence,now_ns,
                'shadow monitor receipt',(Sample(self.session.topology.joints[0],'angle','fused',None,'rad',
                    self.encoder_alignment.target_frame_id,'uninitialized shadow observer','invalid',reason,
                    'estimated',None,None,now_ns),))
            diagnostics=dict(status='unavailable',reason=reason)
        else:
            observation=self.fusion.observation(now_ns=now_ns);diagnostics=self.fusion.diagnostics
        if self.error:
            observation=replace(observation,samples=tuple(replace(s,value=None,validity='invalid',reason=self.error)
                if s.source=='fused' else s for s in observation.samples))
            diagnostics.update(status='reset_required',reason=self.error)
        diagnostics.update(pending_vision=len(self.pending_vision),pending_vision_dropped=self.pending_vision_dropped,
            encoder_history_truncations=self.history_truncations,
            encoder_clock=self.mapping,vision_clock=asdict(self.vision_mapping),
            configuration_sha256=calibration_fingerprint(self.config),commissioned=False,control_eligible=False)
        result=dict(schema='cartpole-sdk-shadow-v1',session_id=self.session.session_id,
            clock_id=self.session.clock_id,sequence=self.sequence,host_t=now_ns/1e9,
            observation=asdict(observation),diagnostics=diagnostics,commissioned=False,control_eligible=False)
        self.sequence+=1
        return result
