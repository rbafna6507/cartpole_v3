"""Explicit, causal camera/encoder composition for development policy inference.

The caller supplies measured angular anchors and a camera-to-board clock map.
Composition retains the board sequence and availability used by action leases.
Camera receipts newer than that snapshot wait for a later board observation.
This module opens no device, changes no calibration and grants no deployment
acceptance. Fused angle is the existing SDK complementary observer; angular
velocity remains the independently named, aligned magnetic rate estimate.
"""
from collections import deque
from dataclasses import asdict, replace

from ..contracts import Sample, ValidationError, validate_observation
from ..policies.rig_observation import rig_channels, validate_observation_contract
from .frames import AngularAlignment
from .fusion import AngleFusion, FusionConfig, SourceIdentity
from .vision import VisionProvider, calibration_fingerprint


class PolicyObservationAssembler:
    def __init__(self, session, *, contract, vision_provider, vision_alignment,
                 encoder_alignment=None, clock_uncertainty_ns=0):
        self.session = session
        self.contract = validate_observation_contract(contract)
        mode = self.contract['angle_source']
        if mode not in ('vision', 'fused'):
            raise ValidationError('camera assembler requires vision or fused mode')
        if not isinstance(vision_provider, VisionProvider) or vision_provider.session != session:
            raise ValidationError('explicit camera provider in the board session required')
        if (not isinstance(vision_alignment, AngularAlignment) or vision_alignment.source != 'vision' or
                vision_alignment.source_frame_id != vision_provider.frame_id or
                vision_alignment.source_calibration_id != vision_provider.calibration_id or
                vision_alignment.target_frame_id != self.contract['angle_frame_id']):
            raise ValidationError('explicit matching camera calibration and angular alignment required')
        if vision_provider.max_age_ns != self.contract['vision_max_age_ns']:
            raise ValidationError('camera provider age budget differs from observation contract')
        if (type(clock_uncertainty_ns) is not int or clock_uncertainty_ns < 0 or
                clock_uncertainty_ns >= self.contract['vision_max_age_ns']):
            raise ValidationError('clock uncertainty must be nonnegative and fit inside camera age budget')
        self.clock_uncertainty_ns = clock_uncertainty_ns
        self.provider, self.vision_alignment = vision_provider, vision_alignment
        self.encoder_alignment = encoder_alignment
        self.channels = rig_channels(angle_source=mode, angle_frame_id=self.contract['angle_frame_id'])
        self.fusion = None
        if mode == 'fused':
            if (not isinstance(encoder_alignment, AngularAlignment) or encoder_alignment.source != 'magnetic' or
                    encoder_alignment.target_frame_id != vision_alignment.target_frame_id or
                    encoder_alignment.reference_id != vision_alignment.reference_id):
                raise ValidationError('fusion requires explicit alignments sharing a measured physical reference')
            self.encoder_identity = SourceIdentity(encoder_alignment.source_id,
                encoder_alignment.calibration_id, 'native-board-clock:'+session.clock_id)
            self.vision_identity = SourceIdentity(vision_alignment.source_id,
                vision_alignment.calibration_id, calibration_fingerprint(asdict(vision_provider.mapping)))
            self.fusion = AngleFusion(session, frame_id=self.contract['angle_frame_id'],
                encoder_identity=self.encoder_identity, vision_identity=self.vision_identity,
                config=FusionConfig(max_encoder_age_ns=20_000_000,
                    max_vision_age_ns=self.contract['vision_max_age_ns']-self.clock_uncertainty_ns))
        self.pending = deque()
        self.latest_vision = None
        self.last_vision_t = None
        self.last_board = None
        self.error = None

    def _fail(self, reason):
        self.error = str(reason)
        raise ValidationError(self.error+'; reconstruct observation assembler after checking calibration and clocks')

    def enqueue_vision(self, snapshot, *, received_ns, source_clock_id):
        """Validate a snapshot at actual mapped receipt; no extrapolation."""
        if self.error:
            self._fail(self.error)
        try:
            if calibration_fingerprint(snapshot.get('calibration')) != self.provider.calibration_id:
                raise ValueError('vision calibration changed')
            if source_clock_id != self.provider.mapping.source_clock_id:
                raise ValueError('camera source clock differs from explicit clock map')
            if snapshot.get('t') is None and snapshot.get('stale') is True and snapshot.get('angle_valid') is False:
                return False
            if self.last_vision_t == snapshot['t']:
                return False
            observation = self.provider.observation(snapshot, received_ns=received_ns,
                                                    source_clock_id=source_clock_id)
            observation = self.vision_alignment.apply(observation,
                source_id=self.vision_alignment.source_id,
                source_calibration_id=self.provider.calibration_id)
            # Retain nominal mapped sample timestamps for interpolation and
            # diagnostics. Eligibility waits until latest-plausible receipt;
            # freshness below uses the oldest-plausible effective time.
            observation = replace(observation,
                available_ns=observation.available_ns+self.clock_uncertainty_ns,
                availability_basis='camera eligibility after bounded clock uncertainty; nominal source timestamps retained')
            if len(self.pending) >= 128:
                raise ValueError('camera pending queue exceeded bounded capacity')
            self.pending.append(observation)
            self.last_vision_t = snapshot['t']
            return True
        except (ValueError, TypeError, KeyError) as exc:
            self._fail(exc)

    def compose(self, control_observation):
        """Add camera/fused channels without changing board action timestamps."""
        if self.error:
            self._fail(self.error)
        try:
            return self._compose(control_observation)
        except (ValueError, TypeError, KeyError) as exc:
            self._fail(exc)

    def _compose(self, board):
        validate_observation(self.session, board)
        now = board.available_ns
        if self.last_board is not None and (now < self.last_board[1] or board.sequence < self.last_board[0]):
            raise ValueError('board observation rollback')
        new_board = self.last_board != (board.sequence, now)
        self.last_board = board.sequence, now
        aligned = board
        if self.fusion is not None:
            aligned = self.encoder_alignment.apply(board, source_id=self.encoder_alignment.source_id,
                source_calibration_id=self.encoder_alignment.source_calibration_id)
            if new_board:
                self.fusion.update_encoder(aligned, identity=self.encoder_identity)
        while self.pending and self.pending[0].available_ns <= now:
            camera = self.pending.popleft()
            self.latest_vision = camera
            if self.fusion is not None:
                # Sensor receipt/effective times remain immutable. Correction is
                # available now, after the actual encoder bracket arrived.
                self.fusion.update_vision(replace(camera, available_ns=now), identity=self.vision_identity)
        samples = {s.channel: s for s in aligned.samples}
        for quantity, unit in (('angle', 'rad'), ('angular_velocity', 'rad/s')):
            sample = None if self.latest_vision is None else next(
                s for s in self.latest_vision.samples if s.source == 'vision' and
                s.entity == self.session.topology.joints[0] and s.quantity == quantity)
            if sample is None:
                sample = Sample(self.session.topology.joints[0], quantity, 'vision', None, unit,
                    self.contract['angle_frame_id'], 'camera acquisition pending', 'unknown',
                    'camera measurement or causal derivative unavailable', 'estimated', None, None, None)
            elif (sample.effective_ns is None or
                    now-sample.effective_ns+self.clock_uncertainty_ns > self.contract['vision_max_age_ns']):
                sample = replace(sample, value=None, validity='invalid', reason='camera effective time stale including clock uncertainty')
            samples[sample.channel] = sample
        if self.fusion is not None:
            fused = self.fusion.observation(now_ns=now).samples[-1]
            samples[fused.channel] = fused
        return replace(board, samples=tuple(samples.values()),
            availability_basis=board.availability_basis+'; explicit causal camera composition')

    @property
    def diagnostics(self):
        return dict(angle_source=self.contract['angle_source'], pending_camera=len(self.pending),
                    clock_uncertainty_ns=self.clock_uncertainty_ns,
                    error=self.error, fusion=None if self.fusion is None else self.fusion.diagnostics,
                    hardware_accepted=False)
