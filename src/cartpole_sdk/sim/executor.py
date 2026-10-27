"""RAW-demand execution model with the original binary32 1 kHz governor.

MujocoBackend retains its direct/executed-acceleration semantics. This wrapper
models the *external* executor path separately. Faults terminate the rollout at
the fault tick: no physical disabled-motor stopping distance is claimed. Sensor
noise/delay belong to the plant and are never bypassed for policy feedback.
"""
from dataclasses import dataclass, replace
from cartpole_sdk.contracts import (ActionLimits, ActionRequest, Observation, Sample,
                                    ValidationError, validate_action)
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.control.motion import State, advance, _Float32
from .backend import MujocoBackend
from .config import SimulationConfig, SensorConfig


@dataclass(frozen=True)
class ExecutorDiagnostics:
    raw_acceleration_m_s2: float = 0.
    executed_acceleration_m_s2: float = 0.
    active_request_sequence: int | None = None
    expires_at_ns: int | None = None
    enabled: bool = False
    fault_reason: str | None = None
    rail_braking: bool = False
    force_saturated: bool = False


class ExecutorSimulation:
    """One governor update per physics tick; policy decisions are held leases.

    An initial None faults closed, as does expiry. After a fault, reset is required.
    Limits reproduce the current external executor profile; they are not learned.
    """
    action_semantics = 'raw acceleration; binary32 cart_motion governor at 1000 Hz'

    def __init__(self, config: SimulationConfig | None = None, sensors: SensorConfig | None = None,
                 *, parameters: ControllerParameters | None = None):
        self.config = config if config is not None else SimulationConfig(control_steps=10)
        self.sensors = sensors if sensors is not None else SensorConfig()
        self.parameters = parameters if parameters is not None else ControllerParameters(rail=self.config.rail_half_m)
        if not isinstance(self.parameters, ControllerParameters) or abs(self.parameters.rail-self.config.rail_half_m)>1e-8:
            raise ValidationError('executor parameters must match the simulated rail geometry')
        if max(self.parameters.amax_s, self.parameters.amax_b) > 25.:
            raise ValidationError('simulated external governor supports acceleration limits up to 25 m/s²')
        if self.config.physics_dt_ns != 1_000_000:
            raise ValidationError('executor governor requires 1 ms physics ticks')
        if self.config.max_acceleration_m_s2 > 25.:
            raise ValidationError('external raw acceleration limit cannot exceed 25 m/s²')
        self.plant = MujocoBackend(replace(self.config, control_steps=1,
            max_steps=self.config.max_steps*self.config.control_steps,
            max_acceleration_m_s2=25.), self.sensors)
        self.action_limits = ActionLimits(self.config.max_acceleration_m_s2,
            min(self.config.max_action_ttl_ns, 50_000_000), self.config.control_steps*1_000_000)
        self._initialized = False

    @property
    def ground_truth(self): return self.plant.ground_truth

    def reset(self, **kwargs):
        observation = self.plant.reset(**kwargs)
        self.session = self.plant.session
        self.motion = State(self.ground_truth.v, 0., 0)
        self._active = self._previous_action_sequence = None
        self._intervals = self._sequence = 0
        self.terminated = self.truncated = False
        self.fault_reason = None
        self.diagnostics = ExecutorDiagnostics()
        self.observation = observation
        self._initialized = True
        return observation

    def control_observation(self) -> Observation:
        """Augment held sensor samples with current commanded governor state.

        Sensor noise, delay, validity, and original timestamps remain intact;
        only the two executor-state estimates use current availability time.
        """
        self.plant._require_reset()
        observation = self.observation
        timestamp = observation.available_ns
        feedback = tuple(Sample('cart', quantity, 'simulation', value, unit,
            self.sensors.frame_id, method, 'valid', None, 'estimated',
            timestamp, timestamp, timestamp)
            for quantity, value, unit, method in (
                ('acceleration', float(self.motion.acceleration), 'm/s^2', 'commanded_acceleration'),
                ('brake_direction', self.motion.brake_direction, '1', 'governor_brake_direction'),
            ))
        return replace(observation, samples=observation.samples+feedback)

    def _fault(self, reason):
        self.terminated, self.fault_reason = True, reason
        self.diagnostics = replace(self.diagnostics, enabled=False, fault_reason=reason)

    def step(self, request: ActionRequest | None):
        if not self._initialized: raise RuntimeError('reset before stepping')
        if self.terminated or self.truncated: raise RuntimeError('episode ended; reset before stepping')
        now = self.ground_truth.time_ns
        if request is not None:
            validate_action(request, self.observation, self.action_limits, now_ns=now,
                            previous_sequence=self._previous_action_sequence)
            if request.issued_at_ns != now:
                raise ValidationError('issue new actions at current simulation time')
            self._active, self._previous_action_sequence = request, request.sequence
        p = self.parameters
        vmax = min(p.vmax, max(p.vmax_s, p.vmax_b))
        amax = max(p.amax_s, p.amax_b)
        for _ in range(self.config.control_steps):
            now = self.ground_truth.time_ns
            if self._active is None or now >= self._active.expires_at_ns:
                self._fault('no action lease' if self._active is None else 'action lease expired')
                break
            if abs(_Float32(self.ground_truth.x)) > _Float32(self.config.rail_half_m)-_Float32(p.rail_fault_margin):
                self._fault('rail exceeded')
                break
            raw = self._active.acceleration_m_s2
            self.motion = advance(self.motion, self.ground_truth.x, raw, vmax, amax, amax, p.jmax,
                                  self.config.rail_half_m, .001, rail_margin=p.rail_margin)
            if abs(_Float32(self.motion.velocity)) > _Float32(p.vmax)+_Float32(.002):
                self._fault('speed exceeded')
                break
            inner = self.plant.observation
            governed = ActionRequest(inner.session_id, inner.clock_id, inner.sequence,
                inner.sequence, inner.available_ns, inner.available_ns+1_000_000,
                float(self.motion.acceleration))
            try:
                self.plant.step(governed)
            except RuntimeError:
                self._fault('numerical plant failure')
                raise
            self.diagnostics = ExecutorDiagnostics(float(raw), float(self.motion.acceleration),
                self._active.sequence, self._active.expires_at_ns, True, None,
                bool(self.motion.brake_direction), self.plant.diagnostics.force_saturated)
            if (self.plant.terminated or abs(_Float32(self.ground_truth.x)) >
                    _Float32(self.config.rail_half_m)-_Float32(p.rail_fault_margin)):
                self._fault('rail exceeded')
                break
        self._intervals += 1
        self.truncated = self._intervals >= self.config.max_steps
        self._sequence += 1
        self.observation = replace(self.plant.observation, sequence=self._sequence)
        return self.observation

    def close(self): self.plant.close()
