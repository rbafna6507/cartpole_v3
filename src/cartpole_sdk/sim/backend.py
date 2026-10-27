"""MuJoCo plant with cart acceleration intent, bounded force, and sensor transport.

This backend is an unvalidated model, not a hardware safety authority. Expired
leases use zero acceleration demand (which can require a nonzero cart force).
"""
from collections import deque
from dataclasses import dataclass, replace
import math
import mujoco
import numpy as np
from cartpole_sdk.contracts import (ActionLimits, ActionRequest, Observation, Sample,
                                    Session, ValidationError, validate_action)
from .config import SimulationConfig, SensorConfig, integer, finite
from .model import model_xml


@dataclass(frozen=True, slots=True)
class GroundTruth:
    x: float
    v: float
    theta: float
    omega: float
    time_ns: int


@dataclass(frozen=True, slots=True)
class SimulationDiagnostics:
    """Force/acceleration values describe the final completed physics substep."""
    requested_acceleration_m_s2: float = 0.
    requested_force_n: float = 0.
    applied_force_n: float = 0.
    achieved_acceleration_m_s2: float = 0.
    active_request_sequence: int | None = None
    active_expires_at_ns: int | None = None
    force_saturated: bool = False


class MujocoBackend:
    """One cart and one free pole; reset starts a deterministic local clock.

    Sensor acquisitions include t=0. Delays and sample periods count physics
    ticks, not control calls. A held delivery retains its original receipt and
    effective timestamps; the observation availability advances independently.
    """
    def __init__(self, config: SimulationConfig | None = None, sensors: SensorConfig | None = None):
        self.config = config if config is not None else SimulationConfig()
        self.sensors = sensors if sensors is not None else SensorConfig()
        if not isinstance(self.config, SimulationConfig) or not isinstance(self.sensors, SensorConfig):
            raise ValidationError('expected SimulationConfig and SensorConfig')
        self.model_xml = model_xml(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self._cart = self.model.joint('cart_slide').dofadr[0]
        self._pole = self.model.joint('pole_hinge').dofadr[0]
        self._mass = np.empty((self.model.nv, self.model.nv), dtype=np.float64)
        self.action_limits = ActionLimits(self.config.max_acceleration_m_s2,
                                         self.config.max_action_ttl_ns,
                                         self.config.control_steps * self.config.physics_dt_ns)
        self._closed = False
        self._initialized = False

    @property
    def ground_truth(self) -> GroundTruth:
        self._require_reset()
        return GroundTruth(float(self.data.qpos[self._cart]), float(self.data.qvel[self._cart]),
                           math.atan2(math.sin(self.data.qpos[self._pole]), math.cos(self.data.qpos[self._pole])),
                           float(self.data.qvel[self._pole]), self._physics_steps * self.config.physics_dt_ns)

    def _require_reset(self):
        if not self._initialized:
            raise RuntimeError('reset must be called before using simulation state')
        if self._closed:
            raise RuntimeError('backend is closed')

    def reset(self, *, seed: int = 0, state: tuple[float, float, float, float] | None = None,
              session_id: str = 'simulation') -> Observation:
        integer(seed, 'seed')
        values = (0., 0., math.pi, 0.) if state is None else state
        if not isinstance(values, (tuple, list)) or len(values) != 4:
            raise ValidationError('state must contain x, v, theta, omega')
        for value in values:
            finite(value, 'state value')
        if abs(values[0]) >= self.config.rail_half_m:
            raise ValidationError('initial cart position must be inside the rail')
        session = Session(session_id, 'simulation:monotonic')
        if self._closed:
            raise RuntimeError('backend is closed')
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._cart], self.data.qvel[self._cart] = values[:2]
        self.data.qpos[self._pole], self.data.qvel[self._pole] = values[2:]
        mujoco.mj_forward(self.model, self.data)
        self.session = session
        self._rng = np.random.default_rng(seed)
        self._physics_steps = self._control_intervals = self._sequence = 0
        self._previous_action_sequence = None
        self._active = None
        self._numerical_failed = False
        self._pending = deque()
        self._samples = self._unknown_samples()
        self.diagnostics = SimulationDiagnostics()
        self.terminated = self.truncated = False
        self._initialized = True
        self._acquire()
        self._deliver()
        self.observation = self._make_observation()
        return self.observation

    def _unknown_samples(self):
        return tuple(Sample(entity, quantity, 'simulation', None, unit, self.sensors.frame_id,
                            'simulated_sensor', 'unknown', 'sensor delay warmup', 'measured',
                            None, None, None)
                     for entity, quantity, unit in [('cart', 'position', 'm'), ('cart', 'velocity', 'm/s'),
                                                   ('pole', 'angle', 'rad'), ('pole', 'angular_velocity', 'rad/s')])

    def _acquire(self):
        truth = self.ground_truth
        values = (truth.x, truth.v, truth.theta, truth.omega)
        stds = (self.sensors.position_noise_std, self.sensors.velocity_noise_std,
                self.sensors.angle_noise_std, self.sensors.angular_velocity_noise_std)
        dropout = self._rng.random() < self.sensors.dropout_probability
        readings = []
        for sample, value, std in zip(self._unknown_samples(), values, stds):
            noisy = None if dropout else float(value + self._rng.normal(0., std))
            if noisy is not None and sample.quantity == 'angle':
                noisy = math.atan2(math.sin(noisy), math.cos(noisy))
            readings.append(replace(sample, value=noisy, validity='invalid' if dropout else 'valid',
                                    reason='simulated dropout' if dropout else None,
                                    acquired_ns=truth.time_ns, effective_ns=truth.time_ns))
        self._pending.append((self._physics_steps + self.sensors.delay_steps, tuple(readings)))

    def _deliver(self):
        while self._pending and self._pending[0][0] <= self._physics_steps:
            _, samples = self._pending.popleft()
            self._samples = tuple(replace(s, received_ns=self._physics_steps * self.config.physics_dt_ns)
                                  for s in samples)

    def _make_observation(self):
        return Observation(self.session.session_id, self.session.clock_id, self._sequence,
                           self._physics_steps * self.config.physics_dt_ns,
                           'simulation physics clock', self._samples)

    def step(self, request: ActionRequest | None) -> Observation:
        self._require_reset()
        if self._numerical_failed:
            raise RuntimeError('numerical plant failure; reset before stepping')
        if self.terminated or self.truncated:
            raise RuntimeError('episode has ended; reset before stepping')
        now = self.ground_truth.time_ns
        # Validate everything before leases, counters, RNG or MuJoCo state mutate.
        if request is not None:
            validate_action(request, self.observation, self.action_limits, now_ns=now,
                            previous_sequence=self._previous_action_sequence)
            if request.issued_at_ns != now:
                raise ValidationError('new actions must be issued at current simulation time')
            self._active = request
            self._previous_action_sequence = request.sequence
        for _ in range(self.config.control_steps):
            now = self._physics_steps * self.config.physics_dt_ns
            if self._active is not None and now >= self._active.expires_at_ns:
                self._active = None
            acceleration = 0. if self._active is None else self._active.acceleration_m_s2
            # M*qacc + bias = passive + applied; include hinge friction before
            # solving the cart force for the requested acceleration.
            self.data.qfrc_applied[:] = 0.
            self.data.qfrc_applied[self._pole] = (-self.config.hinge_coulomb_friction_nm
                * math.tanh(float(self.data.qvel[self._pole])
                            / self.config.hinge_friction_velocity_scale_rad_s))
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_fullM(self.model, self.data, self._mass)
            residual = self.data.qfrc_bias - self.data.qfrc_passive - self.data.qfrc_applied
            c, h = self._cart, self._pole
            hinge_acceleration = -(self._mass[h, c] * acceleration + residual[h]) / self._mass[h, h]
            force = float(self._mass[c, c] * acceleration + self._mass[c, h] * hinge_acceleration + residual[c])
            applied = float(np.clip(force, -self.config.max_force_n, self.config.max_force_n))
            self.data.qfrc_applied[c] = applied
            mujoco.mj_step(self.model, self.data)
            self._physics_steps += 1
            numerical_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                                  mujoco.mjtWarning.mjWARN_BADQACC)
            if (any(self.data.warning[warning].number for warning in numerical_warnings)
                    or not all(np.all(np.isfinite(values)) for values in
                               (self.data.qpos, self.data.qvel, self.data.qacc))):
                self._numerical_failed = True
                raise RuntimeError('numerical plant failure; reset before stepping')
            self.diagnostics = SimulationDiagnostics(float(acceleration), force, applied,
                float(self.data.qacc[c]), None if self._active is None else self._active.sequence,
                None if self._active is None else self._active.expires_at_ns, applied != force)
            if self._physics_steps % self.sensors.sample_every_steps == 0:
                self._acquire()
            self._deliver()
            if abs(self.ground_truth.x) >= self.config.rail_half_m:
                self.terminated = True
                break
        self._control_intervals += 1
        self.truncated = self._control_intervals >= self.config.max_steps
        self._sequence += 1
        self.observation = self._make_observation()
        return self.observation

    def close(self):
        """No renderer/resources are created by this headless backend."""
        self._closed = True
