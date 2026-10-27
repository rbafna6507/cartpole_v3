"""Headless Gymnasium interface to the provisional acceleration-driven plant.

Action: one cart acceleration in m/s². Sensor observation: x, v, sin(theta),
cos(theta), omega, with a five-element validity mask and effective age in seconds.
Unavailable values use finite zero placeholders with mask=0; unknown age is -1.
Reward uses true plant cos(theta) - (x/rail_half_m)**2 - .001*action**2, an
illustrative upright/centering/control-cost objective, not a validated target.
Default control interval is 10 ms; backend/reference integration defaults to 1 ms.
"""
from dataclasses import asdict
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from cartpole_sdk.contracts import ActionRequest, ValidationError
from .backend import MujocoBackend
from .config import SimulationConfig, SensorConfig


class CartpoleEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config: SimulationConfig | None = None,
                 sensors: SensorConfig | None = None, render_mode=None):
        if render_mode is not None:
            raise ValueError('only headless render_mode=None is supported')
        self.render_mode = render_mode
        self.backend = MujocoBackend(config if config is not None else SimulationConfig(control_steps=10), sensors)
        limit = self.backend.config.max_acceleration_m_s2
        self.action_space = spaces.Box(-limit, limit, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64),
            'valid': spaces.MultiBinary(5),
            'age_s': spaces.Box(-1., np.inf, shape=(5,), dtype=np.float64),
        })
        self._action_sequence = 0

    def _sensor_observation(self):
        observation = self.backend.observation
        samples = {s.quantity: s for s in observation.samples}
        state, valid, ages = [], [], []
        for quantity, transform in [('position', float), ('velocity', float), ('angle', math.sin),
                                    ('angle', math.cos), ('angular_velocity', float)]:
            sample = samples[quantity]
            usable = sample.validity == 'valid' and sample.value is not None
            state.append(transform(sample.value) if usable else 0.)
            valid.append(int(usable))
            ages.append(-1. if sample.effective_ns is None else
                        (observation.available_ns - sample.effective_ns) / 1e9)
        return {'state': np.array(state, dtype=np.float64),
                'valid': np.array(valid, dtype=np.int8), 'age_s': np.array(ages, dtype=np.float64)}

    def _info(self):
        return {'ground_truth': asdict(self.backend.ground_truth),
                'diagnostics': asdict(self.backend.diagnostics),
                'observation_sequence': self.backend.observation.sequence,
                'available_ns': self.backend.observation.available_ns}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise ValueError('options must be a dictionary')
        state = None if options is None else options.get('state')
        # Derive a backend seed from Gym's generator, so explicit seeds repeat and
        # seed=None follows the continuing Gym random stream.
        sensor_seed = int(self.np_random.integers(0, 2**63 - 1))
        self.backend.reset(seed=sensor_seed, state=state)
        self._action_sequence = 0
        return self._sensor_observation(), self._info()

    def step(self, action):
        try:
            array = np.asarray(action)
            if array.shape != (1,) or array.dtype.kind not in 'fiu' or not np.all(np.isfinite(array)):
                raise ValueError('action must contain one finite real acceleration')
            acceleration = float(array[0])
        except (TypeError, OverflowError) as error:
            raise ValueError('invalid acceleration action') from error
        if abs(acceleration) > self.backend.config.max_acceleration_m_s2:
            raise ValueError('action exceeds acceleration limits')
        observation = self.backend.observation
        interval_ns = self.backend.config.control_steps * self.backend.config.physics_dt_ns
        if interval_ns > self.backend.action_limits.max_ttl_ns:
            raise ValidationError('control interval exceeds maximum action TTL')
        request = ActionRequest(observation.session_id, observation.clock_id, self._action_sequence,
                                observation.sequence, observation.available_ns,
                                observation.available_ns + interval_ns, acceleration)
        self.backend.step(request)
        self._action_sequence += 1
        truth = self.backend.ground_truth
        reward = math.cos(truth.theta) - (truth.x / self.backend.config.rail_half_m)**2 - .001 * acceleration**2
        return self._sensor_observation(), float(reward), self.backend.terminated, self.backend.truncated, self._info()

    def close(self):
        self.backend.close()
