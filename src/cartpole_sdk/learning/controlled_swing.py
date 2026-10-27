"""Opt-in controlled swing objective; legacy training and plant are unchanged.

The extra cost depends only on current position/rate. Rotation history is a
reporting metric, never a hidden reward state. A half-turn from hanging is valid.
Engineering randomization ranges are coverage assumptions, not fitted intervals.
"""
import math
from .ppo import TrainingEnv
from .training_recipe import default_randomization

OBJECTIVE_ID = 'controlled-swing-v1'
OBJECTIVE = dict(id=OBJECTIVE_ID, rail_zone_m=.22, rail_weight=2.,
                 free_angular_speed_rad_s=6., speed_weight=.01,
                 horizon='time-limit truncation, no terminal dwell bonus',
                 faults='legacy terminal penalty retained',
                 rotations='diagnostic only; no spin termination or history reward')


def control_penalty(*, x, omega, rail_half):
    zone = min(.22, .75 * rail_half)
    return 2. * (max(abs(x) - zone, 0.) / (rail_half - zone))**2 + .01 * max(abs(omega) - 6., 0.)**2


class SwingMetrics:
    """Unwrap sampled physical angle; simulator decision increments must be <pi."""
    def __init__(self, theta, rail_zone=.22):
        self.previous = self.unwrapped = self.minimum = self.maximum = theta
        self.travel = self.rail_dwell = 0.
        self.rail_zone = rail_zone

    def update(self, theta, x, dt):
        delta = math.atan2(math.sin(theta-self.previous), math.cos(theta-self.previous))
        self.previous = theta
        self.unwrapped += delta
        self.travel += abs(delta)
        self.minimum, self.maximum = min(self.minimum,self.unwrapped), max(self.maximum,self.unwrapped)
        self.rail_dwell += dt if abs(x) >= self.rail_zone else 0.

    def report(self):
        span = (self.maximum-self.minimum)/(2*math.pi)
        return dict(angular_travel_rotations=self.travel/(2*math.pi),
                    unwrapped_range_rotations=span, full_rotation_excursions=int(span),
                    rail_zone_dwell_s=self.rail_dwell, rail_zone_m=self.rail_zone)


def controlled_randomization():
    result = default_randomization(angle_source='filtered')
    result['encoder'] = dict(encoder_bias_rad=[-math.radians(1),math.radians(1)],
        encoder_sin_error_rad=[-math.radians(4),math.radians(4)],
        encoder_cos_error_rad=[-math.radians(5),math.radians(5)])
    return result


class ControlledSwingEnv(TrainingEnv):
    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.swing_metrics = SwingMetrics(self.core.ground_truth.theta,
                                         min(.22, .75*self.core.config.rail_half_m))
        return observation, info

    def step(self, action):
        before = self.core.ground_truth.time_ns
        observation, reward, terminated, truncated, info = super().step(action)
        truth = self.core.ground_truth
        dt = (truth.time_ns-before)/1e9
        penalty = 0.
        if dt > 0:
            self.swing_metrics.update(truth.theta, truth.x, dt)
            penalty = control_penalty(x=truth.x,omega=truth.omega,rail_half=self.core.config.rail_half_m)
        if (terminated or truncated) and self.fault is None and self.steps == self.core.config.max_steps:
            penalty += self.dwell*100
            terminated, truncated = False, True
        self.total_reward -= penalty
        reward -= penalty*self.reward_scale
        if self._ended:
            info['episode_result'] = self.result()
        return observation, float(reward), terminated, truncated, info

    def result(self, *, hold_seconds=5.):
        return dict(super().result(hold_seconds=hold_seconds), objective_id=OBJECTIVE_ID,
                    **self.swing_metrics.report())
