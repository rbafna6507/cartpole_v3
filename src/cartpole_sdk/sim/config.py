"""Explicit provisional plant and simulated sensor assumptions (SI units)."""
from dataclasses import dataclass
import math
from cartpole_sdk.contracts import ValidationError


def finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValidationError(f'{name} must be finite and not bool')


def number(value, name, *, positive=False):
    finite(value, name)
    if value < 0 or (positive and value == 0):
        raise ValidationError(f'{name} must be {"positive" if positive else "nonnegative"}')


def integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValidationError(f'{name} must be a {"positive" if positive else "nonnegative"} integer')


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Unvalidated surrogate: saved effective length/rail, assumed dynamics."""
    physics_dt_ns: int = 1_000_000
    control_steps: int = 1
    max_steps: int = 10_000
    rail_half_m: float = .26
    effective_length_m: float = .124
    cart_mass_kg: float = .35
    pole_mass_kg: float = .02275
    cart_damping: float = .05
    hinge_damping: float = .0001
    max_force_n: float = 20.
    max_acceleration_m_s2: float = 25.
    max_action_ttl_ns: int = 1_000_000_000
    hinge_coulomb_friction_nm: float = 0.
    hinge_friction_velocity_scale_rad_s: float = .05

    def __post_init__(self):
        for name in ('physics_dt_ns', 'control_steps', 'max_steps', 'max_action_ttl_ns'):
            integer(getattr(self, name), name, positive=True)
        for name in ('rail_half_m', 'effective_length_m', 'cart_mass_kg', 'pole_mass_kg',
                     'max_force_n', 'max_acceleration_m_s2',
                     'hinge_friction_velocity_scale_rad_s'):
            number(getattr(self, name), name, positive=True)
        for name in ('cart_damping', 'hinge_damping', 'hinge_coulomb_friction_nm'):
            number(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Acquisition and transport in physics ticks; independent seeded errors."""
    sample_every_steps: int = 1
    delay_steps: int = 0
    position_noise_std: float = 0.
    velocity_noise_std: float = 0.
    angle_noise_std: float = 0.
    angular_velocity_noise_std: float = 0.
    dropout_probability: float = 0.
    frame_id: str = 'sim:upright+x'

    def __post_init__(self):
        integer(self.sample_every_steps, 'sample_every_steps', positive=True)
        integer(self.delay_steps, 'delay_steps')
        for name in ('position_noise_std', 'velocity_noise_std', 'angle_noise_std',
                     'angular_velocity_noise_std', 'dropout_probability'):
            number(getattr(self, name), name)
        if self.dropout_probability > 1:
            raise ValidationError('dropout_probability must be at most one')
        if not isinstance(self.frame_id, str) or not self.frame_id or self.frame_id.strip() != self.frame_id:
            raise ValidationError('frame_id must be nonempty without outer whitespace')
