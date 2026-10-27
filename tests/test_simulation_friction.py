"""Independent rod-equation checks for the configured smooth hinge torque."""
import importlib.util
import math
import unittest
from dataclasses import asdict

if importlib.util.find_spec('mujoco') is None:
    raise unittest.SkipTest('optional sim extra not installed')

from cartpole_sdk.contracts import ActionRequest, ValidationError
from cartpole_sdk.sim import MujocoBackend, SimulationConfig


def request(backend, acceleration):
    observation = backend.observation
    return ActionRequest(observation.session_id, observation.clock_id, 0,
                         observation.sequence, observation.available_ns,
                         observation.available_ns + 100_000_000, acceleration)


def friction_config(**overrides):
    # Uniform rod: length .3 m, COM .15 m, pivot inertia .0009 kg*m².
    values = dict(effective_length_m=.2, pole_mass_kg=.03, cart_mass_kg=.4,
                  cart_damping=.07, hinge_damping=.0002,
                  hinge_coulomb_friction_nm=.0003,
                  hinge_friction_velocity_scale_rad_s=.05)
    values.update(overrides)
    return SimulationConfig(**values)


class FrictionConfigTests(unittest.TestCase):
    def test_defaults_disable_dry_friction_and_define_positive_scale(self):
        values = asdict(SimulationConfig())
        self.assertEqual(values.get('hinge_coulomb_friction_nm'), 0.)
        self.assertEqual(values.get('hinge_friction_velocity_scale_rad_s'), .05)

    def test_legacy_positional_config_keeps_force_and_acceleration_arguments(self):
        config = SimulationConfig(1_000_000, 1, 10_000, .26, .124, .35, .02275,
                                  .05, .0001, 12., 18., 500_000_000)
        self.assertEqual(config.max_force_n, 12.)
        self.assertEqual(config.max_acceleration_m_s2, 18.)
        self.assertEqual(config.max_action_ttl_ns, 500_000_000)
        self.assertEqual(config.hinge_coulomb_friction_nm, 0.)

    def test_rejects_nonphysical_or_nonfinite_friction_parameters(self):
        invalid = [-.001, float('nan'), float('inf'), -float('inf'),
                   True, False, '.001', None, 10**1000]
        for name in ('hinge_coulomb_friction_nm', 'hinge_friction_velocity_scale_rad_s'):
            for value in invalid + ([0.] if 'velocity_scale' in name else []):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValidationError):
                        SimulationConfig(**{name: value})


class FrictionPhysicsTests(unittest.TestCase):
    def test_fixed_cart_angular_acceleration_matches_independent_rod_equation(self):
        backend = MujocoBackend(friction_config())
        # Covers signs, the tanh transition, rest and full rotations. Using
        # only saturated velocities would fail to distinguish tanh from sign.
        for theta in (-math.pi + .2, -.3, 0., .4, math.pi - .1):
            for omega in (-1.5, -.05, -.001, 0., .001, .05, 1.5):
                with self.subTest(theta=theta, omega=omega):
                    backend.reset(state=(0., 0., theta, omega))
                    expected_torque = -.0003 * math.tanh(omega / .05)
                    expected_alpha = (9.81 * math.sin(theta) / .2
                                      + (expected_torque - .0002 * omega) / .0009)
                    backend.step(request(backend, 0.))
                    self.assertAlmostEqual(backend.ground_truth.v, 0., places=13)
                    self.assertAlmostEqual((backend.ground_truth.omega - omega) / .001,
                                           expected_alpha, places=10)
                    self.assertAlmostEqual(float(backend.data.qfrc_applied[1]),
                                           expected_torque, places=15)

    def test_nondefault_friction_scale_controls_the_smooth_transition(self):
        backend = MujocoBackend(friction_config(hinge_damping=0.,
                                hinge_coulomb_friction_nm=.0009,
                                hinge_friction_velocity_scale_rad_s=.2))
        backend.reset(state=(0., 0., 0., .1))
        backend.step(None)
        # tau_c/I=1 and omega/scale=.5, so alpha=-tanh(.5).
        self.assertAlmostEqual((backend.ground_truth.omega - .1) / .001,
                               -.46211715726000974, places=12)
        self.assertAlmostEqual(backend.ground_truth.v, 0., places=13)

    def test_cart_force_compensates_hinge_torque_at_signed_and_zero_speed(self):
        backend = MujocoBackend(friction_config())
        for theta, omega, acceleration in ((.3, .02, 2.), (-.4, -.07, -3.),
                                           (math.pi - .2, .4, 0.), (.2, 0., 1.)):
            with self.subTest(theta=theta, omega=omega, acceleration=acceleration):
                velocity = .13
                backend.reset(state=(0., velocity, theta, omega))
                # I*alpha = m*r*(g*sin(theta)-a*cos(theta))-b*omega+tau.
                torque = -.0003 * math.tanh(omega / .05)
                alpha = ((9.81 * math.sin(theta) - acceleration * math.cos(theta)) / .2
                         + (torque - .0002 * omega) / .0009)
                # F = (M+m)*a + m*r*cos(theta)*alpha - m*r*sin(theta)*omega² + d*v.
                force = (.43 * acceleration + .0045 * math.cos(theta) * alpha
                         - .0045 * math.sin(theta) * omega**2 + .07 * velocity)
                backend.step(request(backend, acceleration))
                self.assertFalse(backend.diagnostics.force_saturated)
                self.assertAlmostEqual(backend.diagnostics.achieved_acceleration_m_s2,
                                       acceleration, places=11)
                self.assertAlmostEqual(backend.diagnostics.requested_force_n, force, places=12)
                self.assertAlmostEqual((backend.ground_truth.omega - omega) / .001,
                                       alpha, places=10)

    def test_friction_dissipates_energy_for_both_velocity_signs_and_is_zero_at_rest(self):
        backend = MujocoBackend(friction_config(cart_damping=0., hinge_damping=0.))
        backend.model.opt.gravity[:] = 0.
        for omega in (-1., -.02, 0., .02, 1.):
            with self.subTest(omega=omega):
                backend.reset(state=(0., 0., .4, omega))
                initial_energy = .5 * .0009 * omega**2
                previous_energy = initial_energy
                for _ in range(100):
                    old_omega = backend.ground_truth.omega
                    backend.step(None)
                    torque = float(backend.data.qfrc_applied[1])
                    self.assertLessEqual(torque * old_omega, 0.)
                    energy = .5 * .0009 * backend.ground_truth.omega**2
                    self.assertLessEqual(energy, previous_energy)
                    previous_energy = energy
                    self.assertAlmostEqual(backend.ground_truth.v, 0., places=13)
                if omega:
                    self.assertLess(previous_energy, initial_energy)
                else:
                    self.assertEqual(previous_energy, 0.)
                    self.assertEqual(torque, 0.)

    def test_torque_is_recomputed_each_substep_and_reset_drops_previous_torque(self):
        backend = MujocoBackend(friction_config(control_steps=17))
        for state, acceleration in (((0., .1, .1, .02), 1.5),
                                    ((0., -.08, -.2, -.035), -.5),
                                    ((0., 0., 0., 0.), 0.)):
            with self.subTest(state=state, acceleration=acceleration):
                backend.reset(state=state)
                self.assertEqual(float(backend.data.qfrc_applied[1]), 0.)
                x, velocity, theta, omega = state
                for _ in range(17):
                    torque = -.0003 * math.tanh(omega / .05)
                    alpha = ((9.81 * math.sin(theta) - acceleration * math.cos(theta)) / .2
                             + (torque - .0002 * omega) / .0009)
                    velocity += acceleration * .001
                    omega += alpha * .001
                    x += velocity * .001
                    theta += omega * .001
                backend.step(request(backend, acceleration))
                truth = backend.ground_truth
                for actual, expected in zip((truth.x, truth.v, truth.theta, truth.omega),
                                            (x, velocity, theta, omega)):
                    self.assertAlmostEqual(actual, expected, places=12)
                self.assertEqual(truth.time_ns, 17_000_000)
                self.assertAlmostEqual(float(backend.data.qfrc_applied[1]), torque, places=15)

    def test_zero_friction_is_neutral_for_legacy_dynamics_and_any_positive_scale(self):
        legacy = MujocoBackend()
        explicit_zero = MujocoBackend(SimulationConfig(hinge_coulomb_friction_nm=0.,
                                      hinge_friction_velocity_scale_rad_s=2.))
        state = (0., .1, .3, -.04)
        legacy.reset(state=state)
        explicit_zero.reset(state=state)
        legacy.step(request(legacy, 1.))
        explicit_zero.step(request(explicit_zero, 1.))
        for _ in range(30):
            self.assertEqual(legacy.ground_truth, explicit_zero.ground_truth)
            self.assertEqual(legacy.diagnostics, explicit_zero.diagnostics)
            self.assertEqual(float(explicit_zero.data.qfrc_applied[1]), 0.)
            legacy.step(None)
            explicit_zero.step(None)


if __name__ == '__main__':
    unittest.main()
