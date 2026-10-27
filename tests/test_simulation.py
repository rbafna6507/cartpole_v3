"""Actual plant, SDK boundary and seeded sensor regression checks."""
import importlib.util
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace

if importlib.util.find_spec('mujoco') is None:
    raise unittest.SkipTest('optional sim extra not installed')
from cartpole_sdk.contracts import ActionRequest, ValidationError
from cartpole_sdk.sim import MujocoBackend, SimulationConfig, SensorConfig


def request(backend, acceleration=0., *, ttl=1_000_000, sequence=0):
    obs = backend.observation
    return ActionRequest(obs.session_id, obs.clock_id, sequence, obs.sequence,
                         obs.available_ns, obs.available_ns + ttl, acceleration)


class PhysicsTests(unittest.TestCase):
    def test_acceleration_servo_and_free_hinge_match_uniform_rod_equation(self):
        b = MujocoBackend(SimulationConfig(cart_damping=0., hinge_damping=0.))
        for theta, acceleration in [(0., 2.), (.2, 0.), (-.2, 0.), (math.pi, 2.)]:
            b.reset(state=(0., 0., theta, 0.))
            b.step(request(b, acceleration))
            self.assertAlmostEqual(b.diagnostics.achieved_acceleration_m_s2, acceleration, places=10)
            if theta == 0.:
                self.assertAlmostEqual(b.diagnostics.requested_force_n, .711375, places=10)
            expected = (9.81 * math.sin(theta) - acceleration * math.cos(theta)) / .124
            self.assertAlmostEqual(b.ground_truth.omega / .001, expected, places=9)
        self.assertEqual(b.model.nu, 0)  # cart generalized force; no hinge actuator
        self.assertEqual(float(b.data.qfrc_applied[1]), 0.)

    def test_force_saturation_changes_real_acceleration(self):
        b = MujocoBackend(SimulationConfig(max_force_n=.01, cart_damping=0., hinge_damping=0.))
        b.reset(state=(0., 0., 0., 0.))
        b.step(request(b, 25.))
        d = b.diagnostics
        self.assertGreater(d.requested_force_n, .01)
        self.assertEqual(d.applied_force_n, .01)
        self.assertTrue(d.force_saturated)
        self.assertGreater(d.achieved_acceleration_m_s2, 0.)
        self.assertLess(d.achieved_acceleration_m_s2, 1.)
        self.assertAlmostEqual(b.ground_truth.v / .001, d.achieved_acceleration_m_s2, places=10)

    def test_passive_damping_is_compensated_with_cart_force_only(self):
        b = MujocoBackend()
        b.reset(state=(0., .1, .3, .4))
        b.step(request(b, 2.))
        self.assertAlmostEqual(b.diagnostics.achieved_acceleration_m_s2, 2., places=10)
        self.assertEqual(float(b.data.qfrc_applied[1]), 0.)


class LeaseTests(unittest.TestCase):
    def test_none_holds_lease_and_exact_substep_expiry_removes_demand(self):
        b = MujocoBackend(SimulationConfig(control_steps=3))
        b.reset(state=(0., 0., 0., 0.))
        b.step(request(b, 2., ttl=2_000_000))
        self.assertAlmostEqual(b.ground_truth.v, .004, places=10)
        self.assertEqual(b.diagnostics.requested_acceleration_m_s2, 0.)
        self.assertIsNone(b.diagnostics.active_request_sequence)
        b.reset(state=(0., 0., 0., 0.))
        b.step(request(b, 2., ttl=9_000_000))
        b.step(None)
        self.assertAlmostEqual(b.ground_truth.v, .012, places=10)
        self.assertEqual(b.diagnostics.active_request_sequence, 0)

    def test_bad_requests_leave_episode_and_previous_lease_unchanged(self):
        b = MujocoBackend()
        b.reset()
        good = request(b, 1., ttl=100_000_000)
        bads = [replace(good, session_id='wrong'), replace(good, clock_id='wrong'),
                replace(good, observation_sequence=1), replace(good, issued_at_ns=1),
                replace(good, expires_at_ns=1_000_000_001), replace(good, acceleration_m_s2=26.)]
        for bad in bads:
            with self.assertRaises(ValidationError):
                b.step(bad)
            self.assertEqual(b.ground_truth.time_ns, 0)
            self.assertEqual(b.observation.sequence, 0)
        b.step(good)
        with self.assertRaises(ValidationError):
            b.step(request(b, sequence=0))
        b.step(None)
        self.assertEqual(b.diagnostics.active_request_sequence, 0)

    def test_old_issue_time_is_rejected_before_mutation(self):
        b = MujocoBackend()
        b.reset()
        b.step(None)
        bad = replace(request(b), issued_at_ns=0)
        with self.assertRaises(ValidationError):
            b.step(bad)
        self.assertEqual(b.ground_truth.time_ns, 1_000_000)


class SensorTests(unittest.TestCase):
    def test_delay_warmup_sample_repeat_and_receipt_are_distinct(self):
        b = MujocoBackend(sensors=SensorConfig(sample_every_steps=2, delay_steps=1))
        first = b.reset()
        self.assertTrue(all(s.validity == 'unknown' and s.value is None for s in first.samples))
        one = b.step(None)
        self.assertEqual(one.samples[0].effective_ns, 0)
        self.assertEqual(one.samples[0].received_ns, 1_000_000)
        two = b.step(None)
        self.assertEqual(two.samples[0].effective_ns, 0)
        self.assertEqual(two.samples[0].received_ns, 1_000_000)
        three = b.step(None)
        self.assertEqual(three.samples[0].effective_ns, 2_000_000)
        self.assertEqual(three.samples[0].received_ns, 3_000_000)
        self.assertEqual(three.available_ns, 3_000_000)
        self.assertTrue(all(s.source == 'simulation' for s in three.samples))

    def test_noise_and_dropout_are_seeded_and_never_truth_fallback(self):
        sensors = SensorConfig(angle_noise_std=.1, dropout_probability=.5)
        b = MujocoBackend(sensors=sensors)
        def rollout(seed):
            return [b.reset(seed=seed), *[b.step(None) for _ in range(20)]]
        first = rollout(42)
        self.assertEqual(first, rollout(42))
        self.assertNotEqual(first, rollout(43))
        self.assertTrue(any(s.validity == 'invalid' and s.value is None for o in first for s in o.samples))
        b = MujocoBackend(sensors=SensorConfig(dropout_probability=1.))
        self.assertTrue(all(s.validity == 'invalid' for s in b.reset().samples))

    def test_full_dropout_replaces_previously_valid_sample(self):
        b = MujocoBackend(sensors=SensorConfig(dropout_probability=.5))
        obs = b.reset(seed=0)
        history = [obs, *[b.step(None) for _ in range(20)]]
        states = [o.samples[0].validity for o in history]
        self.assertIn('valid', states)
        self.assertIn('invalid', states)
        self.assertTrue(all(s.value is None for o in history for s in o.samples if s.validity == 'invalid'))


class EpisodeTests(unittest.TestCase):
    def test_horizon_and_rail_end_reject_further_steps_and_reset_clears(self):
        b = MujocoBackend(SimulationConfig(max_steps=2, control_steps=3))
        b.reset()
        b.step(None)
        b.step(None)
        self.assertTrue(b.truncated)
        self.assertFalse(b.terminated)
        with self.assertRaises(RuntimeError): b.step(None)
        b.reset(state=(.259, 2., math.pi, 0.))
        b.step(None)
        self.assertTrue(b.terminated)
        self.assertEqual(b.ground_truth.time_ns, 1_000_000)
        with self.assertRaises(RuntimeError): b.step(None)
        b.reset()
        self.assertFalse(b.terminated or b.truncated)
        self.assertEqual(b.observation.sequence, 0)

    def test_numerical_failure_never_silently_resets_or_continues_plant(self):
        # MuJoCo intentionally logs numerical warnings. Isolate its log file and
        # capture diagnostic output without altering its global warning callback.
        code = """
from cartpole_sdk.sim import MujocoBackend
b = MujocoBackend()
b.reset(state=(0., 1e100, 0., 0.))
try:
    b.step(None)
except RuntimeError as error:
    assert 'numerical' in str(error)
else:
    raise AssertionError('unstable plant continued without error')
assert abs(b.ground_truth.v) > 1e90, 'plant silently reset'
try:
    b.step(None)
except RuntimeError as error:
    assert 'reset' in str(error)
else:
    raise AssertionError('numerical failure did not latch')
b.reset()
b.step(None)
assert b.ground_truth.time_ns == 1_000_000
"""
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                     capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_huge_integer_config_is_validation_error(self):
        with self.assertRaises(ValidationError):
            SimulationConfig(max_force_n=10**1000)

    def test_config_and_reset_validate_before_replacing_state(self):
        for kw in [{'physics_dt_ns':True}, {'control_steps':0}, {'max_steps':-1},
                   {'cart_mass_kg':0}, {'max_force_n':float('nan')}, {'cart_damping':-.1}]:
            with self.assertRaises(ValidationError): SimulationConfig(**kw)
        for kw in [{'delay_steps':True}, {'dropout_probability':1.1}, {'angle_noise_std':-.1}, {'frame_id':' '}]:
            with self.assertRaises(ValidationError): SensorConfig(**kw)
        b = MujocoBackend()
        b.reset()
        before = b.observation
        for state in [(0., 0., float('nan'), 0.), (0., 0., 0.), (.3, 0., 0., 0.), (0., 10**1000, 0., 0.)]:
            with self.assertRaises(ValidationError): b.reset(state=state)
            self.assertEqual(b.observation, before)
        with self.assertRaises(ValidationError): b.reset(seed=True)


if __name__ == '__main__': unittest.main()
