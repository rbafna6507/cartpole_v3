import importlib.util
import unittest
import warnings
if importlib.util.find_spec('gymnasium') is None or importlib.util.find_spec('mujoco') is None:
    raise unittest.SkipTest('optional sim extra not installed')
import numpy as np
from gymnasium.utils.env_checker import check_env
from cartpole_sdk.sim import SimulationConfig, SensorConfig
from cartpole_sdk.sim.gym_env import CartpoleEnv


class GymTests(unittest.TestCase):
    def test_gym_checker_and_seeded_sensor_api(self):
        env = CartpoleEnv(sensors=SensorConfig(angle_noise_std=.01, dropout_probability=.2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            check_env(env, skip_render_check=True)
        a, info = env.reset(seed=42)
        b, _, _, _, second_info = env.step(np.array([1.], dtype=np.float32))
        again, _ = env.reset(seed=42)
        repeated, _, _, _, _ = env.step(np.array([1.], dtype=np.float32))
        for key in a: np.testing.assert_array_equal(a[key], again[key])
        for key in b: np.testing.assert_array_equal(b[key], repeated[key])
        self.assertTrue(env.observation_space.contains(b))
        self.assertIn('ground_truth', info)
        self.assertIn('diagnostics', second_info)
        self.assertIsNot(a['state'], b['state'])
        env.close()
        env.close()

    def test_unknown_sensor_state_mask_age_and_copy(self):
        env = CartpoleEnv(sensors=SensorConfig(delay_steps=20))
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs['valid'], [0]*5)
        np.testing.assert_array_equal(obs['age_s'], [-1.]*5)
        np.testing.assert_array_equal(obs['state'], [0.]*5)
        saved = obs['state'].copy()
        env.step(np.array([0.], dtype=np.float32))
        np.testing.assert_array_equal(obs['state'], saved)

    def test_invalid_actions_reject_before_time_moves_and_episode_flags(self):
        env = CartpoleEnv(SimulationConfig(max_steps=1))
        env.reset(seed=2)
        for action in [0., [0., 0.], [True], [float('nan')], [26.], ['1']]:
            with self.assertRaises(ValueError): env.step(action)
            self.assertEqual(env.backend.ground_truth.time_ns, 0)
        _, reward, terminated, truncated, _ = env.step(np.array([0.], dtype=np.float32))
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertTrue(np.isfinite(reward))
        with self.assertRaises(RuntimeError): env.step(np.array([0.], dtype=np.float32))
        env.reset(seed=2, options={'state':(.259, 2., 3.14, 0.)})
        _, _, terminated, _, _ = env.step(np.array([0.], dtype=np.float32))
        self.assertTrue(terminated)
        with self.assertRaises(ValueError): CartpoleEnv(render_mode='rgb_array')


if __name__ == '__main__': unittest.main()
