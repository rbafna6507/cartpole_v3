"""Training-objective ablation: dense reward with a continuing normal horizon.

This jointly removes the finite terminal-dwell bonus at a normal horizon and
allows SB3's time-limit value bootstrap there. It therefore compares a dense
continuing-horizon objective with the existing finite terminal-dwell objective;
it does not isolate either change by itself.

All physical transitions, observations, actions, leases, randomization, and
fault endings/penalties remain those of RobustSwingEnv. In particular the
existing remaining-time fault penalty is unchanged and can still be partially
observed: this ablation does not establish a fully observed continuing MDP.
Episode score uses the modified reward, while balance/hold metrics are unchanged.
"""
from .robust_swing import RobustSwingEnv


class ContinuingSwingEnv(RobustSwingEnv):
    """Remove normal-horizon dwell payoff and expose a time-limit truncation."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if ((terminated or truncated) and self.fault is None
                and self.steps == self.core.config.max_steps):
            bonus = self.dwell * 100
            self.total_reward -= bonus
            reward -= bonus * self.reward_scale
            info['episode_result'] = self.result()
            terminated, truncated = False, True
        return observation, float(reward), terminated, truncated, info
