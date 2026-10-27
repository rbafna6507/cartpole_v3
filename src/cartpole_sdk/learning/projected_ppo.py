"""PPO with a zero-action equilibrium projection after each optimizer rollout.

The result is an ordinary feedforward actor. No controller fallback or special
inference wrapper is required. This is an explicit constrained PPO experiment,
not vanilla PPO; projecting between rollouts does not preserve PPO's surrogate
trust-region guarantees.
"""
import numpy as np
import torch
from stable_baselines3 import PPO


def equilibrium_observation():
    """Centered, stationary, upright; valid fresh channels and no rail brake."""
    return dict(state=np.array([0.,0.,0.,1.,0.,0.,0.]),valid=np.ones(7),age_s=np.zeros(7))


class EquilibriumPPO(PPO):
    def project_equilibrium(self):
        observation,_=self.policy.obs_to_tensor(equilibrium_observation())
        with torch.no_grad():
            offset=self.policy.get_distribution(observation).distribution.mean.flatten()
            self.policy.action_net.bias.sub_(offset)
        return float(offset.item())

    def train(self):
        super().train()
        offset=self.project_equilibrium()
        self.logger.record('train/equilibrium_projection_normalized',offset)
        from .diagnostics import preserved_rng
        values=[]
        with preserved_rng(), torch.no_grad():
            for batch in self.rollout_buffer.get(self.batch_size):
                _,log_prob,_=self.policy.evaluate_actions(batch.observations,batch.actions)
                log_ratio=log_prob-batch.old_log_prob
                values.extend((torch.exp(log_ratio)-1-log_ratio).cpu().tolist())
        self.logger.record('train/post_projection_approx_kl',float(np.mean(values)))
