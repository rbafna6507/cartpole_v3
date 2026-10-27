"""Experimental PPO with a fixed-anchor actor retention step.

After EquilibriumPPO's update, a separate Adam optimizer fits the Gaussian actor
mean to pinned, caller-supplied initial-actor outputs, then projects equilibrium
again. This is an explicit proximal supervised intervention, not the original
PPO objective or a trust-region guarantee. Neither improved return nor improved
anchor error after equilibrium projection is guaranteed.

Callers supply observed training/demonstration states and frozen actor means in
normalized action coordinates, without clipping. Targets may exceed [-1, 1].
The module never collects data or changes environment/randomization settings.
Multiply reported normalized RMSE by the experiment's action scale (e.g. 25)
to obtain raw acceleration disagreement.

Use RetainedPPO.load for configured checkpoints: anchors, private sampling RNG,
and separate Adam state are preserved. Plain PPO.load cannot restore the extra
optimizer. SB3's usual environment/global-RNG resume limitations still apply.
"""
from hashlib import sha256
import json
import math

from gymnasium import spaces
import numpy as np
import torch

from .projected_ppo import EquilibriumPPO


class RetainedPPO(EquilibriumPPO):
    """Add actor-only fixed-anchor regression between PPO rollouts."""

    def __init__(self, *args, **kwargs):
        self._retention_config = None
        self._retention_observations = None
        self._retention_targets = None
        self._retention_rng = None
        self._retention_optimizer = None
        self.retention_diagnostics = {}
        super().__init__(*args, **kwargs)

    def _actor_parameters(self):
        # Shared feature extractors, the value network, and log_std stay fixed.
        return (list(self.policy.mlp_extractor.policy_net.parameters())
                + list(self.policy.action_net.parameters()))

    def _setup_model(self):
        super()._setup_model()
        # SB3 restores pickled data before _setup_model, then optimizer state.
        if self._retention_config is not None:
            self._retention_optimizer = torch.optim.Adam(
                self._actor_parameters(), lr=self._retention_config['learning_rate'])

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        if self._retention_optimizer is not None:
            state_dicts = state_dicts + ['_retention_optimizer']
        return state_dicts, variables

    def configure_retention(self, observations, targets, *, updates=4,
                            batch_size=256, learning_rate=1e-4, seed=71002,
                            weight=1.):
        """Copy and pin anchors; reset retention Adam/RNG when reconfigured.

        Observations must match the policy's dict space, with N rows per key.
        Targets are finite (N, 1) Gaussian means from the frozen initial actor.
        ``updates`` counts minibatches sampled with replacement, independently
        of rollout sampling. Zero updates or zero weight disables the hook.
        Weight multiplies MSE; Adam's normalization largely cancels its scale,
        so learning_rate and updates are the main intervention controls.
        """
        for name, value, minimum in (('updates', updates, 0), ('batch_size', batch_size, 1),
                                     ('seed', seed, 0)):
            if type(value) is not int or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name, value, positive in (('learning_rate', learning_rate, True), ('weight', weight, False)):
            if (isinstance(value, (bool, str)) or not np.isscalar(value)
                    or not math.isfinite(value) or value < 0 or (positive and value == 0)):
                raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
        if (not isinstance(self.observation_space, spaces.Dict)
                or not isinstance(self.action_space, spaces.Box) or self.action_space.shape != (1,)):
            raise ValueError('retention requires dict observations and a scalar Box action')
        expected = self.observation_space.spaces
        if not isinstance(observations, dict) or set(observations) != set(expected):
            raise ValueError('anchor observation keys must exactly match the policy observation space')
        arrays = {}
        count = None
        for key in sorted(expected):
            value = np.array(observations[key], dtype=np.float32, copy=True, order='C')
            if value.ndim != len(expected[key].shape) + 1 or value.shape[1:] != expected[key].shape:
                raise ValueError(f'anchor {key} must have shape (N, {expected[key].shape})')
            if count is None:
                count = len(value)
            if len(value) != count or count == 0 or not np.isfinite(value).all():
                raise ValueError('anchor observations require equal positive row counts and finite values')
            arrays[key] = value
        target_array = np.array(targets, dtype=np.float32, copy=True, order='C')
        if target_array.shape != (count, 1) or not np.isfinite(target_array).all():
            raise ValueError('anchor targets must be a finite (N, 1) array')
        fingerprint = sha256()
        for name, value in [*arrays.items(), ('targets', target_array)]:
            fingerprint.update(json.dumps([name, value.shape], separators=(',', ':')).encode())
            fingerprint.update(value.astype('<f4', copy=False).tobytes())
        config = dict(updates=updates, batch_size=batch_size, learning_rate=float(learning_rate),
                      seed=seed, weight=float(weight), samples=count,
                      anchor_sha256=fingerprint.hexdigest())
        optimizer = torch.optim.Adam(self._actor_parameters(), lr=learning_rate)
        rng = np.random.default_rng(seed)
        # Invalid inputs above leave an existing configuration intact.
        self._retention_config = config
        self._retention_observations = arrays
        self._retention_targets = target_array
        self._retention_optimizer = optimizer
        self._retention_rng = rng
        self.retention_diagnostics = {}
        return self.retention_record()

    def retention_record(self):
        """Return JSON-safe settings and a digest of the exact float32 anchors."""
        return dict(self._retention_config) if self._retention_config is not None else None

    def _anchor_loss(self, indices):
        obs = {k: torch.as_tensor(v[indices], device=self.device)
               for k, v in self._retention_observations.items()}
        targets = torch.as_tensor(self._retention_targets[indices], device=self.device)
        mean = self.policy.get_distribution(obs).distribution.mean
        return (mean - targets).square().mean()

    def _anchor_mse(self):
        total = 0.
        count = self._retention_config['samples']
        batch_size = self._retention_config['batch_size']
        with torch.no_grad():
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                total += float(self._anchor_loss(slice(start, end)).item()) * (end - start)
        return total / count

    def retain_actor(self):
        """Apply configured updates and projection; return normalized metrics.

        This method is also usable as a standalone offline optimizer probe.
        All anchors are measured before fitting, after fitting, and after final
        projection. Only the private generator is advanced; ordinary SB3
        feedforward policies are evaluated without dropout or buffer updates.
        """
        config = self._retention_config
        if config is None or config['updates'] == 0 or config['weight'] == 0:
            return {}
        modes = [(module, module.training) for module in self.policy.modules()]
        self.policy.set_training_mode(False)
        try:
            initial = self._anchor_mse()
            losses = []
            parameters = self._actor_parameters()
            for _ in range(config['updates']):
                indices = self._retention_rng.integers(
                    config['samples'], size=min(config['batch_size'], config['samples']))
                loss = config['weight'] * self._anchor_loss(indices)
                self._retention_optimizer.zero_grad(set_to_none=True)
                # autograd.grad leaves critic/extractor/std gradients untouched.
                gradients = torch.autograd.grad(loss, parameters)
                for parameter, gradient in zip(parameters, gradients):
                    parameter.grad = gradient
                self._retention_optimizer.step()
                losses.append(float(loss.detach().item()))
            before_projection = self._anchor_mse()
            offset = self.project_equilibrium()
            final = self._anchor_mse()
        finally:
            for module, training in modes:
                module.training = training
        metrics = dict(mse_initial=initial, mse_before_projection=before_projection,
                       mse_final=final, rmse_initial=math.sqrt(initial),
                       rmse_final=math.sqrt(final), update_loss_normalized=float(np.mean(losses)),
                       equilibrium_projection_normalized=offset)
        self.retention_diagnostics = metrics
        if hasattr(self, '_logger'):
            for name, value in metrics.items():
                self.logger.record(f'train/retention_{name}', value)
        return dict(metrics)

    def train(self):
        super().train()
        self.retain_actor()
