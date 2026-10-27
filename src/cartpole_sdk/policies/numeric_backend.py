"""Opt-in, prepacked float64 inference for the fixed numeric actor.

This module does not load artifacts or admit hardware. The caller must retain
the ordinary actor loader's provenance checks and all policy/runner guards.
NumPy is imported only when this backend is explicitly constructed.
"""
from hashlib import sha256
import math
from pathlib import Path

from cartpole_sdk.contracts import ValidationError
from .neural import validate_actor


class NumpyActorBackend:
    """A validated actor snapshot exposing ``predict_encoded`` as a delegate.

    Unoptimized einsum uses NumPy's direct contraction, avoiding BLAS worker
    pools for these tiny matrix-vector products. No process-wide thread or
    floating-point settings are changed. Each call owns its activation arrays.
    """

    def __init__(self, actor):
        validate_actor(actor)
        try:
            import numpy as np
        except ImportError as error:
            raise ValidationError('NumPy inference backend requires NumPy to be installed') from error
        self._np = np
        self._feature_order = tuple(actor['feature_order'])
        self._scale = float(actor['action_scale_m_s2'])
        layers = []
        for layer in actor['layers']:
            weights = np.array(layer['weights'], dtype=np.float64, order='C', copy=True)
            bias = np.array(layer['bias'], dtype=np.float64, order='C', copy=True)
            weights.setflags(write=False)
            bias.setflags(write=False)
            layers.append((weights, bias, layer['activation'] == 'tanh'))
        self._layers = tuple(layers)
        self._provenance = dict(backend='numpy-einsum', dtype='float64',
            contraction='einsum-optimize-false', numpy_version=np.__version__,
            actor_content_sha256=actor['content_sha256'],
            source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())

    def provenance(self):
        """Return JSON-safe metadata captured before the action loop starts."""
        return dict(self._provenance)

    def predict_encoded(self, encoded):
        """Preserve scalar feature order, finite checks, tanh and action clip."""
        values = [float(value) for key in self._feature_order for value in encoded[key]]
        if len(values) != 21 or not all(math.isfinite(value) for value in values):
            raise ValidationError('invalid actor features')
        np = self._np
        values = np.asarray(values, dtype=np.float64)
        # Reject overflow before tanh can turn infinity into a finite value;
        # suppress only NumPy's warning, never the explicit finite check.
        with np.errstate(over='ignore', invalid='ignore'):
            for weights, bias, tanh in self._layers:
                values = np.einsum('ij,j->i', weights, values, optimize=False) + bias
                if not np.isfinite(values).all():
                    raise ValidationError('nonfinite actor inference')
                if tanh:
                    np.tanh(values, out=values)
        return max(-1., min(1., float(values[0]))) * self._scale
