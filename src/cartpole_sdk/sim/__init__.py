"""Simulation configuration with lazily loaded optional MuJoCo backend."""
from .config import SimulationConfig, SensorConfig

__all__ = ['SimulationConfig', 'SensorConfig', 'MujocoBackend', 'GroundTruth', 'SimulationDiagnostics']


def __getattr__(name):
    # Importing a dependency-free bridge/profile or hardware portal client must
    # not require the sim extra. Preserve the public backend exports on demand.
    if name in ('MujocoBackend', 'GroundTruth', 'SimulationDiagnostics'):
        from importlib import import_module
        value = getattr(import_module('.backend', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
