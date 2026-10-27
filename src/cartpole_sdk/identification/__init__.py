"""Offline identification. Optional numerical dependencies load only when used."""
from importlib import import_module

_EXPORTS = {
    'FitConfig': '.fit', 'FitReport': '.fit', 'identify_pendulum': '.fit',
    'read_rows': '.adapters', 'sdk_run_rows': '.adapters', 'rig_log_rows': '.rig_logs',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
