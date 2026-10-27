"""Explicit runtime scopes for bounded live rig sessions."""
from contextlib import contextmanager
import gc
import threading


_scope_lock = threading.RLock()
_active_scopes = 0
_restore_enabled = False


@contextmanager
def defer_cyclic_gc():
    """Defer automatic cyclic GC until every overlapping scope has exited.

    GC state is process-wide. A short lock protects a shared scope count, so
    nested calls and cooperating threads cannot restore GC while another scope
    remains active. The first entrant saves the original state; the last exit
    restores it, including after exceptions. Reference counting remains active.

    Callers must bound their session and place this context outside the Rig
    lifetime: ``with defer_cyclic_gc(), Rig(...) as rig: ...``. Rig STOP and
    disconnect then run before GC is restored. This helper never explicitly
    collects, changes thresholds, or modifies leases. Other code must not call
    ``gc.enable()`` or ``gc.collect()`` inside the scope. It does not prevent
    unrelated scheduling pauses or make Python a real-time runtime.
    """
    global _active_scopes, _restore_enabled
    with _scope_lock:
        if _active_scopes == 0:
            _restore_enabled = gc.isenabled()
            gc.disable()
        _active_scopes += 1
    try:
        yield
    finally:
        with _scope_lock:
            _active_scopes -= 1
            if _active_scopes == 0:
                if _restore_enabled:
                    gc.enable()
                else:
                    gc.disable()
