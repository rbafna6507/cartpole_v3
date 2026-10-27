"""Trusted source-derived execution budgets shared by queue and worker.

Submitted duration is a simulation horizon or physical experiment duration.
Simulation gets twice that duration in wall time; physical duration is unchanged.
There is no submitted wall budget field.
"""
import math

def execution_wall_seconds(sensor_source, duration):
    if sensor_source not in ('simulation','magnetic-step'):
        raise ValueError('unsupported source for execution budget')
    if type(duration) not in (int,float) or not .05<=duration<=60 or not math.isfinite(duration):
        raise ValueError('duration must be finite and within 0.05..60 seconds')
    return float(duration)*(2 if sensor_source=='simulation' else 1)
