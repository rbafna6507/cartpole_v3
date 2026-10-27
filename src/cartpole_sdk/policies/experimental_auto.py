"""October 27 candidate; not a qualified replacement for native AUTO."""
import math
from cartpole_sdk.contracts import ValidationError
from .executor_feedback import ExecutorFeedbackPolicy

class ExperimentalAutoPolicy(ExecutorFeedbackPolicy):
    def _read(self, observation):
        x,v,theta,omega,a,brake,raw_theta=super()._read(observation)
        weight=max(0.,min(1.,(.8-abs(theta))/.4))
        innovation=math.atan2(math.sin(raw_theta-theta),math.cos(raw_theta-theta))
        omega+=weight*2*math.pi*self.parameters.bw*innovation
        theta=math.atan2(math.sin(theta+weight*innovation),math.cos(theta+weight*innovation))
        if abs(omega)>self.parameters.spin_trip_rad_s:
            raise ValidationError('corrected rate exceeds spin guard')
        return x,v,theta,omega,a,brake,raw_theta

from cartpole_sdk.contracts import Channel
from cartpole_sdk.rig.policy import run_policy

def run_trial(rig, *, mode, duration_s=10., max_gap_periods=None, experimental=True):
    """Saved encoder-only AUTO trial using stopped live readback.

    The MCU owns observer, pulse generation and the single 1 kHz motion governor.
    Python owns energy shaping, capture, balance and rail-return decisions at the
    telemetry cadence. No calibration, parameter write or rehome is implicit.
    Optional max_gap_periods=2/3 permits bounded 250 Hz observation gaps; the
    default remains strict. All configurations retain the 20 ms anchored lease.
    """
    from dataclasses import fields
    from cartpole_sdk.control.parameters import ControllerParameters
    from cartpole_sdk.policies.executor_feedback import ExecutorFeedbackPolicy
    if mode not in ('auto','bal') or rig.rate_hz not in (100,250):
        raise ValidationError('host controller requires auto/bal and 100 or 250 Hz telemetry')
    rig._duration(duration_s)
    rig._stopped_healthy()
    values=rig.get_parameters()
    if set(values)!={f.name for f in fields(ControllerParameters)}:
        raise ValidationError('host controller requires the complete v6 live parameter catalog')
    if experimental:
        if mode!='auto' or rig.rate_hz!=250:
            raise ValidationError('python-auto-trial requires AUTO at 250 Hz')
        values=dict(values,kpx=60.)
        max_gap_periods=3
    parameters=ControllerParameters(**values)
    channels=(Channel('cart','position','step','cart-step-home'),
              Channel('cart','velocity','step','cart-step-home'),
              Channel('pole','angle','magnetic_filtered','encoder-zero'),
              Channel('pole','angular_velocity','magnetic','encoder-zero'),
              Channel('cart','acceleration','step','cart-step-home'),
              Channel('cart','brake_direction','step','cart-step-home'))
    rig._record('host_controller',mode=mode,rate_hz=rig.rate_hz,max_gap_periods=max_gap_periods,
                controller_parameters=values,policy='ExperimentalAutoPolicy',
                raw_safety_channel=dict(entity='pole',quantity='angle',source='magnetic',frame_id='encoder-zero'),
                hardware_validated=False)
    controller_math=None
    policy_type=ExecutorFeedbackPolicy
    if experimental:
        from cartpole_sdk.policies.experimental_auto import ExperimentalAutoPolicy
        from cartpole_sdk.control.native_controller import NativeController
        policy_type=ExperimentalAutoPolicy
        controller_math=NativeController(parameters)
        rig._record('experimental_profile',name='consistent-lead-20261027',qualified=False)
    from cartpole_sdk.rig.runtime import defer_cyclic_gc
    try:
        with defer_cyclic_gc():
            run_policy(rig,lambda selected:policy_type(selected,rate_hz=rig.rate_hz,
                   parameters=parameters,start_mode=mode,max_gap_periods=max_gap_periods,
                   controller_math=controller_math,
                   raw_angle_channel=Channel('pole','angle','magnetic','encoder-zero')),channels=channels,
                   observation_provider=rig.control_observation,duration_s=duration_s,
                   ttl_ms=20,max_acceleration=max(parameters.amax_s,parameters.amax_b))
    finally:
        if controller_math is not None:controller_math.close()
