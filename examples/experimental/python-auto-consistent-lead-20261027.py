import gc
gc.disable()
import math
"""Frozen Python AUTO baseline. Explicit native snapshot, no hardware writes."""
from cartpole_sdk.control.native_controller import NativeController
from cartpole_sdk.contracts import Channel, ValidationError
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.policies.executor_feedback import ExecutorFeedbackPolicy
PARAMETERS = {'leff': 0.123999998, 'pw': 3.0, 'pz': 0.850000024, 'bal_pw': 4.5, 'pc1': -5.0, 'pc2': -6.0, 'vmax': 1.5, 'vmax_s': 1.5, 'vmax_b': 1.5, 'approach_v': 0.200000003, 'catch_v': 0.300000012, 'approach_angle': 0.800000012, 'catch_da': 4.0, 'amax_s': 25.0, 'amax_b': 25.0, 'jmax': 150.0, 'amax_m': 0.5, 'jmax_m': 10.0, 'ke': 4.0, 'kpx': 60.0, 'kdx': 2.0, 'phase_soft': 2.0, 'catch_a': 0.25, 'catch_r': 3.0, 'giveup': 0.800000012, 'spin_trip_rad_s': 150.0, 'spin_resume_rad_s': 10.0, 'rail': 0.300112516, 'bw': 10.0, 'vman': 0.0500000007, 'manual_speed_max': 0.150000006, 'rail_fraction': 0.949999988, 'rail_margin': 0.0149999997, 'rail_fault_margin': 0.00999999978, 'rail_hysteresis': 0.00499999989, 'rail_recover_v': 0.150000006, 'rail_recover_a': 1.5, 'rail_recover_s': 8.0, 'rehome_v': 0.0799999982, 'rehome_a': 0.5, 'rehome_j': 10.0}
RATE_HZ = 250
MAX_GAP_PERIODS = 3

class PortalPythonAuto(ExecutorFeedbackPolicy):
    def _read(self, observation):
        x,v,theta,omega,a,brake,raw_theta = super()._read(observation)
        weight = max(0.,min(1.,(.8-abs(theta))/.4))
        innovation = math.atan2(math.sin(raw_theta-theta),math.cos(raw_theta-theta))
        omega += weight*2*math.pi*self.parameters.bw*innovation
        theta = math.atan2(math.sin(theta+weight*innovation),math.cos(theta+weight*innovation))
        if abs(omega)>self.parameters.spin_trip_rad_s:
            raise ValidationError('corrected rate exceeds spin guard')
        return x,v,theta,omega,a,brake,raw_theta

    def act(self, observation):
        action = super().act(observation)
        if action is None and self.fault_reason:
            raise ValidationError('host policy fault: ' + self.fault_reason)
        return action

def create_policy():
    channels = (Channel(entity='cart', quantity='position', source='step', frame_id='cart-step-home'), Channel(entity='cart', quantity='velocity', source='step', frame_id='cart-step-home'), Channel(entity='pole', quantity='angle', source='magnetic_filtered', frame_id='encoder-zero'), Channel(entity='pole', quantity='angular_velocity', source='magnetic', frame_id='encoder-zero'), Channel(entity='cart', quantity='acceleration', source='step', frame_id='cart-step-home'), Channel(entity='cart', quantity='brake_direction', source='step', frame_id='cart-step-home'))
    return PortalPythonAuto(channels, parameters=ControllerParameters(**PARAMETERS),
        controller_math=NativeController(ControllerParameters(**PARAMETERS)),
        rate_hz=RATE_HZ, start_mode='auto', max_gap_periods=MAX_GAP_PERIODS,
        raw_angle_channel=Channel('pole','angle','magnetic','encoder-zero'))
