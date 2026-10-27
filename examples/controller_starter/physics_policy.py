"""Offline physics controller using actual firmware swing/capture/balance equations.

Build the native binding first; this file never connects to a rig. The same
explicit observation adapter accepts magnetic-step samples, but host scheduling
and physical transfer are experimental and require separate validation.
"""
from cartpole_sdk import Acceleration, ValidationError
from cartpole_sdk.developer import ObservationAdapter
from cartpole_sdk.control.native_controller import NativeController
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.sim.profiles import load_profile


class Controller:
    def __init__(self):
        self.adapter = ObservationAdapter(angle_source='filtered')
        self.parameters = ControllerParameters(**load_profile()['experiment']['executor_parameters'])
        self.native = NativeController(self.parameters)

    def reset(self, session):
        self.adapter.reset(session)
        self.native.reset()
        self.previous = None
        self.balancing = False
        self.fault_reason = None

    def act(self, observation):
        if self.fault_reason is not None:
            return None
        try:
            x, velocity, angle, rate, acceleration, brake = self.adapter.select(observation)
            period = .01 if self.adapter.source == 'simulation' else .004
            dt = period
            if self.previous is not None:
                sequence, timestamp = self.previous
                dt = (observation.available_ns - timestamp) / 1e9
                if observation.sequence <= sequence or not .5 * period <= dt <= 1.5 * period:
                    raise ValidationError('expected fresh observations at the selected source cadence')
            self.previous = observation.sequence, observation.available_ns
            demand, self.balancing = self.native.advance(
                angle.value, rate.value, x.value, velocity.value,
                acceleration.value, dt, balancing=self.balancing)
            return Acceleration(max(-25., min(25., demand)))
        except (ValidationError, ValueError) as exc:
            self.fault_reason = str(exc)
            return None


def create_policy():
    return Controller()
