"""Experimental native-header AUTO/BAL; host sampling is not MCU 1 kHz parity."""
from cartpole_sdk.control.native_controller import NativeController
from .executor_feedback import ExecutorFeedbackPolicy


class NativeExecutorFeedbackPolicy(ExecutorFeedbackPolicy):
    def __init__(self, channels, *, parameters, library=None, rate_hz=100,
                 start_mode='auto', raw_angle_channel=None):
        native=NativeController(parameters,library=library,upright=start_mode=='bal')
        try:
            super().__init__(channels,parameters=parameters,rate_hz=rate_hz,
                start_mode=start_mode,raw_angle_channel=raw_angle_channel,controller_math=native)
        except BaseException:
            native.close()
            raise
        self._native_parameters=parameters
        self.native_provenance=native.provenance

    def act(self, observation):
        if self.parameters!=self._native_parameters:
            return self._fault('native controller parameters changed; create a new policy')
        try:return super().act(observation)
        except ValueError as exc:return self._fault(str(exc))

    def close(self):self.controller_math.close()
