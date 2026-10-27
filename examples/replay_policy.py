"""Edit MyPolicy to experiment locally. This example never connects to hardware.

Run from the checkout: uv run python examples/replay_policy.py RECORDING TRACE --frame-id FRAME
This illustrative mapping is not the tuned swing-up/balance controller.
"""
import argparse

from cartpole_sdk import (
    Acceleration, ActionLimits, Channel, Observation, SelectionError, Session,
    select_channels,
)
from cartpole_sdk.recording import read_recording
from cartpole_sdk.replay import replay, write_trace


class MyPolicy:
    def __init__(self, frame_id: str):
        self.channel = Channel('pole', 'angular_velocity', 'vision', frame_id)

    def reset(self, session: Session) -> None:
        # Clear any stateful filter/model here. Policy state must not leak between runs.
        if self.channel.entity not in session.topology.joints:
            raise ValueError('This example expects a joint named pole')

    def act(self, observation: Observation) -> Acceleration | None:
        try:
            (omega,) = select_channels(observation, (self.channel,),
                now_ns=observation.available_ns, max_age_ns=100_000_000)
        except SelectionError:
            return None
        # SI: angular velocity is rad/s, returned acceleration is m/s².
        # Sign, gain and clamp below are illustrative offline choices.
        return Acceleration(max(-1.0, min(1.0, -0.05 * omega.value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording')
    parser.add_argument('trace')
    parser.add_argument('--frame-id', required=True)
    args = parser.parse_args()
    session, observations = read_recording(args.recording)
    steps = replay(session, observations, MyPolicy(args.frame_id),
                   ActionLimits(1.0, 20_000_000, 100_000_000))
    write_trace(args.trace, session, steps)
    print(f'Replayed {len(steps)} observations; saved {args.trace}')


if __name__ == '__main__':
    main()
