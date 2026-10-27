"""Edit MyPolicy, then: uv run --extra sim python examples/simulate_policy.py runs/my-sim"""
import argparse
from cartpole_sdk import Acceleration, Channel, SelectionError, select_channels
from cartpole_sdk.sim import SimulationConfig
from cartpole_sdk.simulation import run_simulation, write_simulation


class MyPolicy:
    """Illustrative bounded cart damping; not a swing-up/balance controller."""
    def reset(self, session):
        self.velocity = Channel(session.topology.cart, 'velocity', 'simulation', 'sim:upright+x')

    def act(self, observation):
        try:
            (velocity,) = select_channels(observation, (self.velocity,),
                now_ns=observation.available_ns, max_age_ns=20_000_000)
        except SelectionError:
            return None
        return Acceleration(max(-1., min(1., -2 * velocity.value)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    args=parser.parse_args()
    run=run_simulation(MyPolicy(),config=SimulationConfig(control_steps=10,max_steps=300),
                       state=(0.,.2,3.14,0.))
    write_simulation(args.output,run)
    print(f'{len(run.steps)} control intervals; {run.end_reason}; results: {args.output}')


if __name__ == '__main__':
    main()
