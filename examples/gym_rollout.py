"""Exercise the RL interface; this random-policy smoke run does not train an agent.

uv run --extra sim python examples/gym_rollout.py --steps 200 --seed 7
Use CartpoleEnv with an RL library supporting continuous Box actions and Dict
observations; keep validity and age in the policy input. Ground truth in info is
for diagnostics and reward, not an automatic replacement for delayed sensors.
"""
import argparse
from cartpole_sdk.sim import SimulationConfig
from cartpole_sdk.sim.gym_env import CartpoleEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    env=CartpoleEnv(SimulationConfig(control_steps=10,max_steps=args.steps))
    try:
        observation,info=env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        reward_sum=0.
        for index in range(args.steps):
            observation,reward,terminated,truncated,info=env.step(env.action_space.sample())
            reward_sum+=reward
            if terminated or truncated:
                break
        print(f'{index+1} steps; reward={reward_sum:.6f}; rail={terminated}; horizon={truncated}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
