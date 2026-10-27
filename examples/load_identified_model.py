#!/usr/bin/env python3
"""Run a content-hashed identification export in headless100Hz Gym; no hardware."""
import argparse
from dataclasses import replace
import json
import math
from cartpole_sdk.identification.workflow import load_model
from cartpole_sdk.sim.gym_env import CartpoleEnv


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model');args=parser.parse_args(argv)
    # Explicit100Hz policy with the model's required1ms physics. Configuration
    # provenance remains in the export; this example is numerical use only.
    config=replace(load_model(args.model),physics_dt_ns=1_000_000,control_steps=10)
    env=CartpoleEnv(config=config)
    try:
        observation,info=env.reset(seed=0,options={'state':(0.,0.,math.pi+.1,0.)})
        for step in range(100):
            observation,reward,terminated,truncated,info=env.step([0.])
            if terminated or truncated:break
        print(json.dumps(dict(steps=step+1,control_interval_s=.01,ground_truth=info['ground_truth'],
            note='Scoped model execution only; not physical validation or hardware RL readiness.')))
    finally:env.close()


if __name__=='__main__':main()
