#!/usr/bin/env python3
"""Train numeric parameters from episodic reward in the provisional simulator."""
import argparse
from pathlib import Path
import sys

from cartpole_sdk.learning import train, save_artifact
from cartpole_sdk.sim import SimulationConfig, SensorConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/learned-policy.json'))
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--training-seeds',type=int,nargs='+',default=[11,12,13])
    parser.add_argument('--generations',type=int,default=4)
    parser.add_argument('--population',type=int,default=10)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--rate',type=int,choices=(100,250),default=100)
    parser.add_argument('--sensor-period-ms',type=int,default=10)
    parser.add_argument('--sensor-delay-ms',type=int,default=0)
    parser.add_argument('--angle-noise',type=float,default=0.)
    args=parser.parse_args()
    if not 1<=args.seconds<=60: parser.error('--seconds must be between 1 and 60')
    config=SimulationConfig(control_steps=1000//args.rate,max_steps=round(args.seconds*args.rate))
    sensors=SensorConfig(sample_every_steps=args.sensor_period_ms,delay_steps=args.sensor_delay_ms,
                         angle_noise_std=args.angle_noise)
    def progress(entry):
        print(f"generation={entry['generation']} best_training_score={entry['best_score']:.6f}",flush=True)
    artifact=train(seed=args.seed,training_seeds=tuple(args.training_seeds),generations=args.generations,
                   population=args.population,config=config,sensors=sensors,progress=progress)
    save_artifact(artifact,args.output)
    print(f"Saved {args.output}; simulation only, hardware_accepted=false")


if __name__ == '__main__': main()
