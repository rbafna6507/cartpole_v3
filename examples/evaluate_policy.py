#!/usr/bin/env python3
"""Compare immutable learned weights with the pretraining handwritten policy."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from hashlib import sha256

from cartpole_sdk.learning import evaluate, read_artifact
from cartpole_sdk.learning.search import provenance
from cartpole_sdk.policies import HostPolicyParameters
from cartpole_sdk.sim import SimulationConfig, SensorConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1001,1002,1003,1004])
    parser.add_argument('--output',type=Path,default=Path('runs/heldout-evaluation.json'))
    parser.add_argument('--seconds',type=float)
    parser.add_argument('--rate',type=int,choices=(100,250))
    parser.add_argument('--sensor-period-ms',type=int)
    parser.add_argument('--sensor-delay-ms',type=int)
    parser.add_argument('--trajectory',action='store_true')
    args=parser.parse_args()
    artifact=read_artifact(args.artifact)
    cfg=dict(artifact['training']['simulation_config'])
    sensor_cfg=dict(artifact['training']['sensor_config'])
    seconds=args.seconds if args.seconds is not None else cfg['max_steps']*cfg['control_steps']/1000
    rate=args.rate or artifact['rate_hz']
    if not 1<=seconds<=60: parser.error('--seconds must be between 1 and 60')
    cfg.update(control_steps=1000//rate,max_steps=round(seconds*rate))
    if args.sensor_period_ms is not None: sensor_cfg['sample_every_steps']=args.sensor_period_ms
    if args.sensor_delay_ms is not None: sensor_cfg['delay_steps']=args.sensor_delay_ms
    config,sensors=SimulationConfig(**cfg),SensorConfig(**sensor_cfg)
    # Supplying a rate override is a separate evaluation condition, never silently
    # rewriting the artifact's deployment rate.
    kw=dict(seeds=tuple(args.seeds),config=config,sensors=sensors,trajectory=args.trajectory)
    learned=evaluate(artifact,**kw)
    baseline=evaluate(HostPolicyParameters(**artifact['training']['baseline_parameters']),**kw)
    report=dict(artifact_sha256=sha256(args.artifact.read_bytes()).hexdigest(),
        artifact_rate_hz=artifact['rate_hz'],evaluated_rate_hz=rate,
        provenance=provenance(config),baseline=baseline,learned=learned,
        mean_score_improvement=learned['mean_score']-baseline['mean_score'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for label,value in [('baseline',baseline),('learned',learned)]:
        print(f"{label}: score={value['mean_score']:.6f} success={value['successes']}/{value['episodes_count']}")
        for e in value['episodes']:
            print(f"  seed={e['seed']} {e['start']} score={e['score']:.4f} success={e['success']} final_dwell={e['final_balance_dwell_s']:.2f}s peak_x={e['max_abs_x_m']:.4f}m fault={e['fault_reason']}")
    print(f'Saved {args.output}; provisional model, no hardware acceptance claim')


if __name__ == '__main__': main()
