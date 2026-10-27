#!/usr/bin/env python3
"""Local MuJoCo auto/bal using the same ExecutorFeedbackPolicy as python-auto/python-bal.

This program has no hardware backend. A profile selects feedback gains AND the
external executor's limits; the underlying plant/sensor model is provisional.
"""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
from cartpole_sdk.contracts import ActionRequest
from cartpole_sdk.control.parameters import ControllerParameters, load_tuned_parameters
from cartpole_sdk.learning.rollout import initial_state, simulation_channels
from cartpole_sdk.policies import HostRatePolicy, HostPolicyParameters, ExecutorFeedbackPolicy
from cartpole_sdk.recording import strict_json_loads
from cartpole_sdk.sim import SimulationConfig, SensorConfig
from cartpole_sdk.sim.executor import ExecutorSimulation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',choices=('executor-feedback','legacy'),default='executor-feedback')
    parser.add_argument('--mode',choices=('auto','bal'),default='auto')
    parser.add_argument('--rate',type=int,choices=(100,250),default=250)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--profile',type=Path,help='complete 41-field saved snapshot, including rail')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0<args.seconds<=60:parser.error('seconds must be within (0,60]')
    if args.seed<0:parser.error('seed must be nonnegative')
    try:
        provenance=dict(profile=None,profile_sha256=None)
        if args.profile:
            raw=args.profile.read_bytes();profile=strict_json_loads(raw)
            if profile.get('schema_version')!=1 or set(profile['parameters'])!={f.name for f in fields(ControllerParameters)}:
                raise ValueError('simulation needs a complete snapshot; do not invent measured rail geometry for a partial profile')
            controller=ControllerParameters(**profile['parameters'])
            provenance=dict(profile=str(args.profile),profile_sha256=hashlib.sha256(raw).hexdigest())
        else:controller=load_tuned_parameters()
        config=SimulationConfig(control_steps=1000//args.rate,max_steps=math.ceil(args.seconds*args.rate),
            rail_half_m=controller.rail,effective_length_m=controller.leff)
        sensors=SensorConfig()
        policy=None
        sim=ExecutorSimulation(config,sensors,parameters=controller)
        trace=[];fault=None;dwell=longest=0.;first_capture=None
        state=initial_state(args.seed,'upright' if args.mode=='bal' else 'hanging')
        if args.controller=='executor-feedback' and args.mode=='bal':
            state=(state[0],0.,state[2],state[3])  # native BAL begins with a stopped cart
        try:
            observation=sim.reset(seed=args.seed,state=state)
            if args.controller=='executor-feedback':
                observation=sim.control_observation()
                lookup={s.quantity:s.channel for s in observation.samples}
                selected=tuple(lookup[q] for q in ('position','velocity','angle','angular_velocity','acceleration','brake_direction'))
                policy=ExecutorFeedbackPolicy(selected,rate_hz=args.rate,parameters=controller,start_mode=args.mode)
            else:
                parameters=HostPolicyParameters.from_controller(controller,start_mode=args.mode)
                policy=HostRatePolicy(simulation_channels(),rate_hz=args.rate,parameters=parameters,start_mode=args.mode)
            policy.reset(sim.session)
            for i in range(config.max_steps):
                action=policy.act(observation)
                if action is None:
                    fault=policy.fault_reason or 'no action';break
                observation=sim.step(ActionRequest(observation.session_id,observation.clock_id,i,
                    observation.sequence,observation.available_ns,observation.available_ns+20_000_000,
                    action.acceleration_m_s2))
                if args.controller=='executor-feedback':observation=sim.control_observation()
                t=sim.ground_truth
                balanced=abs(t.theta)<.15 and abs(t.omega)<1 and abs(t.x)<.12
                dwell=dwell+1/args.rate if balanced else 0.
                longest=max(longest,dwell)
                if balanced and first_capture is None:first_capture=t.time_ns/1e9
                trace.append(dict(**asdict(t),policy_mode=policy.mode,
                    raw_acceleration_m_s2=action.acceleration_m_s2,
                    executed_acceleration_m_s2=sim.diagnostics.executed_acceleration_m_s2,
                    rail_braking=sim.diagnostics.rail_braking))
                if sim.terminated:fault=sim.fault_reason;break
                if sim.truncated:break
        finally:sim.close()
        result=dict(backend='simulation',hardware_validated=False,controller=args.controller,mode=args.mode,rate_hz=args.rate,
            seed=args.seed,initial_state=state,config=asdict(config),sensors=asdict(sensors),
            executor_parameters=asdict(controller),policy_parameters=asdict(controller if args.controller=="executor-feedback" else parameters),
            provenance=provenance,fault=fault,first_balance_s=first_capture,
            longest_continuous_balance_s=longest,final_continuous_balance_s=dwell,
            balance_definition='abs(theta)<.15 rad, abs(omega)<1 rad/s, abs(x)<.12 m',
            trajectory=trace,notes=['Provisional plant and ideal simulated sensor; no real encoder filter or transport jitter.',
                'Raw demands pass through one 1 kHz governor. Host capture/recovery are sampled approximations; no native 1 kHz parity is claimed.'])
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    except (OSError,KeyError,ValueError,TypeError) as exc:parser.error(str(exc))
    print(json.dumps(dict(output=str(args.output),steps=len(trace),fault=fault,
                         longest_continuous_balance_s=longest,hardware_validated=False)))
    return 0


if __name__=='__main__':raise SystemExit(main())
