"""Offline AUTO replay and declared ideal-plant counterfactuals; never opens a rig.

Replay checks the implemented controller against saved actions. Closed-loop
simulation is deliberately a diagnostic surrogate, not a hardware acceptance
model: ideal cart tracking, assumed angular damping, quantized encoder, 1 kHz
observer/governor, no serial delay, missed steps, or encoder nonlinearity.
"""
from dataclasses import asdict
import argparse
import json
import math

from cartpole_sdk.contracts import Observation, Sample, Session, UNITS
from cartpole_sdk.control.motion import State, advance
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.control.reference import ReferenceController, estimate, wrap
from cartpole_sdk.policies.executor_feedback import ExecutorFeedbackPolicy

_QUANTITIES=('position','velocity','angle','angular_velocity','acceleration','brake_direction')


def _observation(values, sequence, timestamp):
    return Observation('diagnostic','diagnostic-clock',sequence,timestamp,'offline snapshot',tuple(
        Sample('pole' if q in ('angle','angular_velocity') else 'cart',q,'simulation',float(value),
               UNITS[q],'diagnostic-frame','offline snapshot','valid',None,'estimated',
               timestamp,timestamp,timestamp) for q,value in zip(_QUANTITIES,values)))


def _policy(observation, parameters, rate_hz, mode='auto', max_gap_periods=None):
    policy=ExecutorFeedbackPolicy(tuple(s.channel for s in observation.samples),parameters=parameters,
        rate_hz=rate_hz,start_mode=mode,max_gap_periods=max_gap_periods)
    policy.reset(Session('diagnostic','diagnostic-clock'))
    return policy


def replay_actions(records, *, tolerance=1e-6):
    """Replay one/multiple host_controller windows; missing states are explicit.

    Input is a finite iterable of existing rig JSONL records. The entire input
    is materialized for matching, so select bounded windows from large logs.
    Missing states are counted, never interpolated. Subsequent policy cadence
    faults count as mismatches. This compares requested actions, not applied
    motor motion, and does not independently validate the raw safety channel.
    """
    if not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('tolerance must be finite and nonnegative')
    records=list(records)
    states={(str(r['state'].get('sid','')),r['state'].get('unwrapped_tick',r['state']['tick'])):r['state']
            for r in records if r.get('type')=='state'}
    snapshot=policy=None
    result=dict(actions=0,max_absolute_error=0.,mismatches=0,missing_states=0,first_mismatch=None)
    for record in records:
        if record.get('type')=='host_controller':
            snapshot=record;policy=None
        if record.get('type')!='action_request':continue
        if snapshot is None:raise ValueError('action requires preceding host controller snapshot')
        request=record['request'];sequence=request['observation_sequence']
        state=states.get((str(request.get('session_id','')),sequence))
        if state is None:
            result['missing_states']+=1;continue
        observation=_observation((state['x'],state['velocity'],state['angle_hat'],state['omega'],
            state['acceleration'],state['brake_direction']),sequence,state['unwrapped_us']*1000)
        if policy is None:
            policy=_policy(observation,ControllerParameters(**snapshot['controller_parameters']),
                snapshot['rate_hz'],snapshot['mode'],snapshot.get('max_gap_periods'))
        action=policy.act(observation)
        error=None if action is None else abs(action.acceleration_m_s2-request['acceleration_m_s2'])
        result['actions']+=1
        if error is not None:result['max_absolute_error']=max(result['max_absolute_error'],error)
        if error is None or error>tolerance:
            result['mismatches']+=1
            if result['first_mismatch'] is None:
                result['first_mismatch']=dict(observation_sequence=sequence,absolute_error=error,
                    reason=policy.fault_reason)
    return result


def simulate_feedback(parameters, *, controller='python', rate_hz=250, duration_s=10.,
                      theta=math.pi, omega=0., x=0., angular_damping_per_s=.2,
                      observer_angle=None, observer_rate=0., reset_observer=False):
    """Compare Python sample/hold with offline 1 kHz ReferenceController.

    Native-reference always resets its observer at AUTO entry, as the live
    native MODE path does. Python can retain a declared observer state or reset
    it as an explicit counterfactual. Neither choice changes deployed software.
    Every simulated action still passes through the 1 kHz binary32 governor.
    """
    if not isinstance(parameters,ControllerParameters):raise ValueError('parameters snapshot required')
    if controller not in ('python','native_reference'):raise ValueError('unknown controller')
    if type(rate_hz) is not int or rate_hz not in (100,250):raise ValueError('rate must be 100 or 250 Hz')
    values=(duration_s,theta,omega,x,angular_damping_per_s,observer_rate)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
        raise ValueError('finite numeric initial state required')
    if not 0<duration_s<=60 or angular_damping_per_s<0:raise ValueError('invalid experiment horizon/damping')
    if observer_angle is not None and (type(observer_angle) not in (int,float) or not math.isfinite(observer_angle)):
        raise ValueError('finite observer angle required')
    if type(reset_observer) is not bool:raise ValueError('reset_observer must be boolean')
    p=parameters;dt=.001;quantum=2*math.pi/4096
    raw=wrap(round(theta/quantum)*quantum)
    angle=raw if observer_angle is None else observer_angle
    rate=observer_rate
    if reset_observer or controller=='native_reference':angle,rate=raw,0.
    initial_observer=dict(angle=angle,rate=rate)
    initial_raw=raw
    motion=State();policy=None;reference=ReferenceController(p)
    reference.reset(raw)
    demand=0.;capture=None;balance_ticks=0;max_acceleration=0.;fault=None
    steps=max(1,round(duration_s*1000));elapsed=0
    for tick in range(steps):
        raw=wrap(round(theta/quantum)*quantum)
        if controller=='native_reference':
            output=reference.tick(raw,x,now_us=tick*1000,command='auto' if tick==0 else None)
            if not output.enabled:
                fault='native reference stopped/faulted';break
            motion=reference.motion
            balancing=output.mode==3
        else:
            angle,rate=estimate(raw,angle,rate,p.bw)
            if tick%(1000//rate_hz)==0:
                observation=_observation((x,motion.velocity,angle,rate,motion.acceleration,motion.brake_direction),tick,tick*1_000_000)
                if policy is None:policy=_policy(observation,p,rate_hz)
                action=policy.act(observation)
                if action is None:fault=policy.fault_reason;break
                demand=action.acceleration_m_s2
            motion=advance(motion,x,demand,min(p.vmax,max(p.vmax_s,p.vmax_b)),
                max(p.amax_s,p.amax_b),max(p.amax_s,p.amax_b),p.jmax,p.rail,dt,rail_margin=p.rail_margin)
            balancing=policy.mode=='balance'
        if balancing:
            if capture is None:capture=tick*dt
            balance_ticks+=1
        max_acceleration=max(max_acceleration,abs(float(motion.acceleration)))
        x+=float(motion.velocity)*dt
        alpha=(9.81*math.sin(theta)-float(motion.acceleration)*math.cos(theta))/p.leff-angular_damping_per_s*omega
        theta=wrap(theta+omega*dt+.5*alpha*dt*dt);omega+=alpha*dt
        elapsed=tick+1
        if abs(x)>p.rail-p.rail_fault_margin:fault='ideal plant rail boundary';break
    return dict(
        hardware_validated=False,model='ideal acceleration pendulum; assumed damping; quantized encoder; no transport delay',
        controller=controller,rate_hz=rate_hz if controller=='python' else 1000,parameters=asdict(p),
        angular_damping_per_s=angular_damping_per_s,initial_observer=initial_observer,
        initial_raw_angle=initial_raw,elapsed_s=elapsed*dt,capture_time_s=capture,balance_duration_s=balance_ticks*dt,
        max_abs_acceleration=max_acceleration,fault=fault,
        final=dict(x=x,velocity=float(motion.velocity),acceleration=float(motion.acceleration),theta=theta,omega=omega))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    replay=sub.add_parser('replay',help='compare requested actions from a bounded rig JSONL window')
    replay.add_argument('jsonl');replay.add_argument('--tolerance',type=float,default=1e-6)
    sim=sub.add_parser('simulate',help='declared surrogate only; never hardware acceptance')
    sim.add_argument('--parameters',required=True,help='JSON object of ControllerParameters values')
    sim.add_argument('--controller',choices=('python','native_reference'),default='python')
    sim.add_argument('--rate-hz',type=int,choices=(100,250),default=250)
    sim.add_argument('--duration-s',type=float,default=10.)
    sim.add_argument('--theta',type=float,default=math.pi);sim.add_argument('--omega',type=float,default=0.)
    sim.add_argument('--x',type=float,default=0.);sim.add_argument('--angular-damping-per-s',type=float,default=.2)
    sim.add_argument('--observer-angle',type=float);sim.add_argument('--observer-rate',type=float,default=0.)
    sim.add_argument('--reset-observer',action='store_true')
    args=vars(parser.parse_args(argv));operation=args.pop('operation')
    if operation=='replay':
        with open(args.pop('jsonl')) as stream:result=replay_actions((json.loads(line) for line in stream),**args)
    else:
        with open(args.pop('parameters')) as stream:parameters=ControllerParameters(**json.load(stream))
        result=simulate_feedback(parameters,**args)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
