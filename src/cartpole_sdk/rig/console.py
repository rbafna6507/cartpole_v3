"""Operator console for board reference modes and explicit external policies."""
import argparse
from dataclasses import asdict
import math
import shlex
import sys
import time
from cartpole_sdk.contracts import Channel, ValidationError
from cartpole_sdk.recording import canonical_json
from .client import Rig, RigError
from .startup import StartupParameters
from ..sensing.telemetry import EncoderPublisher
from .policy import ZeroAccelerationPolicy, run_policy, run_host_controller

HELP = ('auto [seconds], bal [seconds], python-auto [seconds], python-auto-trial [seconds], python-bal [seconds], manual [seconds], velocity m/s [ttl_ms], '
        'stop, center, home, rehome, upright, hanging MODEL IDENTITY | hanging ideal, switches on|off, status, info, clear, rate hz, '
        'z status|limit [MM]|reference AXIS|home AXIS|move AXIS up|down MM [SPEED]|stop|sine [A HZ CENTER SEC], '
        'params, get key, set key value, save path, load path, '
        'policy load PATH|status|rail-tolerance [MM]|check [swingup]|shadow [swingup] [seconds]|run [seconds]|swingup [seconds]|unload, quit')

def _status(rig, output):
    state=rig.state
    data=asdict(state)
    data['host_receipt_age_ms']=(time.monotonic_ns()-state.received_ns)/1e6
    data['cart_position_basis']='estimated step count'
    data['cart_velocity_basis']='commanded velocity'
    data['mode_basis']='external host policy' if state.mode==12 else 'board reference/operator'
    data['error']=None if rig.error is None else str(rig.error)
    data['switch_inputs']={label:'open' if state.flags & (16 << i) else 'closed'
                           for i,label in enumerate(('1_left','2_right','3','4'))}
    data['switch_checking_enabled']=state.switch_profile
    data['encoder_healthy']=bool(state.flags & 1)
    data['drivers_enabled']=state.enabled
    data['homed']=state.homed
    data['returning_to_existing_home']=state.mode==14
    data['calibrating_encoder']=bool(state.flags & 256)
    print(canonical_json(data),file=output)

def execute_command(rig, line, *, output=None):
    """One interactive trusted operator command; return False to quit."""
    output=sys.stdout if output is None else output
    words=shlex.split(line)
    if not words:return True
    command,*args=words
    if command in ('quit','exit'):
        rig.stop(); return False
    if command in ('help','?'):
        print(HELP,file=output);return True
    if command=='z':
        if args == ['status']:
            print(canonical_json(asdict(rig.z_status())), file=output)
        elif args and args[0] == 'limit' and len(args) in (1, 2):
            print(canonical_json(asdict(rig.z_limit(None if len(args) == 1 else float(args[1])))), file=output)
        elif args == ['stop']:
            rig.z_stop()
            print('vertical motion stopped', file=output)
        elif len(args) == 2 and args[0] in ('reference', 'home'):
            getattr(rig, 'z_'+args[0])(args[1])
            print('operator confirmed physical bottom; counted reference established' if args[0] == 'reference'
                  else 'timed downward home started: 40mm/s for 5s; confirm bottom before referencing', file=output)
        elif len(args) in (4, 5) and args[0] == 'move' and args[2] in ('up', 'down'):
            distance = float(args[3])
            if not math.isfinite(distance) or distance <= 0:
                raise ValidationError('move distance must be finite and positive; choose up or down')
            rig.z_move(args[1], distance if args[2] == 'up' else -distance,
                       2. if len(args) == 4 else float(args[4]))
            print('vertical move started', file=output)
        elif args and args[0] == 'sine' and len(args) in (1, 5):
            rig.z_sine(*map(float, args[1:]))
            print('finite paired vertical sine started', file=output)
        else:
            raise ValidationError('use z status|limit [MM]|reference AXIS|home AXIS|move AXIS up|down MM [SPEED]|stop|sine [A HZ CENTER SEC]')
    elif command=='policy':
        from .console_policy import execute_policy_command
        execute_policy_command(rig,args,output)
    elif command in ('auto','bal','balance','manual'):
        if len(args)>1:raise ValidationError('mode accepts at most one duration')
        mode='bal' if command=='balance' else command
        duration=10. if not args else float(args[0])
        rig.start(mode,duration)
        suffix='; normal expiry returns to the existing home within 15s, then disables' if rig._return_expected else ''
        print(f'board reference {mode} active for at most {duration:g}s{suffix}',file=output)
    elif command in ('python-auto','python-auto-trial','python-bal'):
        if len(args)>1:raise ValidationError('host mode accepts at most one duration')
        print('Experimental Python control, filtered magnetic feedback, raw-angle guard and step governor state; Ctrl-C stops.', file=output, flush=True)
        runner=run_host_controller
        if command=='python-auto-trial':
            from cartpole_sdk.policies.experimental_auto import run_trial
            runner=run_trial
        runner(rig, mode='auto' if command=='python-auto-trial' else command.removeprefix('python-'), duration_s=10. if not args else float(args[0]))
        print('Python run ended; stopped and disabled. Explicit rehome is available.', file=output)
    elif command=='velocity':
        if not 1<=len(args)<=2:raise ValidationError('velocity requires m/s and optional TTL milliseconds')
        seq=rig.set_velocity(float(args[0]),50 if len(args)==1 else int(args[1]))
        print(f'one manual velocity lease, sequence {seq}; expires unless explicitly renewed',file=output)
    elif command=='switches':
        if args not in (['on'],['off']):raise ValidationError('switches requires on or off')
        rig.configure_switches(args[0]=='on')
        print(f'physical switch profile {args[0]}',file=output)
    elif command in ('upright','hanging'):
        options={}
        if command=='upright':
            if args:raise ValidationError('upright accepts no arguments')
        elif args==['ideal']:
            print('Using the explicit ideal half-turn assumption; no measured correction.',file=output)
        elif len(args)==2:
            from ..sensing.calibration import CalibrationProfile, load_json
            model=CalibrationProfile.load(args[0]);identity=load_json(args[1])
            if not isinstance(identity,dict):raise ValidationError('installation identity must be an object')
            try:offset=model.approved_offset_counts(**identity)
            except (ValueError,TypeError) as exc:raise ValidationError(str(exc)) from exc
            options=dict(hanging_model=model,calibration_identity=identity)
            print(f'Using calibration {model.calibration_id}; hanging correction {offset:+d} counts.',file=output)
        else:
            raise ValidationError('hanging requires an approved model and installation identity: hanging MODEL.json IDENTITY.json; or explicitly hanging ideal')
        rig._stopped_healthy()  # Check before asking the operator to touch the pole.
        instruction='hold the pole straight up' if command=='upright' else 'let the pole hang fully down'
        print(f'Encoder calibration: {instruction} and keep it still; waiting for capture and EEPROM verification (up to 5s).',file=output,flush=True)
        info=rig.calibrate_encoder(command,**options)
        print(f'encoder calibration saved; upright raw reference {info.encoder_upright_raw}',file=output)
    elif command in ('params','get','set','save','load'):
        required={'params':0,'get':1,'set':2,'save':1,'load':1}[command]
        if len(args)!=required:raise ValidationError(f'{command} requires {required} argument(s)')
        if command=='params':values=rig.get_parameters()
        elif command=='get':values={args[0]:rig.get_parameter(args[0])}
        elif command=='set':values={args[0]:rig.set_parameter(args[0],float(args[1]))}
        elif command=='save':
            profile=rig.save_parameters(args[0])
            print(f'verified live parameter profile saved to {args[0]}',file=output)
            values=profile['parameters']
        else:
            values=rig.load_parameters(args[0])
            print(f'profile applied and read back from {args[0]}; executor remains stopped',file=output)
        for name,value in values.items():print(f'{name} = {value:.9g}',file=output)
    elif command=='rate':
        if len(args)!=1:raise ValidationError('rate requires one Hz value')
        rig.set_rate(int(args[0]))
    elif command in ('stop','center','home','rehome','status','info','clear'):
        if args:raise ValidationError(f'{command} accepts no arguments')
        if command=='status':_status(rig,output)
        elif command=='info':print(canonical_json(asdict(rig.get_info())),file=output)
        elif command=='center':
            rig.center()
            print('operator confirmed cart physically centered; step origin established',file=output)
        elif command=='rehome':
            rig.rehome()
            print('returning to the existing counted home; no switch seek; at most 15s',file=output)
        elif command=='clear':rig.clear_fault()
        else:getattr(rig,command)()
    else:
        raise ValidationError(f'unknown command: {command}; {HELP}')
    return True

def _wait_run(rig, duration_s, output):
    if rig._return_expected:
        timeout=max(0,(rig._deadline_ns-time.monotonic_ns())/1e9) if rig._deadline_ns is not None else 0
        _wait_home(rig,timeout,output,operation='automatic return',require_counted_center=True)
        return
    deadline=time.monotonic()+duration_s
    if rig._deadline_ns is not None:
        deadline=min(deadline,rig._deadline_ns/1e9)-min(.01,duration_s*.25)
    try:
        while time.monotonic()<deadline:
            if rig.error is not None:raise RigError(str(rig.error))
            remaining=deadline-time.monotonic()
            if remaining<=0:break
            try:
                rig.wait_state(after_tick=rig.state.unwrapped_tick,timeout=min(remaining,.1))
            except TimeoutError:
                if time.monotonic()>=deadline:break
                raise
        _status(rig,output)
    finally:
        rig.stop()

def _wait_home(rig, timeout, output, *, operation='physical homing',require_counted_center=False):
    deadline=time.monotonic()+timeout
    try:
        while True:
            if rig.error is not None:raise RigError(str(rig.error))
            state=rig.state
            if require_counted_center and not rig._return_expected:
                raise RigError('return cancelled before completion')
            if (state.homed and state.healthy and not state.enabled and state.mode==0 and
                (not require_counted_center or state.x==0)):
                _status(rig,output)
                return
            remaining=deadline-time.monotonic()
            if remaining<=0:
                raise TimeoutError(f'{operation} completion timeout')
            rig.wait_state(after_tick=state.unwrapped_tick,timeout=min(remaining,.1))
    finally:
        rig.stop()

def main(argv=None, *, rig_factory=None, input_fn=None, output=None):
    output=sys.stdout if output is None else output
    input_fn=input if input_fn is None else input_fn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--rate',type=int,default=100,choices=(0,100,250,500,1000))
    parser.add_argument('--log-dir',default='runs/rig')
    parser.add_argument('--vision-telemetry-port',type=int,default=3096,
                        help='read-only loopback encoder mirror port (default 3096; 0 disables)')
    parser.add_argument('--no-vision-telemetry',action='store_true',help='disable loopback encoder mirror')
    parser.add_argument('--hanging-model',help='approved hanging calibration model JSON')
    parser.add_argument('--calibration-identity',help='separate current installation identity JSON')
    parser.add_argument('--ideal-half-turn',action='store_true',help='explicitly assume hanging is exactly opposite upright')
    parser.add_argument('--command',choices=('auto','bal','manual','velocity','stop','center','home','rehome','upright','hanging','status','info','clear','params','get','set','save','load','policy-zero','policy-run','python-auto','python-auto-trial','python-bal'))
    parser.add_argument('--actor',help='validated numeric actor JSON for --command policy-run')
    parser.add_argument('--development-only',action='store_true',help='acknowledge unqualified learned policy execution')
    parser.add_argument('--duration',type=float,default=None,help='motion duration (default 10s; learned policy 5s), HOME timeout (75s), or REHOME timeout (15.25s)')
    parser.add_argument('--switches',choices=('on','off'),help='explicit stopped physical switch profile selection')
    parser.add_argument('--center-confirmed',action='store_true',help='operator asserts cart is physically centered now')
    parser.add_argument('--param',help='parameter name for get or set')
    parser.add_argument('--value',type=float,help='parameter value for set')
    parser.add_argument('--profile',help='profile path for save or load')
    startup_options=parser.add_mutually_exclusive_group()
    startup_options.add_argument('--startup-profile',help='apply this profile instead of the preserved tuned default')
    startup_options.add_argument('--keep-board-params',action='store_true',help='keep live board parameters; disable automatic startup profile')
    parser.add_argument('--velocity',type=float,default=0.)
    parser.add_argument('--ttl-ms',type=int,default=50)
    parser.add_argument('--source',choices=('magnetic-step',),help='explicit source/frame selection for external Python policies')
    args=parser.parse_args(argv)
    if args.command=='policy-run':
        if not args.actor or not args.development_only:
            parser.error('policy-run requires --actor PATH and --development-only')
    elif args.actor or args.development_only:
        parser.error('--actor and --development-only require --command policy-run')
    if args.hanging_model or args.calibration_identity or args.ideal_half_turn:
        if args.command!='hanging':parser.error('hanging calibration options require --command hanging')
        if args.ideal_half_turn and (args.hanging_model or args.calibration_identity):
            parser.error('choose a hanging model or ideal half-turn, not both')
        if bool(args.hanging_model)!=bool(args.calibration_identity):
            parser.error('model calibration requires --hanging-model and --calibration-identity')
    if not 0<=args.vision_telemetry_port<=65535:
        parser.error('vision telemetry port must be 0..65535')
    if args.duration is None:
        args.duration=75. if args.command=='home' else (15.25 if args.command=='rehome' else (5. if args.command=='policy-run' else 10.))
    maximum=75 if args.command=='home' else (15.25 if args.command=='rehome' else 60)
    if not math.isfinite(args.duration) or not .001<=args.duration<=maximum:
        parser.error(f'duration must be finite and within 1ms..{maximum}s')
    if args.command in ('get','set') and not args.param:
        parser.error('get/set requires --param')
    if args.command=='set' and args.value is None:
        parser.error('set requires --value')
    if args.command in ('save','load') and not args.profile:
        parser.error('save/load requires --profile')
    if args.command in ('policy-zero','python-auto','python-auto-trial','python-bal') and args.source is None:
        parser.error('external policy requires --source magnetic-step')
    if args.command in ('python-auto','python-auto-trial','python-bal') and args.rate not in (100,250):
        parser.error('Python controller supports 100 or 250 Hz')
    if args.command=='python-auto-trial' and args.rate!=250:
        parser.error('python-auto-trial requires --rate 250')
    factory=Rig if rig_factory is None else rig_factory
    try:
        if args.command=='policy-run':
            from .console_policy import ConsolePolicy
            ConsolePolicy.validate_actor(args.actor)  # Refuse before opening serial.
        publisher=EncoderPublisher(0 if args.no_vision_telemetry else args.vision_telemetry_port)
        with StartupParameters(command=args.command,profile=args.startup_profile,keep_board=args.keep_board_params) as startup, \
                factory(args.port,rate_hz=args.rate,log_dir=args.log_dir,vision_publisher=publisher) as rig:
            rig.get_info()  # Read reference/configuration before any optional operator motion.
            print(f'connected stopped, session {rig.session.session_id}; log {rig.log_path}',file=output)
            startup.apply(rig,output,defer_unhealthy=args.command is None)
            if args.switches is not None:rig.configure_switches(args.switches=='on')
            if args.center_confirmed:rig.center()
            if args.command in ('auto','bal','manual'):
                execute_command(rig,f'{args.command} {args.duration}',output=output)
                _wait_run(rig,args.duration,output)
            elif args.command in ('python-auto','python-auto-trial','python-bal'):
                execute_command(rig,f'{args.command} {args.duration}',output=output)
            elif args.command=='policy-run':
                execute_command(rig,f'policy load {shlex.quote(args.actor)}',output=output)
                execute_command(rig,f'policy run {args.duration}',output=output)
            elif args.command=='policy-zero':
                print('external host zero-acceleration policy; no swing-up/balance controller',file=output)
                channels=(Channel('pole','angle','magnetic','encoder-zero'),Channel('cart','position','step','cart-step-home'))
                run_policy(rig,ZeroAccelerationPolicy,channels=channels,duration_s=args.duration)
            elif args.command=='velocity':
                rig.start('manual',args.duration)
                rig.set_velocity(args.velocity,args.ttl_ms)
                _wait_run(rig,min(args.duration,args.ttl_ms/1000*.8),output)
            elif args.command=='home':
                rig.home()
                _wait_home(rig,args.duration,output)
            elif args.command=='rehome':
                execute_command(rig,'rehome',output=output)
                _wait_home(rig,args.duration,output,operation='return to existing home',require_counted_center=True)
            elif args.command in ('get','set'):
                line=f'{args.command} {shlex.quote(args.param)}'
                if args.command=='set':line+=f' {args.value!r}'
                execute_command(rig,line,output=output)
            elif args.command in ('save','load'):
                execute_command(rig,f'{args.command} {shlex.quote(args.profile)}',output=output)
            elif args.command=='hanging':
                line='hanging'
                if args.ideal_half_turn:line+=' ideal'
                elif args.hanging_model:
                    line+=f' {shlex.quote(args.hanging_model)} {shlex.quote(args.calibration_identity)}'
                execute_command(rig,line,output=output)
            elif args.command is not None:
                execute_command(rig,args.command,output=output)
            else:
                print(HELP,file=output)
                while True:
                    try:
                        line=input_fn('cartpole> ')
                    except EOFError:break
                    try:
                        startup.before_command(rig,line,output)
                        if not execute_command(rig,line,output=output):break
                        startup.after_command(rig,line,output)
                    except KeyboardInterrupt:
                        if shlex.split(line)[:2] not in (['policy','run'],['policy','shadow'],['policy','swingup']):
                            raise
                        rig.stop()
                        print('policy interrupted; stopped and disabled; console remains connected',file=output)
                    except (RigError,ValidationError,ValueError,TimeoutError,OSError) as exc:
                        print(f'error: {exc}',file=output)
        return 0
    except KeyboardInterrupt:
        print('interrupted; stop requested',file=output)
        return 130
    except (RigError,ValidationError,ValueError,TimeoutError,OSError,RuntimeError) as exc:
        print(f'error: {exc}',file=output)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
