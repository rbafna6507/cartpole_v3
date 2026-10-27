"""Operator procedures for immutable collection, fitting and scoped simulation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def parser():
    root=argparse.ArgumentParser(description=__doc__,epilog='Help and offline commands never open serial. See docs/system-identification-workflow.md for purpose, annotations and model limits.')
    commands=root.add_subparsers(dest='command')
    for name,purpose in (('inspect','Read saved reference, home, health and live parameters in a stopped session.'),
        ('collect-passive','Record operator-released oscillations with the motor disabled; physically restrain the cart.'),
        ('collect-pulse','Execute one allowlisted cart pulse then12s feedback hold; requires --execute.')):
        p=commands.add_parser(name,help=purpose,description=purpose)
        p.add_argument('--port',required=True,help='Explicit serial device; no discovery, home, clear or calibration.')
        p.add_argument('--output',type=Path,required=True,help='New immutable run directory, including failure evidence.')
        p.add_argument('--expected-reference',type=int,help='Optional exact saved upright raw encoder reference.')
        p.add_argument('--expected-home',type=int,help='Optional exact current home epoch.')
        if name!='inspect':
            p.add_argument('--dataset-role',choices=('training','validation','development'),default='development')
            p.add_argument('--run-label',default='unspecified',help='Descriptive immutable recording label.')
            camera=p.add_mutually_exclusive_group()
            camera.add_argument('--camera-url',default='http://127.0.0.1:3090/api/sensing-stream?max_hz=100',help='Raw sensing-stream URL; must become ready before serial collection.')
            camera.add_argument('--no-camera',action='store_true',help='Pulse only: explicitly record missing independent camera evidence; passive collection requires camera.')
        if name=='collect-passive':
            p.add_argument('--seconds',type=float,default=120.,help='Maximum recording duration, 1..180s; type done then Enter to finish early after recording begins.')
            p.add_argument('--release-angles-deg',type=float,nargs='+',help='Requested degrees away from hanging; labels are not measured angles.')
            p.add_argument('--release-directions',choices=('left','right'),nargs='+',default=['left','right'],help='Directions in camera view.')
            p.add_argument('--repetitions',type=int,default=1)
            p.add_argument('--cart-restraint',default='',help='How you will keep the cart fixed; required with requested angles.')
        if name=='collect-pulse':
            p.add_argument('--pulse-seconds',type=float,choices=(.45,.65,.8,1.),default=.65,help='Pulse duration; unsafe amplitude/duration pairs fail before serial. Default0.65s.')
            p.add_argument('--hold-seconds',type=float,default=12.,help='Explicit3..12s active cart feedback hold, default12s;20ms leases are unchanged.')
            p.add_argument('--execute',action='store_true',help='Authorize this invocation to issue ARM and acceleration requests.')
            p.add_argument('--amplitude-mm',type=int,required=True,choices=(-30,-20,-10,-5,5,10,20,30),help='Signed pulse amplitude; begin with5mm and inspect before escalating.')
    p=commands.add_parser('recover-passive',help='Explicitly review a passive KeyboardInterrupt and preserve a separate recovery decision; never rewrite raw evidence.')
    p.add_argument('--input',required=True,help='Local Rig JSONL or JSONL.gz with its original sibling reports and camera evidence.')
    p.add_argument('--reason',required=True,help='Why the operator interrupted this passive recording; required for the recovery audit.')
    p=commands.add_parser('review-recording',help='Check completed raw data and plot observations without fitting or selecting intervals.')
    p.add_argument('--input',required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('review-releases',help='Compare raw camera/encoder relative hanging and qualify explicitly annotated release poses; no fitting.')
    p.add_argument('--input',required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('review-fit',help='Plot frozen passive-fit predictions on the exact validation data; never refit or change acceptance.')
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--validation',required=True)
    p.add_argument('--validation-annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('annotations-template',help='Create an explicit manual free-release annotation template; never select fit windows automatically.')
    p.add_argument('--input',required=True,help='Rig JSONL, JSONL.gz, or archive.tar.gz::exact/member.jsonl')
    p.add_argument('--output',required=True,type=Path)
    p=commands.add_parser('fit-passive',help='Fit training-only passive dynamics and score independently annotated validation releases.')
    p.add_argument('--training',required=True);p.add_argument('--training-annotations',type=Path,required=True)
    p.add_argument('--validation',required=True);p.add_argument('--validation-annotations',type=Path,required=True)
    p.add_argument('--model',choices=('viscous','viscous_coulomb'),default='viscous',help='Declare before examining validation; never select the winner from validation.')
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('fit-driven',help='Run the existing strict driven pendulum fitter using raw theta and MCU acceleration.')
    p.add_argument('--training',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('assumptions-template',help='Create an editable complete provisional simulation configuration at100Hz; values remain assumptions.')
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('export-model',help='Export an accepted scoped fit with explicit assumptions; unsupported dynamics fail closed.')
    p.add_argument('--report',type=Path,required=True);p.add_argument('--scope',choices=('passive-down','driven'),required=True)
    p.add_argument('--assumptions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('gym-smoke',help='Load an exported model and run a bounded headless zero-action check; no hardware/RL claim.')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,default=100)
    p.add_argument('--output',type=Path,required=True)
    return root


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.command is None:p.print_help();return 0
    if args.command=='collect-pulse' and not args.execute:p.error('collect-pulse requires explicit --execute')
    from . import workflow
    try:
        if args.command=='recover-passive':
            from .interruption_recovery import recover_passive
            path=recover_passive(args.input,reason=args.reason)
            print(json.dumps(dict(recovery=str(path),status='recovered_for_review')));return 0
        if args.output.exists():raise FileExistsError(f'output exists; preserve earlier evidence: {args.output}')
        if args.command in ('inspect','collect-passive','collect-pulse'):
            from .experiments import collect
            from .protocol import make_protocol
            from .finish import TerminalFinish
            kind={'inspect':'inspect','collect-passive':'passive','collect-pulse':'pulse'}[args.command]
            camera_url=None if kind=='inspect' or getattr(args,'no_camera',False) else args.camera_url
            protocol=make_protocol(kind,dataset_role=getattr(args,'dataset_role','development'),run_label=getattr(args,'run_label','unspecified'),
                angles_deg=getattr(args,'release_angles_deg',None),directions=getattr(args,'release_directions',['left','right']),
                repetitions=getattr(args,'repetitions',1),cart_restraint=getattr(args,'cart_restraint',''))
            def announce(status):
                print(json.dumps(dict(output=str(args.output),**status)),flush=True)
                if status.get('status')=='recording':
                    if status.get('instruction'):print(status['instruction'],flush=True)
                    for line in status.get('requested_release_instructions',[]):print(line,flush=True)
            result=collect(kind,output=args.output,port=args.port,seconds=getattr(args,'seconds',120.),
                amplitude_m=getattr(args,'amplitude_mm',5)/1000,pulse_seconds=getattr(args,'pulse_seconds',.65),hold_seconds=getattr(args,'hold_seconds',12.),camera_url=camera_url,
                expected_reference=args.expected_reference,expected_home=args.expected_home,
                ready=announce,protocol=protocol,finish_requested=TerminalFinish(sys.stdin) if kind=='passive' else None)
            completion={key:result.get('result',{})[key] for key in ('finish_reason','elapsed_seconds','requested_seconds') if key in result.get('result',{})}
            print(json.dumps(dict(output=str(args.output),completed=result['completed'],stop_confirmed=result.get('stop_confirmed'),**completion)));return 0
        if args.command=='review-releases':
            from .release_review import review_releases
            result=review_releases(args.input,args.annotations,args.output)
            print(json.dumps(dict(output=str(args.output),passed=result['passed'],all_release_poses_verified=result['all_release_poses_verified'],physical_model_validated=False)))
            return 0 if result['passed'] and result['all_release_poses_verified'] else 2
        if args.command=='review-recording':
            from .review import review_recording
            result=review_recording(args.input,args.output)
            print(json.dumps(dict(output=str(args.output),source_sha256=result['source_sha256'],physical_model_validated=False)));return 0
        elif args.command=='review-fit':
            from .fit_review import review_fit
            result=review_fit(args.fit,args.validation,args.validation_annotations,args.output)
            print(json.dumps(dict(output=str(args.output),accepted=result['saved_admission']['accepted'],physical_model_validated=False)));return 0
        elif args.command=='annotations-template':result=workflow.annotation_template(args.input)
        elif args.command=='fit-passive':result=workflow.fit_passive_logs(args.training,args.training_annotations,args.validation,args.validation_annotations,model=args.model)
        elif args.command=='fit-driven':result=workflow.fit_driven_logs(args.training,args.validation)
        elif args.command=='assumptions-template':
            from cartpole_sdk.sim.config import SimulationConfig
            result=dict(simulation_config=asdict(SimulationConfig(control_steps=10)),rationale='',
                instructions='Review all assumed masses, force, rail and actuator limits; provide rationale. Export replaces only effective length and mapped hinge damping. control_steps=10 with1ms physics gives100Hz.')
        elif args.command=='export-model':
            result=workflow.export_model(workflow.read_json(args.report),scope=args.scope,
                assumptions=workflow.read_json(args.assumptions),output=args.output)
            print(json.dumps(dict(output=str(args.output),scope=result['acceptance_scope'],recommended_for_hardware_rl=False)));return 0
        else:result=workflow.gym_smoke(args.model,steps=args.steps)
        workflow.write_json(args.output,result)
        print(json.dumps(dict(output=str(args.output),accepted=result.get('accepted'),reasons=result.get('reasons',[]))))
        return 2 if result.get('accepted') is False else 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'cartpole-id: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
