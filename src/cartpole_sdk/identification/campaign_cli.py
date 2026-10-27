"""Run the remaining system-ID measurements and package raw evidence for analysis."""
import argparse
import json
from pathlib import Path
import sys
from . import campaign


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command')
    for name in ('plan','rehearse','collect'):
        s=sub.add_parser(name)
        s.add_argument('--suite',choices=campaign.SUITES,default='autonomous')
        s.add_argument('--level',choices=('gentle','standard'),default='gentle')
        s.add_argument('--repetitions',type=int,default=2,help='1..5 repeated powered suites; manual suites have their own paired repetitions.')
        s.add_argument('--seconds',type=float,default=15.,help='5..25 seconds for each timing/coupled trial.')
        if name!='plan':s.add_argument('--output',type=Path,required=True,help='New directory; never overwrite evidence.')
        if name=='collect':
            s.add_argument('--port',required=True)
            s.add_argument('--encoder-only',action='store_true',help='Powered suites only: explicitly omit camera; no independent physical cart measurement or actuator calibration.')
            s.add_argument('--execute',action='store_true',help='Authorize guarded motion AND return to the saved home. Required for powered suites.')
            s.add_argument('--camera-url',default='http://127.0.0.1:3090/api/sensing-stream?max_hz=100')
            s.add_argument('--dataset-role',choices=('training','validation','development'),default='development')
            s.add_argument('--run-label',default='')
            s.add_argument('--cart-restraint',default='',help='Required for disabled manual suites; state how physical cart position is controlled.')
            s.add_argument('--manual-timeout-s',type=float,default=1800.)
            s.add_argument('--expected-reference',type=int)
            s.add_argument('--expected-home',type=int)
    for name in ('review','bundle'):
        s=sub.add_parser(name);s.add_argument('--input',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    return p


def main(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.command is None:p.print_help();return 0
    try:
        if a.command in ('plan','rehearse','collect'):
            options=dict(suite=a.suite,level=a.level,repetitions=a.repetitions,seconds=a.seconds)
            if a.command=='plan':result=campaign.plan_suite(**options)
            elif a.command=='rehearse':result=campaign.rehearse(a.output,**options)
            else:
                result=campaign.collect_campaign(output=a.output,port=a.port,execute=a.execute,
                    dataset_role=a.dataset_role,run_label=a.run_label,camera_url=a.camera_url,
                    encoder_only=a.encoder_only,
                    cart_restraint=a.cart_restraint,manual_timeout_s=a.manual_timeout_s,
                    expected_reference=a.expected_reference,expected_home=a.expected_home,**options)
        elif a.command=='review':result=campaign.review_campaign(a.input,a.output)
        else:result=campaign.bundle_campaign(a.input,a.output)
        if a.command in ('collect','rehearse'):
            result=dict(output=str(a.output),completed=result['completed'],
                synthetic=result['synthetic'],stop_confirmed=result['stop_confirmed'],
                hardware_accepted=False,report=str(a.output/'report.json'))
        elif a.command=='review':
            result=dict(output=str(a.output),integrity_ok=result['integrity_ok'],
                analysis_usable=result['analysis_usable'],ready_for_fit=False,
                blockers=result['blockers'],report=str(a.output/'review.json'))
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except KeyboardInterrupt:
        print('Interrupted. STOP was attempted; inspect the preserved campaign report before any next run.',file=sys.stderr);return 130
    except Exception as exc:
        print(f'Error: {exc}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
