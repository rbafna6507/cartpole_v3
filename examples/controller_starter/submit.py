"""Explicit authenticated submission; physical jobs still require operator approval."""
import argparse
import json
import os
from pathlib import Path
from cartpole_sdk.portal.client import PortalClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,default=Path(__file__).with_name('policy.py'))
    parser.add_argument('--source',choices=('simulation','magnetic-step'),default='simulation')
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--key',required=True,help='Stable idempotency key for this exact request')
    parser.add_argument('--reservation-id')
    parser.add_argument('--recording',choices=('standard','detailed'),default='standard')
    parser.add_argument('--collect',type=Path,help='Wait for run and verified bundle; destination must be new')
    parser.add_argument('--timeout',type=float,default=180.)
    args=parser.parse_args()
    origin=os.environ.get('CARTPOLE_PORTAL_URL');token=os.environ.get('CARTPOLE_SUBMIT_TOKEN')
    if not origin or not token:parser.error('set CARTPOLE_PORTAL_URL and CARTPOLE_SUBMIT_TOKEN')
    client=PortalClient(origin,token)
    job=client.submit(args.artifact,sensor_source=args.source,duration=args.seconds,
                      idempotency_key=args.key,reservation_id=args.reservation_id,recording=args.recording)
    print(json.dumps({'id':job['id'],'status':job['status'],'artifact':job['artifact']}),flush=True)
    if args.collect is not None:
        manifest=client.collect(job['id'],args.collect,timeout=args.timeout)
        print(json.dumps({'bundle':str(args.collect),'status':manifest['status'],'recording':manifest['recording']}))


if __name__=='__main__':main()
