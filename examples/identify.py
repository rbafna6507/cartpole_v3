#!/usr/bin/env python3
"""Offline fit: independent input files, explicit adapters, reviewable JSON."""
import argparse
import hashlib
import json
from pathlib import Path
from cartpole_sdk.identification import FitConfig, identify_pendulum, read_rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--format',choices=('rows-jsonl','csv'),required=True)
    parser.add_argument('--column',action='append',default=[],metavar='SDK_FIELD=CSV_COLUMN')
    parser.add_argument('--source',choices=('magnetic','vision','fused','simulation'),default='magnetic')
    parser.add_argument('--acceleration-kind',choices=('commanded','measured'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length-bounds-m',type=float,nargs=2,default=(.05,1.5))
    parser.add_argument('--max-prediction-rmse-rad',type=float,default=.08)
    args=parser.parse_args(argv)
    if args.training.resolve()==args.validation.resolve():parser.error('training and validation must be separate independent files')
    columns={}
    for item in args.column:
        key,separator,value=item.partition('=')
        if not separator or not key or not value or key in columns:parser.error('--column requires distinct SDK_FIELD=CSV_COLUMN mappings')
        columns[key]=value
    try:
        train=read_rows(args.training,format=args.format,columns=columns)
        valid=read_rows(args.validation,format=args.format,columns=columns)
        config=FitConfig(length_bounds_m=tuple(args.length_bounds_m),max_prediction_rmse_rad=args.max_prediction_rmse_rad)
        report=identify_pendulum(train,valid,source=args.source,acceleration_kind=args.acceleration_kind,config=config,
                                provenance={'training_file':str(args.training.resolve()),'validation_file':str(args.validation.resolve())})
        result=report.to_dict()
        for key,path in (('training',args.training),('validation',args.validation)):
            result['provenance'][key+'_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    except (ValueError,KeyError,OSError,TypeError) as exc:parser.error(str(exc))
    print(f"{'accepted' if report.accepted else 'rejected'}: {args.output}")
    return 0 if report.accepted else 2


if __name__=='__main__':raise SystemExit(main())
