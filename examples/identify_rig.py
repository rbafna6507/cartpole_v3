#!/usr/bin/env python3
"""Offline rig-log fit. Inputs can be logs, .jsonl.gz, or ARCHIVE.tar.gz::MEMBER."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from cartpole_sdk.identification import identify_pendulum, rig_log_rows
from cartpole_sdk.recording import strict_json_loads


def load(spec):
    filename, separator, member = spec.partition('::')
    path=Path(filename)
    if separator:
        with tarfile.open(path,'r:*') as archive:
            entry=archive.getmember(member)
            if not entry.isfile(): raise ValueError('archive member must be a regular log file')
            with archive.extractfile(entry) as stream: raw=stream.read()
    else:
        raw=path.read_bytes()
        if path.suffix=='.gz':raw=gzip.decompress(raw)
    records=[strict_json_loads(line) for line in raw.decode().splitlines() if line.strip()]
    headers=[r for r in records if r.get('type')=='session']
    if len(headers)!=1: raise ValueError('one session per input required')
    run_id=headers[0]['session']['session_id']
    rows,admission=rig_log_rows(records,run_id=run_id)
    return rows,dict(path=filename,member=member if separator else None,
        sha256=hashlib.sha256(raw).hexdigest(),admission=admission)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',required=True)
    parser.add_argument('--validation',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        training,train_manifest=load(args.training)
        validation,valid_manifest=load(args.validation)
        if train_manifest['sha256']==valid_manifest['sha256']:
            raise ValueError('training and validation recordings must be independent')
        report=identify_pendulum(training,validation,source='magnetic',acceleration_kind='commanded',
            provenance=dict(inputs=dict(training=train_manifest,validation=valid_manifest),
                adapter='rig-log-v1',hardware_activity=False,
                caveat='Closed-loop data, commanded acceleration and board snapshot timestamps; not independent motion metrology.'))
        result=report.to_dict()
        result['provenance']['adapter_sha256']=hashlib.sha256(
            (Path(__file__).resolve().parents[1]/'src/cartpole_sdk/identification/rig_logs.py').read_bytes()).hexdigest()
        result['provenance']['fitter_sha256']=hashlib.sha256(
            (Path(__file__).resolve().parents[1]/'src/cartpole_sdk/identification/fit.py').read_bytes()).hexdigest()
        with args.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    except (ValueError,KeyError,TypeError,OSError,tarfile.TarError) as exc:parser.error(str(exc))
    print(json.dumps(dict(accepted=report.accepted,parameters=report.parameters,
                          reasons=report.reasons,output=str(args.output)),allow_nan=False))
    return 0 if report.accepted else 2


if __name__=='__main__':raise SystemExit(main())
