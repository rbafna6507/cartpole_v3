"""Explicit adapters; no inference of columns, units, angle sign or acceleration."""
import csv
import json
from pathlib import Path
from ..recording import strict_json_loads


def read_rows(path, *, format, columns=None):
    path=Path(path)
    if format=='rows-jsonl':
        return [strict_json_loads(line) for line in path.read_text().splitlines() if line.strip()]
    if format!='csv': raise ValueError('format must be csv or rows-jsonl; use sdk_run_rows for SDK run records')
    if not columns or not {'t_s','theta_rad','cart_acceleration_m_s2'}<=columns.keys():
        raise ValueError('CSV columns require explicit t_s, theta_rad and cart_acceleration_m_s2 mapping')
    result=[]
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        if any(name not in (reader.fieldnames or []) for name in columns.values()): raise ValueError('mapped columns missing from CSV header')
        for row in reader:
            item={}
            for key,column in columns.items():
                raw=row[column]
                if key=='valid':
                    if raw not in ('true','false','1','0'):raise ValueError('valid CSV field must be true/false/1/0')
                    item[key]=raw in ('true','1')
                elif key in ('t_s','theta_rad','omega_rad_s','cart_acceleration_m_s2'):item[key]=float(raw) if raw else None
                else:item[key]=raw
            result.append(item)
    return result


def sdk_run_rows(records, *, angle_channel, acceleration_kind, acceleration_field, time_field,
                 time_scale_s, run_id, calibration_id, clock_id):
    """Flatten explicit SDK run dictionaries with observation + acceleration.

    Each record has an ``observation`` dictionary matching SDK Observation,
    plus caller-selected top-level acceleration/time fields. Acceleration cannot
    be recovered from base recording v1, which records observations only. Caller
    must export/join action or measured-acceleration records on a calibrated clock.
    """
    if acceleration_kind not in ('commanded','measured'):raise ValueError('declare acceleration_kind')
    if not clock_id or not run_id or not calibration_id:raise ValueError('explicit run, clock and calibration identities required')
    result=[]
    for record in records:
        observation=record['observation']
        if observation['clock_id']!=clock_id:raise ValueError('SDK observation clock mismatch')
        samples=[s for s in observation['samples'] if all(s[k]==v for k,v in angle_channel.items())]
        if len(samples)!=1:raise ValueError('angle channel must select exactly one source/frame sample')
        sample=samples[0]
        if sample['quantity']!='angle' or sample['unit']!='rad':raise ValueError('SDK channel must be angle in rad')
        t=record[time_field]*time_scale_s
        effective=sample['effective_ns']
        if effective is None or abs(effective/1e9-t)>1e-8:raise ValueError('angle and acceleration effective time mismatch; align explicitly before fitting')
        result.append(dict(t_s=t,theta_rad=sample['value'],cart_acceleration_m_s2=record[acceleration_field],
                           valid=sample['validity']=='valid',source=sample['source'],frame_id=sample['frame_id'],
                           acceleration_kind=acceleration_kind,run_id=run_id,calibration_id=calibration_id,clock_id=clock_id))
    return result
