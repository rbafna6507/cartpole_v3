"""Offline adapter for Rig's JSONL, with explicit board-clock/input semantics."""
import math
from collections import Counter


def rig_log_rows(records, *, run_id):
    """Return canonical fit rows and admission counts; never contact a device.

    Only healthy, homed, enabled native auto/bal or external states are eligible.
    Other states remain invalid boundaries, rather than disappearing across a
    gap. Theta is decoded magnetic angle; acceleration is the jerk-limited MCU
    command, NOT raw demand or a measurement of physical cart acceleration.
    The filtered omega channel is deliberately omitted: its delay is unknown.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('explicit run_id required')
    session = info = previous = None
    rows, counts, duplicates = [], Counter(), 0
    host_error = False
    for record in records:
        kind = record.get('type')
        if kind == 'session':
            if session is not None:
                raise ValueError('one rig session required per recording')
            session = record['session']
            if not session.get('session_id') or not session.get('clock_id'):
                raise ValueError('session and clock identity required')
        elif kind == 'info':
            info = record['info']
            if session is None or str(info['sid']) != session['session_id']:
                raise ValueError('INFO session mismatch')
            for key, upper in (('encoder_upright_raw',4095),('home_epoch',2**32-1)):
                if type(info.get(key)) is not int or not 0 <= info[key] <= upper:
                    raise ValueError(f'invalid INFO {key}')
            if not isinstance(info.get('firmware'),str) or not info['firmware']:
                raise ValueError('firmware identity required')
        elif kind == 'error':
            host_error = True
        elif kind == 'state':
            state = record['state']
            if session is None or str(state['sid']) != session['session_id']:
                raise ValueError('STATE session mismatch')
            for key in ('unwrapped_us','unwrapped_tick','mode','flags','fault','misses'):
                if type(state.get(key)) is not int or state[key] < 0:
                    raise ValueError(f'{key} must be a nonnegative integer')
            for key in ('theta','acceleration'):
                if type(state.get(key)) not in (int,float) or not math.isfinite(state[key]):
                    raise ValueError(f'{key} must be finite')
            if previous is not None:
                if state['unwrapped_tick'] == previous['unwrapped_tick']:
                    # Receipt time can change on an exact retransmission. Every
                    # board field must agree; never choose between conflicts.
                    canonical = lambda s:{k:v for k,v in s.items() if k!='received_ns'}
                    if canonical(state) != canonical(previous):
                        raise ValueError('conflicting duplicate board tick')
                    duplicates += 1
                    continue
                if (state['unwrapped_tick'] <= previous['unwrapped_tick'] or
                        state['unwrapped_us'] <= previous['unwrapped_us']):
                    raise ValueError('board timestamps must strictly increase')
            reasons=[]
            if state['mode'] not in (2,3,12): reasons.append('outside control interval')
            if state['fault'] or state['flags'] & 7 != 7 or state['flags'] & 256:
                reasons.append('unhealthy, unhomed, disabled or calibrating')
            if previous is not None and state['misses'] != previous['misses']:
                reasons.append('control deadline counter changed')
            if host_error: reasons.append('host error latched in recording')
            if info is None:
                if not reasons: raise ValueError('eligible state has no preceding reference INFO')
                reasons.append('reference not yet available')
            calibration = 'unknown' if info is None else f"{info['firmware']}:upright:{info['encoder_upright_raw']}"
            home = 'unknown' if info is None else str(info['home_epoch'])
            row = dict(t_s=state['unwrapped_us']/1e6, theta_rad=state['theta'],
                cart_acceleration_m_s2=state['acceleration'], valid=not reasons,
                source='magnetic', acceleration_kind='commanded', run_id=run_id,
                calibration_id=calibration, clock_id=session['clock_id'],
                frame_id='encoder-upright/cart-home:'+home)
            if reasons: row['exclusion_reasons']=reasons
            rows.append(row)
            counts.update(reasons)
            previous=state
    if not rows: raise ValueError('recording contains no states')
    return rows, dict(run_id=run_id, session_id=session['session_id'], clock_id=session['clock_id'],
        total_states=len(rows), eligible_states=sum(r['valid'] for r in rows),
        excluded_states=sum(not r['valid'] for r in rows), duplicate_states=duplicates,
        exclusion_counts=dict(counts), calibration_ids=sorted({r['calibration_id'] for r in rows if r['valid']}),
        angle_field='theta', acceleration_field='acceleration',
        time_basis='board acquisition snapshot proxy; sensor/input latency not independently calibrated',
        rate_channel='omitted: filtered omega has unknown effective delay')
