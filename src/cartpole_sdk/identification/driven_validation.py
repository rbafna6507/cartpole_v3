"""Offline, encoder-only prediction conditioned on commanded acceleration.

No fitting or hardware access. The fixed pole law does not turn MCU command
states into physical acceleration measurements. Phase baselines are explicit
sensitivities, and every half-second prediction reinitializes from past data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from . import campaign
from .workflow import digest


@dataclass(frozen=True)
class FrozenDrivenModel:
    effective_length_m: float = .160303873
    angular_damping_per_s: float = .006891975
    coulomb_rad_s2: float = 1.6268943
    friction_width_rad_s: float = .05
    gravity_m_s2: float = 9.81
    commanded_input_gain: float = 1.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if min(self.effective_length_m, self.friction_width_rad_s, self.gravity_m_s2) <= 0:
            raise ValueError('length, friction width and gravity must be positive')


def causal_encoder_rate(t, angle, index, samples=7):
    """Quadratic derivative at the last of seven past/current raw samples."""
    t, angle = np.asarray(t, float), np.asarray(angle, float)
    if type(index) is not int or index < samples-1 or index >= len(t):
        raise ValueError('causal rate needs seven past/current samples')
    tx = t[index-samples+1:index+1]-t[index]
    values = angle[index-samples+1:index+1]
    if not np.isfinite(tx).all() or not np.isfinite(values).all() or np.any(np.diff(tx) <= 0):
        raise ValueError('causal rate requires finite increasing sample history')
    return float(np.polyfit(tx, np.unwrap(values), 2)[-2])


def advance(q, omega, dt, commanded_acceleration, model):
    """RK4 step at one fixed left-held command value; q has chosen phase removed."""
    def rhs(angle, rate):
        acceleration = ((model.gravity_m_s2*math.sin(angle)-
                         model.commanded_input_gain*commanded_acceleration*math.cos(angle)) /
                        model.effective_length_m-model.angular_damping_per_s*rate-
                        model.coulomb_rad_s2*math.tanh(rate/model.friction_width_rad_s))
        return rate, acceleration
    q1, w1 = rhs(q, omega)
    q2, w2 = rhs(q+dt*q1/2, omega+dt*w1/2)
    q3, w3 = rhs(q+dt*q2/2, omega+dt*w2/2)
    q4, w4 = rhs(q+dt*q3, omega+dt*w3)
    return q+dt*(q1+2*q2+2*q3+q4)/6, omega+dt*(w1+2*w2+2*w3+w4)/6


def _errors(values, windows):
    x = np.asarray(values, float)
    return dict(prediction_windows=windows, prediction_samples=len(x),
                prediction_rmse_rad=float(np.sqrt(np.mean(x*x))) if len(x) else None,
                prediction_p95_rad=float(np.quantile(np.abs(x), .95)) if len(x) else None)


def score_trajectory(t_s, theta_rad, commanded_acceleration_m_s2, recorded_omega_rad_s,
                     *, model=FrozenDrivenModel(), phase_offset_rad=0., start_index=6,
                     phases=None, max_gap_s=.02):
    """Fixed 0.5 s conditional predictions; no window crosses a missing sample.

    Future *recorded inputs* are supplied as forcing, not predicted. Initial
    state uses only samples at/before each forecast start; the primary never
    consumes recorded omega. No angle, rate or input gaps are interpolated.
    """
    t, angle, acceleration, recorded = [np.asarray(x, float) for x in
        (t_s, theta_rad, commanded_acceleration_m_s2, recorded_omega_rad_s)]
    if (t.ndim != 1 or len(t) < 8 or any(x.shape != t.shape or not np.isfinite(x).all()
            for x in (t, angle, acceleration, recorded))):
        raise ValueError('finite matching time/encoder/command/omega vectors required')
    if np.any(np.diff(t) <= 0):
        raise ValueError('trajectory timestamps must strictly increase')
    if type(max_gap_s) not in (int, float) or not math.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError('positive finite continuity bound required')
    if np.any(np.diff(t) >= max_gap_s-1e-10):
        raise ValueError('trajectory gap reaches continuity bound; no interpolation permitted')
    if type(phase_offset_rad) not in (int, float) or not math.isfinite(phase_offset_rad):
        raise ValueError('explicit finite phase offset required')
    if type(start_index) is not int or not 6 <= start_index < len(t):
        raise ValueError('forecast start needs causal history within the same trial')
    phases = ['unclassified']*len(t) if phases is None else list(phases)
    if len(phases) != len(t) or any(not isinstance(p, str) for p in phases):
        raise ValueError('one phase label per sample required')
    raw = np.unwrap(angle); q = raw-phase_offset_rad
    errors = {'causal_encoder': [], 'recorded_omega': []}; phase_errors = {}
    windows = []; i = start_index
    while i+8 < len(t):
        end = int(np.searchsorted(t, t[i]+.5+1e-10, side='right'))
        if t[end-1]-t[i] < .49-1e-10:
            break
        causal = causal_encoder_rate(t, raw, i)
        states = {'causal_encoder': (float(q[i]), causal),
                  'recorded_omega': (float(q[i]), float(recorded[i]))}
        window = dict(start_s=float(t[i]), end_s=float(t[end-1]),
                      initial_encoder_angle_rad=float(raw[i]),
                      causal_initial_omega_rad_s=causal,
                      recorded_initial_omega_rad_s=float(recorded[i]), points=[])
        for j in range(i+1, end):
            duration = t[j]-t[j-1]; steps = max(1, math.ceil(duration/.001)); h = duration/steps
            point = dict(t_s=float(t[j]), measured_encoder_rad=float(raw[j]), phase=phases[j],
                         commanded_acceleration_m_s2=float(acceleration[j-1]))
            for name, state in states.items():
                for _ in range(steps):
                    state = advance(*state, h, float(acceleration[j-1]), model)
                states[name] = state
                predicted = state[0]+phase_offset_rad
                error = math.atan2(math.sin(predicted-raw[j]), math.cos(predicted-raw[j]))
                point[name+'_prediction_rad'] = float(predicted)
                point[name+'_error_rad'] = error
                errors[name].append(error)
                phase_errors.setdefault(phases[j], {}).setdefault(name, []).append(error)
            window['points'].append(point)
        windows.append(window); i = end
    scores = {name: _errors(value, len(windows)) for name, value in errors.items()}
    span = float(np.ptp(raw[start_index:])); rms_input = float(np.sqrt(np.mean(acceleration[start_index:]**2)))
    primary = scores['causal_encoder']; reasons = []
    if len(windows) < 3: reasons.append('fewer than three half-second forecasts')
    if span < .15: reasons.append('encoder angle span below 0.15 rad')
    if rms_input < .05: reasons.append('commanded acceleration RMS below 0.05 m/s2')
    if (primary['prediction_rmse_rad'] is None or primary['prediction_rmse_rad'] > .08 or
            primary['prediction_p95_rad'] > .15):
        reasons.append('causal forecast errors exceed 0.08 rad RMS or 0.15 rad p95')
    return dict(**scores, windows=windows, angle_span_rad=span,
        commanded_acceleration_rms_m_s2=rms_input, forecast_checks_pass=not reasons,
        forecast_check_reasons=reasons, hardware_accepted=False,
        by_phase={phase: {name:_errors(values, None) for name,values in groups.items()}
                  for phase,groups in phase_errors.items()},
        initialization='primary: seven past/current encoder samples, quadratic endpoint derivative; comparator: current recorded firmware omega',
        horizon_s=.5, max_gap_s=max_gap_s, input_reconstruction='left-held MCU commanded acceleration; zero modeled delay',
        causal_initialization=True, physical_acceleration_measured=False)


def _states(records):
    states = []
    for row in records:
        if row.get('type') != 'state': continue
        state = row['state']
        for name in ('unwrapped_us', 'received_ns', 'theta', 'omega', 'acceleration'):
            value = state.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('missing/nonfinite state '+name)
        if states:
            if state['unwrapped_us'] < states[-1]['unwrapped_us']:
                raise ValueError('reversed board timestamps')
            if state['unwrapped_us'] == states[-1]['unwrapped_us']:
                if {k:v for k,v in state.items() if k != 'received_ns'} != {
                        k:v for k,v in states[-1].items() if k != 'received_ns'}:
                    raise ValueError('contradictory duplicate board snapshot')
                continue
            if state['received_ns'] <= states[-1]['received_ns']:
                raise ValueError('reversed/duplicate state receipt timestamps')
        states.append(state)
    if not states or len({s.get('sid') for s in states}) != 1 or states[0].get('sid') is None:
        raise ValueError('one nonempty board session required')
    return states


def _trial(report, records, trial, model, fixed_phase_offset_rad):
    base = dict(id=trial['id'], kind=trial.get('kind'), reference=trial.get('trial'),
                completed=trial.get('completed') is True)
    if not base['completed']:
        return dict(base, status='excluded_incomplete', failure=trial.get('failure'),
                    reason='failed/incomplete trial is never model validation')
    if trial.get('kind') not in ('actuator', 'coupled', 'timing'):
        return dict(base, status='excluded_kind', reason='not a driven encoder trial')
    try:
        rows = [r for r in records if trial['started_ns'] <= r.get('host_monotonic_ns', -1) <= trial['finished_ns']]
        starts = [r for r in rows if r.get('type') in ('driven_start','driven_coupled_start')]
        if len(starts) != 1: raise ValueError('exactly one driven start marker required')
        start = starts[0]; hanging = start.get('initial_hanging')
        if type(hanging) not in (int,float) or not math.isfinite(hanging):
            raise ValueError('finite preceding hanging reading required for explicit sensitivity')
        stops = [r for r in rows if r.get('type') == 'driven_stop' and
                 r['host_monotonic_ns'] > start['host_monotonic_ns']]
        if not stops or stops[0].get('confirmed') is not True:
            raise ValueError('matching confirmed driven STOP required')
        before_stop = [r for r in rows if r['host_monotonic_ns'] < stops[0]['host_monotonic_ns']]
        states = _states(before_stop)
        allowed = {12} if trial['kind'] in ('actuator','timing') else {2,3,5,6,7,8,9,10,11}
        active = [i for i,s in enumerate(states) if s['received_ns'] >= start['host_monotonic_ns'] and
                  s.get('flags',0)&2 and s.get('mode') in allowed]
        if not active: raise ValueError('no enabled driven trajectory')
        first = active[0]; end = first
        while end < len(states) and states[end].get('flags',0)&2 and states[end].get('mode') in allowed:
            end += 1
        # Stop/return transitions end support, never bridge into another phase.
        left = max(0, first-10); selected = states[left:end]; first -= left
        if first < 6: raise ValueError('insufficient causal encoder history before driven start')
        for s in selected:
            if s.get('fault') != 0 or s.get('flags',0)&13 != 13 or s.get('flags',0)&48:
                raise ValueError('unhealthy/unhomed/switch-active state in scored support')
        for name in ('misses','dropped'):
            values = [s.get(name) for s in selected]
            if any(type(v) is not int for v in values) or any(v != values[0] for v in values):
                raise ValueError(name+' counter changed or unavailable in scored support')
        ts = np.array([s['unwrapped_us'] for s in selected],float)/1e6
        ts -= ts[first]
        phase = []
        ref = trial.get('trial') or {}
        excitation = ref.get('segment_s',0.)*(3 if ref.get('kind')=='reversal' else 2)
        for t,s in zip(ts,selected):
            phase.append('native_mode_'+str(s['mode']) if trial['kind']=='coupled' else
                         'driven_excitation' if t < excitation else 'feedback_cart_hold')
        baselines = dict(firmware_reference=0., local_hanging_anchor=math.atan2(math.sin(hanging-math.pi), math.cos(hanging-math.pi)))
        if fixed_phase_offset_rad is not None: baselines['fixed_training_phase'] = fixed_phase_offset_rad
        args = (ts, [s['theta'] for s in selected], [s['acceleration'] for s in selected], [s['omega'] for s in selected])
        scores = {name:dict(phase_offset_rad=offset,
                    scope=('pretrial resting equilibrium sensitivity; friction may displace rest from true vertical'
                           if name=='local_hanging_anchor' else 'explicit fixed phase; global encoder calibration unproven'),
                    scores=score_trajectory(*args, model=model, phase_offset_rad=offset,start_index=first,phases=phase))
                  for name,offset in baselines.items()}
        return dict(base, status='scored', initial_hanging_rad=hanging, source_ticks=[selected[first]['tick'],selected[-1]['tick']],
                    active_duration_s=float(ts[-1]), phase_baselines=scores,
                    max_board_gap_s=float(np.max(np.diff(ts))),
                    max_host_receipt_gap_s=float(np.max(np.diff([s['received_ns'] for s in selected]))/1e9),
                    phase_anchor_source_host_ns=start['host_monotonic_ns'],
                    protocol=trial.get('result',{}).get('policy',{}), return_home_scored=False)
    except (ValueError, KeyError, TypeError) as exc:
        return dict(base,status='rejected_continuity_or_admission',reason=str(exc))


def _read_campaign(directory, expected_role, model, fixed_phase_offset_rad, allow_partial):
    root = Path(directory); integrity,reason = campaign.verify(root)
    if not integrity: raise ValueError(reason)
    report = campaign._read_json(root/'report.json')
    if report.get('schema') != campaign.SCHEMA: raise ValueError('campaign schema required')
    if report.get('dataset_role') != expected_role: raise ValueError('declared dataset role differs from requested role')
    if report.get('synthetic'): raise ValueError('synthetic campaign cannot be physical evidence')
    partial = report.get('completed') is not True
    if partial and not allow_partial: raise ValueError('incomplete campaign requires explicit partial diagnostics')
    if any(report.get(k) for k in ('records_dropped','pending_log_records','timing_record_errors')):
        raise ValueError('loss-free sealed recording required')
    if not partial and (report.get('stop_confirmed') is not True or report.get('failure')):
        raise ValueError('completed campaign requires confirmed final STOP and no failure')
    raw = report.get('raw_log')
    if not isinstance(raw,str) or Path(raw).name != raw: raise ValueError('local raw log basename required')
    records = list(campaign._read_lines(root/raw))
    return dict(name=root.name,declared_dataset_role=expected_role,
                source_manifest_sha256=campaign._hash(root/'manifest.json'),source_raw_sha256=campaign._hash(root/raw),
                source_report_sha256=campaign._hash(root/'report.json'),sensor_mode=report.get('sensor_mode'),
                initial_info=report.get('initial_info'),live_parameters=report.get('parameters'),
                campaign_completed=not partial,partial_diagnostics=partial,failure=report.get('failure'),
                stop_confirmed=report.get('stop_confirmed') is True,
                trials=[_trial(report,records,t,model,fixed_phase_offset_rad) for t in report.get('trials',[])])


def analyze_driven_campaigns(training_dirs=(), validation_dirs=(), development_dirs=(), *,
                             model=FrozenDrivenModel(), fixed_phase_offset_rad=None,
                             allow_partial=False, output=None, plot=True):
    """Score fixed parameters in separately declared roles; no model selection."""
    sources = [Path(p) for p in (*training_dirs,*validation_dirs,*development_dirs)]
    if not sources: raise ValueError('at least one sealed campaign required')
    if fixed_phase_offset_rad is not None and (type(fixed_phase_offset_rad) not in (float,int) or not math.isfinite(fixed_phase_offset_rad)):
        raise ValueError('finite explicit fixed phase required')
    if type(allow_partial) is not bool: raise ValueError('allow_partial must be explicit boolean')
    if output is not None:
        output = Path(output)
        if any(output.resolve()==p.resolve() or p.resolve() in output.resolve().parents for p in sources):
            raise ValueError('analysis output must be outside sealed inputs')
        if output.exists(): raise FileExistsError(output)
    runs = {role:[_read_campaign(p,role,model,fixed_phase_offset_rad,allow_partial) for p in paths]
            for role,paths in [('training',training_dirs),('validation',validation_dirs),('development',development_dirs)]}
    hashes = [r['source_raw_sha256'] for group in runs.values() for r in group]
    if len(hashes)!=len(set(hashes)): raise ValueError('duplicate raw source across declared datasets')
    result = dict(schema='cartpole-driven-frozen-prediction-v1',parameters_fitted=False,
                  frozen_model=asdict(model),frozen_model_sha256=digest(asdict(model)),
                  model_law='qdd=(g*sin(q)-commanded_input_gain*a_mcu*cos(q))/L-gamma*omega-c*tanh(omega/friction_width), q=encoder-phase',
                  fixed_phase_offset_rad=fixed_phase_offset_rad,phase_selected_from_scores=False,
                  hardware_accepted=False,recommended_model=None,physical_acceleration_measured=False,
                  global_encoder_phase_calibrated=False,analysis_source_sha256=campaign._hash(__file__),**runs,
                  limitations=[
                      'MCU acceleration is a command-state proxy; unit input gain and zero delay are assumptions, not measured actuator properties.',
                      'Left-held 100Hz input omits intersample 1kHz governor ramps. Input history over each forecast is conditioned on recorded commands.',
                      'Every 0.5s forecast reinitializes from past raw encoder data; full-episode open-loop accuracy is not established.',
                      'Firmware-reference and pertrial local-hanging sensitivities are both retained; resting equilibrium may be shifted by friction. Neither establishes global angle calibration.',
                      'Recorded firmware omega is a comparator only; the primary estimate uses seven past/current encoder samples.',
                      'Pertrial forecast checks do not replace original full-model residual/calibration/physical-validity gates. Partial campaigns remain partial diagnostics.',
                      'No vision is loaded; physical cart tracking, masses, force authority, input delay and metric actuator gain remain unmeasured.'])
    if output is not None:
        output.mkdir(parents=True,exist_ok=False)
        (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        if plot: _draw(result,output)
    return result


def _draw(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for role in ('training','validation','development'):
        for number,run in enumerate(result[role]):
            for index,trial in enumerate(run['trials']):
                if trial['status']!='scored': continue
                fig,axes=plt.subplots(len(trial['phase_baselines']),1,figsize=(11,6),squeeze=False,layout='constrained')
                for ax,(name,baseline) in zip(axes[:,0],trial['phase_baselines'].items()):
                    for wi,window in enumerate(baseline['scores']['windows']):
                        points=window['points'];t=[p['t_s'] for p in points]
                        for key,label,style in [('measured_encoder_rad','Encoder','k.'),
                            ('causal_encoder_prediction_rad','Causal encoder initial rate','C0-'),
                            ('recorded_omega_prediction_rad','Recorded omega initial rate','C1--')]:
                            ax.plot(t,[p[key] for p in points],style,label=label if wi==0 else None,markersize=2)
                    ax.set(title=name+' — conditional on MCU commanded acceleration',xlabel='Board seconds from active start',ylabel='Raw encoder angle (rad)')
                    ax.grid(alpha=.2);ax.legend()
                fig.suptitle(f'{role}: {run["name"]}/{trial["id"]}; 0.5s reinitialized windows')
                fig.savefig(output/f'{role}-{number+1}-{index+1}-prediction.png',dpi=140)
                plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for role in ('training','validation','development'):
        parser.add_argument('--'+role,type=Path,action='append',default=[])
    parser.add_argument('--fixed-phase-offset-rad',type=float)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args(argv)
    result=analyze_driven_campaigns(args.training,args.validation,args.development,
        fixed_phase_offset_rad=args.fixed_phase_offset_rad,allow_partial=args.allow_partial,
        output=args.output,plot=not args.no_plot)
    print(json.dumps(dict(output=str(args.output),parameters_fitted=False,hardware_accepted=False,
        trials={role:[{'campaign':r['name'],'id':t['id'],'status':t['status']} for r in result[role] for t in r['trials']]
                for role in ('training','validation','development')}),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
