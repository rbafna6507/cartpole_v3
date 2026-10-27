#!/usr/bin/env python3
"""Reusable bounded collection procedures. Import never opens hardware.

The reference is A*sin^4(pi*t/T), defaultT=.65s, followed by active cart feedback.
The hold is feedback-forced, not proof that the physical cart is fixed. Import
and --check-waveform are offline. No home, clear, calibration or SET is issued.
"""
import argparse
from dataclasses import asdict, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from cartpole_sdk.contracts import (Acceleration,Channel,ValidationError,
                                    select_channels,validate_observation)
from cartpole_sdk.control.parameters import ControllerParameters
from cartpole_sdk.rig.policy import run_policy
from .workflow import provenance,write_json,digest
from .protocol import make_protocol,validate_protocol,instructions

PULSE_S=.65
PULSE_DURATIONS=(.45,.65,.8,1.)
QUIET_PREFLIGHT_S=1.
HOLD_S=12.
COMPLETION_PAD_S=.04
AMPLITUDES_MM=(-30,-20,-10,-5,5,10,20,30)
GAP_NS=20_000_000
CHANNELS=(Channel('cart','position','step','cart-step-home'),
          Channel('cart','velocity','step','cart-step-home'),
          Channel('pole','angle','magnetic','encoder-zero'),
          Channel('pole','angular_velocity','magnetic','encoder-zero'))

def amplitude(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value not in tuple(mm/1000 for mm in AMPLITUDES_MM):
        raise ValidationError('amplitude must be exactly signed 5, 10, 20 or 30 mm')
    return value

def hold_duration(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 3<=value<=12:
        raise ValidationError('feedback hold must be3..12 finite seconds')
    return float(value)

def pulse_duration(value):
    if type(value) not in (int,float) or value not in PULSE_DURATIONS:
        raise ValidationError('pulse duration must be exactly0.45,0.65,0.8 or1.0 seconds')
    return float(value)

def validate_excitation(amplitude_m,pulse_seconds):
    amplitude(amplitude_m);pulse_seconds=pulse_duration(pulse_seconds)
    if 4*abs(amplitude_m)*(math.pi/pulse_seconds)**2>3.:
        raise ValidationError('amplitude/duration pair exceeds3m/s² reference acceleration bound')
    return pulse_seconds

def wrap(value):return math.atan2(math.sin(value),math.cos(value))

def waveform(t,amplitude_m,*,pulse_seconds=PULSE_S):
    """Analytic position, velocity, acceleration and jerk; exactly zero endpoints."""
    A=amplitude(amplitude_m);pulse_seconds=validate_excitation(A,pulse_seconds)
    if not math.isfinite(t):raise ValueError('finite waveform time required')
    if t<=0 or t>=pulse_seconds:return (0.,)*4
    w=math.pi/pulse_seconds;u=w*t;s=math.sin(u)
    return (A*s**4,A*w*(math.sin(2*u)-.5*math.sin(4*u)),
            2*A*w*w*(math.cos(2*u)-math.cos(4*u)),
            A*w**3*(-4*math.sin(2*u)+8*math.sin(4*u)))

def waveform_bounds(amplitude_m,*,pulse_seconds=PULSE_S):
    """Exact continuous reference extrema, independently checked on a 10 us grid.

    For jerk, extrema satisfy cos(2u)=(1 +/- sqrt(129))/16. These
    certify only the analytic reference; measured trajectories retain guards.
    """
    pulse_seconds=validate_excitation(amplitude_m,pulse_seconds)
    A=abs(amplitude(amplitude_m));w=math.pi/pulse_seconds
    c=((1+math.sqrt(129))/16,(1-math.sqrt(129))/16)
    jerk_factor=max(abs(4*math.sqrt(1-z*z)*(4*z-1)) for z in c)
    analytic=(A,3*math.sqrt(3)/4*A*w,4*A*w*w,jerk_factor*A*w**3)
    grid=[0.]*4
    for i in range(round(pulse_seconds*100_000)+1):
        for k,value in enumerate(waveform(i/100_000,amplitude_m,pulse_seconds=pulse_seconds)):grid[k]=max(grid[k],abs(value))
    return dict(amplitude_m=amplitude_m,pulse_s=pulse_seconds,hold_s=HOLD_S,
        grid_interval_s=.00001,analytic_abs_max=analytic,grid_abs_max=grid,
        quantities=('position_m','velocity_m_s','acceleration_m_s2','jerk_m_s3'),
        endpoint_start=waveform(0,amplitude_m,pulse_seconds=pulse_seconds),endpoint_end=waveform(pulse_seconds,amplitude_m,pulse_seconds=pulse_seconds),
        reference_only=True,physical_trajectory_guaranteed=False)

def validate_board(state):
    if not(state.healthy and state.homed and state.switch_profile) or state.flags&48:
        raise ValidationError('requires healthy homed board with both installed switches released')
    if state.mode not in (0,12):raise ValidationError('requires idle or external action mode')

def validate_preflight(state,info,*,expected_reference=None,expected_home=None):
    validate_board(state)
    if (info.firmware!='executor-v6-measuredrail' or type(info.encoder_upright_raw) is not int or
            not 0<=info.encoder_upright_raw<=4095 or not info.reference_saved or not info.encoder_conf_valid or
            type(info.home_epoch) is not int or info.home_epoch<=0):
        raise ValidationError('requires v6, saved reference, valid encoder configuration and established home')
    if expected_reference is not None and info.encoder_upright_raw!=expected_reference:
        raise ValidationError('saved reference differs from expected reference')
    if expected_home is not None and info.home_epoch!=expected_home:
        raise ValidationError('home epoch differs from expected home')
    if state.enabled or state.mode!=0 or state.brake_direction:
        raise ValidationError('requires stopped executor with settled governor')
    if (abs(state.x)>=.003 or abs(state.velocity)>=.005 or abs(state.acceleration)>=.1 or
            abs(state.omega)>=.15 or abs(wrap(state.theta-math.pi))>=.35):
        raise ValidationError('requires centered quiet cart and quietly hanging pole')

def stationary_preflight(rig,info,**pins):
    state=rig.state;rig._fresh(state);validate_preflight(state,info,**pins)
    first=state.unwrapped_us;initial=state.theta;deadline=time.monotonic()+2.
    while state.unwrapped_us-first<round(QUIET_PREFLIGHT_S*1e6):
        if time.monotonic()>=deadline:raise ValidationError('quiet preflight deadline exceeded')
        state=rig.wait_state(after_tick=state.unwrapped_tick,timeout=.1)
        rig._fresh(state);validate_preflight(state,info,**pins)
        if abs(wrap(state.theta-initial))>.03:raise ValidationError('pole changed during quiet preflight')
    return state.theta

class RingdownPolicy:
    def __init__(self,channels,*,amplitude_m,initial_hanging,record=None,clock_ns=None,pulse_seconds=PULSE_S):
        if tuple(channels)!=CHANNELS:raise ValidationError('exact magnetic/step channels required')
        if not math.isfinite(initial_hanging):raise ValidationError('finite hanging reference required')
        self.channels=tuple(channels);self.amplitude_m=amplitude(amplitude_m)
        self.pulse_seconds=validate_excitation(amplitude_m,pulse_seconds)
        self.initial_hanging=initial_hanging;self.record=record
        self.clock_ns=clock_ns if clock_ns is not None else lambda:time.monotonic_ns()
    def reset(self,session):
        self.session=session;self.origin_ns=None;self.previous=None;self.mode='pulse'
        self.clamped_commands=0;self.hold_max_abs_x=0.;self.hold_max_abs_v=0.
        self.last_elapsed_s=0.
    def _read(self,observation):
        validate_observation(self.session,observation)
        selected=select_channels(observation,self.channels,now_ns=observation.available_ns,max_age_ns=GAP_NS-1)
        if len({s.effective_ns for s in selected})!=1:raise ValidationError('requires coherent same-snapshot channels')
        return tuple(s.value for s in selected)
    def validate_start(self,observation):
        x,v,theta,omega=self._read(observation)
        if (abs(x)>=.003 or abs(v)>=.005 or abs(omega)>=.15 or
            abs(wrap(theta-self.initial_hanging))>.03 or abs(wrap(theta-math.pi))>=.35):
            raise ValidationError('start no longer centered, quiet and hanging')
    def act(self,observation):
        host_ns=self.clock_ns();x,v,theta,omega=self._read(observation);now=observation.available_ns
        if self.previous is None:self.validate_start(observation);self.origin_ns=now
        else:
            sequence,board_ns,previous_host=self.previous;board_gap=now-board_ns;host_gap=host_ns-previous_host
            if observation.sequence<=sequence or not 0<board_gap<GAP_NS or not 0<=host_gap<GAP_NS:
                raise ValidationError(f'ringdown cadence fault: seq={observation.sequence} previous={sequence} board_gap_ns={board_gap} host_gap_ns={host_gap}')
        if abs(x)>=.05 or abs(v)>=.3 or abs(wrap(theta-self.initial_hanging))>=1. or abs(omega)>=10.:
            raise ValidationError('measured/estimated trajectory exceeded ringdown guard')
        t=(now-self.origin_ns)/1e9;reference=waveform(t,self.amplitude_m,pulse_seconds=self.pulse_seconds)
        self.mode='pulse' if t<self.pulse_seconds else 'feedback_cart_hold'
        unbounded=reference[2]+80*(reference[0]-x)+14*(reference[1]-v)
        command=max(-3.,min(3.,unbounded));clamped=command!=unbounded
        self.clamped_commands+=int(clamped);self.last_elapsed_s=t
        if t>=self.pulse_seconds:
            self.hold_max_abs_x=max(self.hold_max_abs_x,abs(x));self.hold_max_abs_v=max(self.hold_max_abs_v,abs(v))
        if self.record:self.record(phase=self.mode,t_s=t,observation_sequence=observation.sequence,board_ns=now,
            host_policy_ns=host_ns,x_reference=reference[0],v_reference=reference[1],a_reference=reference[2],
            jerk_reference=reference[3],x=x,v=v,theta=theta,omega=omega,command=command,
            unclamped_command=unbounded,clamped=clamped,tail_is_independent_physical_proof=False)
        self.previous=observation.sequence,now,host_ns
        return Acceleration(command)
    def summary(self):
        return dict(clamped_commands=self.clamped_commands,fit_protocol_deviation_clamped=bool(self.clamped_commands),
            hold_max_abs_x_m=self.hold_max_abs_x,hold_max_abs_v_m_s=self.hold_max_abs_v,
            elapsed_board_s=self.last_elapsed_s,hold_type='feedback_cart_hold',physical_cart_fixed=False)

def plain(value):return asdict(value) if is_dataclass(value) else vars(value)

def run_pulse(rig,*,amplitude_m,hold_seconds=HOLD_S,pulse_seconds=PULSE_S,provenance=None,expected_reference=None,expected_home=None,check=None,ready=None):
    """Trusted bounded runner. STOP on preflight, policy, transport or wait failure."""
    failure=None;policy=None;result=None
    try:
        pulse_seconds=validate_excitation(amplitude_m,pulse_seconds);hold_seconds=hold_duration(hold_seconds)
        if rig.rate_hz!=100:raise ValidationError('ringdown requires 100 Hz telemetry')
        rig._stopped_healthy();parameters=rig.get_parameters()
        names={f.name for f in fields(ControllerParameters)}
        if len(names)!=41 or set(parameters)!=names:raise ValidationError('complete live 41-parameter catalog required')
        ControllerParameters(**parameters)
        info=rig.get_info();initial_hanging=stationary_preflight(rig,info,expected_reference=expected_reference,expected_home=expected_home)
        rig._record('ringdown_start',amplitude_m=amplitude_m,parameters=parameters,info=plain(info),
            state=plain(rig.state),initial_hanging=initial_hanging,provenance=provenance or {},rate_hz=100,
            pulse_s=pulse_seconds,quiet_preflight_s=QUIET_PREFLIGHT_S,feedback_hold_s=hold_seconds,completion_pad_s=COMPLETION_PAD_S,ttl_ms=20,
            feedback_kp=80.,feedback_kd=14.,command_limit_m_s2=3.,hardware_validated=False)
        if ready is not None:ready(dict(status='pulse_starting',amplitude_m=amplitude_m,drivers_enabled=False))
        def observe():
            if check is not None:check()
            state=rig.state;rig._fresh(state);validate_board(state)
            return rig._observation(state)
        def factory(channels):
            nonlocal policy
            policy=RingdownPolicy(channels,amplitude_m=amplitude_m,pulse_seconds=pulse_seconds,initial_hanging=initial_hanging,
                record=lambda **record:rig._record('ringdown_reference',**record))
            return policy
        # 40 ms of additional feedback hold compensates the runner's bounded
        # wall-deadline margin; it never extends the excitation waveform.
        run_policy(rig,factory,channels=CHANNELS,duration_s=pulse_seconds+hold_seconds+COMPLETION_PAD_S,
            ttl_ms=20,max_acceleration=3.,observation_provider=observe)
        if policy is None or policy.last_elapsed_s<pulse_seconds+hold_seconds:
            raise ValidationError('incomplete board-clock pulse/hold duration')
        result=dict(parameters=parameters,info=plain(info),initial_hanging=initial_hanging,
            amplitude_m=amplitude_m,pulse_seconds=pulse_seconds,hold_seconds=hold_seconds,policy=policy.summary(),hardware_validated=False)
    except BaseException as exc:
        failure=exc;rig._record('ringdown_failure',reason=repr(exc));raise
    finally:
        if policy is not None:rig._record('ringdown_summary',**policy.summary())
        try:
            rig.stop();state=rig.state
            if state.enabled:raise ValidationError('STOP did not confirm disabled executor')
            rig._record('ringdown_stop',confirmed=True,state=plain(state))
        except BaseException as exc:
            rig._record('ringdown_stop',confirmed=False,reason=repr(exc))
            if failure is None:raise
    return result


def validate_camera_finish(sample):
    """Require a fresh source frame at finish; marker loss remains missing data."""
    if not isinstance(sample,dict) or not isinstance(sample.get('vision'),dict):
        raise ValidationError('camera frame is missing at passive finish')
    vision=sample['vision'];received=sample.get('collector_received_ns');now=time.monotonic_ns()
    if type(received) is not int or not 0<=received<=now:
        raise ValidationError('camera receipt timestamp is invalid at passive finish')
    source=vision.get('t');host=sample.get('host_t');age=vision.get('age_since_host_read_s')
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in (source,host,age)):
        raise ValidationError('camera source clock and age are invalid at passive finish')
    if (type(vision.get('seq')) is not int or vision['seq']<0 or vision.get('stale') is not False or
            vision.get('worker_error') or not -.01<=host-source<=.15):
        raise ValidationError('camera source frame is stale, errored or incoherent at passive finish')
    # Source and collector clocks need not share an origin. The source age at
    # publication plus elapsed local receipt time bounds the latest frame age.
    if max(age,host-source)+(now-received)/1e9>.15:
        raise ValidationError('camera source frame exceeds 0.15 seconds at passive finish')


def run_passive(rig,*,seconds=120.,expected_reference=None,expected_home=None,check=None,ready=None,camera_read=None,finish_requested=None):
    """Manual free-release recording, always disabled; operator restrains cart."""
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 1<=seconds<=180:
        raise ValidationError('passive duration must be 1..180 seconds')
    failure=None
    try:
        rig.stop();state=rig._stopped_healthy();info=rig.get_info()
        if camera_read is None:raise ValidationError('passive collection requires camera measurements')
        validate_preflight(state,info,expected_reference=expected_reference,expected_home=expected_home)
        values=rig.get_parameters()
        if set(values)!={f.name for f in fields(ControllerParameters)}:raise ValidationError('complete live parameter catalog required')
        ControllerParameters(**values)
        state=rig.state;rig._fresh(state)
        validate_preflight(state,info,expected_reference=expected_reference,expected_home=expected_home)
        if ready is not None:ready(dict(status='measuring_hanging_baseline',drivers_enabled=False,
            seconds=2.,instruction='Keep the pole hanging freely, cart fixed, both markers visible; wait for recording.'))
        states=[plain(state)];camera_rows=[];first_host=state.received_ns;baseline_deadline=time.monotonic()+4.
        while state.received_ns-first_host<2_000_000_000:
            if time.monotonic()>=baseline_deadline:raise ValidationError('hanging baseline deadline exceeded')
            if check is not None:check()
            state=rig.wait_state(after_tick=state.unwrapped_tick,timeout=.15)
            rig._fresh(state);validate_preflight(state,info,expected_reference=expected_reference,expected_home=expected_home)
            states.append(plain(state));sample=camera_read()
            if sample is not None:
                received=sample.get('collector_received_ns')
                if type(received) is not int or not 0<=time.monotonic_ns()-received<=200_000_000:
                    raise ValidationError('camera baseline measurement is stale; keep markers visible')
                if not camera_rows or sample!=camera_rows[-1]:camera_rows.append(sample)
        from .passive_baseline import build_baseline
        baseline=build_baseline(states,camera_rows)
        rig._record('passive_baseline',baseline=baseline)
        rig._record('passive_start',info=plain(info),parameters=values,state=plain(state),seconds=seconds,
            measurement_version=2,baseline_sha256=digest(baseline),
            physical_cart_restraint='operator must independently restrain; step count does not prove physical stationarity')
        from .finish import PassiveFinish
        recording_started=time.monotonic();deadline=recording_started+seconds;next_display=recording_started
        with PassiveFinish(finish_requested) as finish:
            if ready is not None:ready(dict(status='recording',drivers_enabled=False,seconds=seconds,reference=info.encoder_upright_raw,home_epoch=info.home_epoch,
                hanging_zero_measured=True,encoder_hanging_raw_rad=baseline['encoder']['raw_angle_rad'],camera_hanging_image_rad=baseline['camera']['image_angle_rad'],
                instruction=f'Type done then Enter to finish; first Ctrl-C also finishes recording. Maximum duration: {seconds:g} seconds.'))
            while True:
                # Check evidence before accepting any requested finish. A signal
                # cannot hide a camera failure or an unhealthy/stale last state.
                if check is not None:check()
                state=rig.state;rig._fresh(state);validate_board(state)
                if state.enabled or state.mode!=0:raise ValidationError('passive recording requires disabled idle executor')
                reason=finish.poll()
                rig._require()  # A finish callback cannot bypass a latched I/O/logger fault.
                if reason is not None or time.monotonic()>=deadline:
                    validate_camera_finish(camera_read())
                    rig._require()
                    finish_reason=reason or 'duration_elapsed'
                    break
                state=rig.wait_state(after_tick=state.unwrapped_tick,timeout=.15)
                rig._fresh(state);validate_board(state)
                if state.enabled or state.mode!=0:raise ValidationError('passive recording requires disabled idle executor')
                if ready is not None and time.monotonic()>=next_display:
                    from .passive_baseline import camera_geometry
                    sample=camera_read();geometry=None
                    if sample is not None and 0<=time.monotonic_ns()-sample.get('collector_received_ns',-1)<=200_000_000:
                        geometry=camera_geometry(sample.get('vision'))
                    camera_angle=None
                    if geometry is not None and abs(geometry['length_px']/baseline['camera']['length_px']-1)<=.2:
                        camera_angle=math.degrees(-wrap(geometry['image_angle_rad']-baseline['camera']['image_angle_rad']))
                    ready(dict(status='angles',encoder_from_hanging_deg=round(math.degrees(wrap(state.theta-baseline['encoder']['raw_angle_rad'])),3),
                        camera_from_hanging_deg=None if camera_angle is None else round(camera_angle,3),
                        camera_detected=camera_angle is not None,release_angle_verified=False))
                    next_display=time.monotonic()+.5
            completion=dict(requested_seconds=float(seconds),elapsed_seconds=time.monotonic()-recording_started,finish_reason=finish_reason)
        rig._record('passive_finish',**completion)
        rig._require()
        return dict(info=plain(info),parameters=values,baseline=baseline,final_state=plain(rig.state),
            physical_cart_stationarity_certified=False,**completion)
    except BaseException as exc:
        failure=exc;rig._record('passive_failure',reason=repr(exc));raise
    finally:
        try:
            rig.stop()
            if rig.state.enabled:raise ValidationError('STOP did not confirm disabled executor')
        except BaseException as exc:
            rig._record('stop_failure',reason=str(exc))
            if failure is None:raise


def inspect_rig(rig,*,expected_reference=None,expected_home=None):
    """Read configuration in a stopped session; no arm or recovery commands."""
    failure=None
    try:
        state=rig._stopped_healthy();info=rig.get_info()
        validate_preflight(state,info,expected_reference=expected_reference,expected_home=expected_home)
        values=rig.get_parameters()
        if set(values)!={f.name for f in fields(ControllerParameters)}:raise ValidationError('complete live parameter catalog required')
        ControllerParameters(**values)
        return dict(info=plain(info),parameters=values,state=plain(state))
    except BaseException as exc:failure=exc;raise
    finally:
        try:rig.stop()
        except BaseException:
            if failure is None:raise


def sample_camera_clock(url):
    """Read-only HTTP round trip: source time to collector monotonic time."""
    from urllib.request import urlopen
    from urllib.parse import urlsplit,urlunsplit
    from cartpole_sdk.recording import strict_json_loads
    parsed=urlsplit(url)
    endpoint=urlunsplit((parsed.scheme,parsed.netloc,'/api/clock','',''))
    before=time.monotonic_ns()
    with urlopen(endpoint,timeout=.5) as response:payload=response.read(4097)
    after=time.monotonic_ns()
    if len(payload)>4096:raise ValueError('camera clock response exceeds byte limit')
    value=strict_json_loads(payload.decode()).get('monotonic_s')
    if type(value) not in (int,float) or not math.isfinite(value) or value<0 or after<before:
        raise ValueError('invalid camera clock response')
    uncertainty=(after-before+1)//2
    if uncertainty>10_000_000:raise ValueError('camera clock round-trip uncertainty exceeds10ms')
    return dict(local_before_ns=before,local_after_ns=after,server_monotonic_s=value,
                offset_ns=(before+after)//2-round(value*1e9),uncertainty_ns=uncertainty)


def record_camera(url,directory,seconds,stop,ready,sample_queue=None,sync_clocks=False):
    """Child process: bounded read-only NDJSON capture; never opens serial."""
    from urllib.request import urlopen
    from urllib.parse import urlsplit
    from cartpole_sdk.recording import strict_json_loads
    directory=Path(directory);path=directory/'vision.ndjson'
    report=dict(url=url,frames=0,heartbeat_records=0,stream_records=0,started_ns=time.monotonic_ns(),clock='camera t is a host-read proxy; round-trip clock anchors do not measure exposure latency')
    if sample_queue is not None:sample_queue.cancel_join_thread()
    try:
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc:raise ValueError('camera requires HTTP(S) URL')
        if sync_clocks:report['clock_sync']=dict(method='http-roundtrip-midpoint',start=sample_camera_clock(url))
        deadline=time.monotonic()+seconds;previous=None;previous_frame=None;previous_host=None
        with path.open('xb') as target,urlopen(url,timeout=.5) as stream:
            while not stop.is_set() and time.monotonic()<deadline:
                line=stream.readline(4*1024*1024+1)
                if not line or len(line)>4*1024*1024 or not line.endswith(b'\n'):raise ValueError('camera stream ended or exceeded record bound')
                received_ns=time.monotonic_ns();row=strict_json_loads(line.decode())
                if row.get('schema')!='cartpole-sensing-stream-v1':raise ValueError('unexpected raw camera stream schema')
                row['collector_received_ns']=received_ns
                target.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode());report['stream_records']+=1
                sequence=row.get('stream_seq');vision=row.get('vision',{});t=vision.get('t')
                frame_sequence=vision.get('seq');host_t=row.get('host_t');age=vision.get('age_since_host_read_s')
                if type(sequence) is not int or (previous is not None and sequence<=previous):raise ValueError('repeated or reversed camera stream sequence')
                if any(type(x) not in (float,int) or not math.isfinite(x) for x in (t,host_t,age)):
                    raise ValueError('camera frame lacks finite source clock and age')
                if type(frame_sequence) is not int or frame_sequence<0:raise ValueError('camera source frame identity required')
                if vision.get('stale') is not False or vision.get('worker_error') or not 0<=age<=.15 or not -.01<=host_t-t<=.15:
                    raise ValueError('camera source frame is stale, errored or has invalid source age')
                if previous_host is not None and host_t<=previous_host:raise ValueError('camera source host clock did not advance')
                current=(frame_sequence,t)
                if previous_frame is not None:
                    if current==previous_frame:report['heartbeat_records']+=1
                    elif frame_sequence<=previous_frame[0] or t<=previous_frame[1]:raise ValueError('camera source frame clock or sequence reversed')
                    else:report['frames']+=1
                else:report['frames']+=1
                if sample_queue is not None and current!=previous_frame:
                    from queue import Empty,Full
                    sample=dict(collector_received_ns=received_ns,host_t=host_t,vision=vision)
                    try:sample_queue.put_nowait(sample)
                    except Full:
                        try:sample_queue.get_nowait()
                        except Empty:pass
                        try:sample_queue.put_nowait(sample)
                        except Full:pass  # Latest-preview queue only; NDJSON above retains every record.
                previous=sequence;previous_frame=current;previous_host=host_t;ready.set()
    except BaseException as exc:
        # Setting stop only requests exit between reads; it does not cancel a
        # blocking read. A timeout or transport error is still missing evidence.
        report['failure']=repr(exc)
    finally:
        if sync_clocks and 'clock_sync' in report:
            try:report['clock_sync']['end']=sample_camera_clock(url)
            except Exception as exc:report.setdefault('failure','camera clock completion failed: '+repr(exc))
        report['finished_ns']=time.monotonic_ns()
        if path.exists():report['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(directory/'vision-report.json',report)


def _camera_worker(*args):
    """Only the spawned child ignores terminal SIGINT; its parent owns cleanup."""
    import signal
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    record_camera(*args)


class CameraRecorder:
    """Separate-process camera work cannot run in the controller's GIL thread."""
    def __init__(self,url,directory,seconds):
        import multiprocessing
        context=multiprocessing.get_context('spawn')
        self.directory=Path(directory);self.stop=context.Event();self.ready=context.Event()
        self.samples=context.Queue(maxsize=8);self._latest=None
        self.process=context.Process(target=_camera_worker,args=(url,str(directory),seconds,self.stop,self.ready,self.samples,True),daemon=True)
    def start(self):
        self.process.start()
        if not self.ready.wait(3.):raise RuntimeError('camera unavailable before collection')
        self.check()
    def check(self):
        if not self.process.is_alive():raise RuntimeError('camera recorder ended during collection')
    def latest(self):
        from queue import Empty
        while True:
            try:self._latest=self.samples.get_nowait()
            except Empty:return self._latest
    def close(self):
        self.stop.set()
        if self.process.pid is not None:
            self.process.join(2.)
            if self.process.is_alive():self.process.terminate();self.process.join(2.)
        path=self.directory/'vision-report.json'
        return json.loads(path.read_text()) if path.exists() else {'failure':'camera worker did not preserve a final report'}


DEFAULT_CAMERA_URL='http://127.0.0.1:3090/api/sensing-stream?max_hz=100'


def collect(kind,*,output,port,seconds=120.,amplitude_m=.005,hold_seconds=HOLD_S,pulse_seconds=PULSE_S,camera_url=DEFAULT_CAMERA_URL,
            expected_reference=None,expected_home=None,rig_factory=None,camera_factory=None,ready=None,protocol=None,finish_requested=None):
    """Create an immutable run directory, preserve failures, and stop before return."""
    if kind not in ('pulse','passive','inspect'):raise ValueError('unknown collection procedure')
    if kind=='passive' and camera_url is None:raise ValueError('passive collection requires camera recording and hanging baseline')
    protocol=make_protocol(kind) if protocol is None else validate_protocol(protocol,kind)
    if kind=='pulse':pulse_seconds=validate_excitation(amplitude_m,pulse_seconds);hold_seconds=hold_duration(hold_seconds)
    if not math.isfinite(seconds) or not 1<=seconds<=180:raise ValueError('duration must be 1..180 seconds')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='cartpole-id-run-v1',procedure=kind,completed=False,provenance=provenance(),
        collection_protocol=protocol,collection_protocol_sha256=digest(protocol),
        started_ns=time.monotonic_ns(),expected_reference=expected_reference,expected_home=expected_home,
        source_semantics={'fit_source':'direct Rig JSONL raw theta and MCU acceleration; never mirrored/fused packets',
            'board_clock':'unwrapped_us snapshot proxy, session-specific',
            'camera_clock':'raw vision.t host-read proxy; explicit HTTP round-trip anchors to collector monotonic clock; exposure latency unknown',
            'encoder_mirror':'current-session read-only UDP when camera enabled; display only'},
        camera={'status':'explicitly disabled; independent camera evidence absent'} if camera_url is None else {'url':camera_url},
        options=dict(port=str(port),seconds=seconds,amplitude_m=amplitude_m,pulse_seconds=pulse_seconds,hold_seconds=hold_seconds,rate_hz=100 if kind=='pulse' else 250))
    camera=None;rig=None;failure=None
    try:
        if camera_url is not None and kind!='inspect':
            camera=(camera_factory or CameraRecorder)(camera_url,output,(pulse_seconds+hold_seconds if kind=='pulse' else seconds)+15)
            camera.start()
        if rig_factory is None:
            from cartpole_sdk.rig import Rig
            rig_factory=Rig
        from cartpole_sdk.sensing.telemetry import EncoderPublisher
        publisher=EncoderPublisher() if camera is not None else None
        rig=rig_factory(port,rate_hz=100 if kind=='pulse' else 250,log_dir=output,record_capacity=8192,max_host_age_s=.02,vision_publisher=publisher)
        with rig:
            rig._record('collection_protocol',protocol=protocol,protocol_sha256=digest(protocol))
            def announce(status):
                if ready is not None:
                    ready(dict(status,collection_protocol=protocol,requested_release_instructions=instructions(protocol)) if status.get('status') in ('recording','pulse_starting') else status)
            pins=dict(expected_reference=expected_reference,expected_home=expected_home)
            check=None if camera is None else camera.check
            if kind=='pulse':report['result']=run_pulse(rig,amplitude_m=amplitude_m,hold_seconds=hold_seconds,pulse_seconds=pulse_seconds,provenance=report['provenance'],check=check,ready=announce,**pins)
            elif kind=='passive':report['result']=run_passive(rig,seconds=seconds,check=check,ready=announce,camera_read=camera.latest,finish_requested=finish_requested,**pins)
            else:report['result']=inspect_rig(rig,**pins)
            report['stop_state']=plain(rig.state)
            report['stop_confirmed']=not rig.state.enabled
        if rig.records_dropped:raise ValidationError('log records lost; collection evidence is incomplete')
        if getattr(rig,'error',None) is not None:raise RuntimeError(f'latched rig error after close: {rig.error}')
        report['completed']=True
    except BaseException as exc:
        failure=exc;report['failure']=repr(exc);report['traceback']=traceback.format_exc();raise
    finally:
        if camera is not None:
            try:
                report['camera']=camera.close()
                if report['camera'].get('failure'):
                    report['completed']=False
                    if failure is None:failure=RuntimeError('camera evidence incomplete: '+report['camera']['failure'])
            except BaseException as exc:
                report['camera']={'failure':repr(exc)};report['completed']=False
                if failure is None:failure=exc
        if rig is not None:
            report['records_dropped']=rig.records_dropped
            report['stop_records']=[r for r in rig.records if r['type'] in ('ringdown_stop','stop_failure')]
            if rig.log_path is not None and rig.log_path.exists():
                report['raw_log']=rig.log_path.name;report['raw_log_sha256']=hashlib.sha256(rig.log_path.read_bytes()).hexdigest()
        report['finished_ns']=time.monotonic_ns()
        if failure is not None:report['failure']=repr(failure)
        write_json(output/'report.json',report)
    if failure is not None:raise failure
    return report
