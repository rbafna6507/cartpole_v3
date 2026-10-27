"""Trusted host policy runner. Untrusted submissions belong in a separate broker.

Factories are explicit Python callables supplied by trusted operator code; no
module/path imports or user code loading occur here. The preserved 1kHz reference
policy is not selected automatically at a slower telemetry rate.
"""
import math
import time
from cartpole_sdk.contracts import (Acceleration, ActionRequest, Channel,
                                    ValidationError, select_channels)

class ZeroAccelerationPolicy:
    """Bounded commissioning example; it does not swing up or balance."""
    def __init__(self, channels):
        self.channels=channels
    def reset(self, session):
        pass
    def act(self, observation):
        select_channels(observation,self.channels,now_ns=observation.available_ns,max_age_ns=20_000_000)
        return Acceleration(0.)

def _timing_snapshot(policy):
    """Optional boundary-only diagnostics must never obscure a control fault."""
    try:
        snapshot=getattr(policy,'timing_diagnostics',None)
        return snapshot() if callable(snapshot) else None
    except Exception as exc:
        return {'diagnostics_error':type(exc).__name__}


def run_policy(rig, policy_factory, *, channels: tuple[Channel,...], duration_s=1.,
               ttl_ms=20, max_acceleration=.25, observation_provider=None,
               channel_max_age_ns=None):
    """Explicitly arm and execute a trusted policy with selected sensor channels.

    Observation and request clocks are the board clock. This runner is paced by
    fresh telemetry and bounds wall duration; firmware still owns lease expiry.
    Every return/exception stops in finally. None does not renew the last action.
    The default commissioning ceiling is intentionally small and configurable
    only by trusted operator code, within the executor's absolute limit.
    """
    rig._duration(duration_s)
    if not callable(policy_factory):
        raise ValidationError('policy_factory must be an explicit trusted callable')
    if type(channels) is not tuple or not channels or not all(isinstance(c,Channel) for c in channels):
        raise ValidationError('policy requires explicit source/frame channels')
    ages = (20_000_000,) * len(channels) if channel_max_age_ns is None else channel_max_age_ns
    if (type(ages) is not tuple or len(ages) != len(channels) or
            any(type(age) is not int or not 0 < age <= 500_000_000 for age in ages)):
        raise ValidationError('one explicit positive bounded effective-age limit per channel required')
    if any(age > 20_000_000 and channel.source != 'vision' for channel, age in zip(channels, ages)):
        raise ValidationError('cart, magnetic and fused channels retain the 20 ms freshness bound')
    def selected(observation):
        return tuple(select_channels(observation, (channel,), now_ns=observation.available_ns,
                                     max_age_ns=age)[0] for channel, age in zip(channels, ages))
    if type(ttl_ms) is not int or not 1<=ttl_ms<=50:
        raise ValidationError('TTL must be 1..50 integer milliseconds')
    if type(max_acceleration) not in (int,float) or not math.isfinite(max_acceleration) or not 0<max_acceleration<=25:
        raise ValidationError('commissioning acceleration bound must be positive and <=25')
    if rig.rate_hz == 0 or 1000/rig.rate_hz >= ttl_ms:
        raise ValidationError('telemetry interval must be shorter than anchored action TTL')
    observe = rig.observation if observation_provider is None else observation_provider
    if not callable(observe):
        raise ValidationError("observation_provider must be callable")
    failure=None;policy=None
    calls=0; compute_ns=0; max_compute_ns=0; last_mode=None
    # Keep only primitive failure context during normal operation; emit it once
    # on failure so a rejected batched frame can be identified without tick logs.
    stage='fetch'; observation=None
    previous_sent_sequence=previous_sent_board_ns=None
    fetch_started_host_ns=fetch_finished_host_ns=None
    act_started_host_ns=act_finished_host_ns=None
    try:
        fetch_started_host_ns=time.monotonic_ns()
        observation=observe()
        fetch_finished_host_ns=time.monotonic_ns()
        selected(observation)
        stage='act'
        policy=policy_factory(channels)
        if not callable(getattr(policy,'reset',None)) or not callable(getattr(policy,'act',None)):
            raise ValidationError('factory must return a Policy')
        policy.reset(rig.session)
        if callable(getattr(policy,"validate_start",None)):
            policy.validate_start(observation)
        rig._record('policy_start',channels=[dict(entity=c.entity,quantity=c.quantity,source=c.source,frame_id=c.frame_id) for c in channels],
                    policy=type(policy).__name__,duration_s=duration_s,ttl_ms=ttl_ms,max_acceleration=max_acceleration,
                    channel_max_age_ns=list(ages),
                    timing=_timing_snapshot(policy))
        stage='step'
        rig.arm(duration_s)
        deadline=rig._deadline_ns/1e9-min(.01,duration_s*.25)
        sequence=0
        while time.monotonic()<deadline:
            stage='fetch'; observation=None
            fetch_finished_host_ns=act_started_host_ns=act_finished_host_ns=None
            fetch_started_host_ns=time.monotonic_ns()
            observation=observe()
            fetch_finished_host_ns=time.monotonic_ns()
            selected(observation)
            stage='act'
            began=time.perf_counter_ns()
            act_started_host_ns=time.monotonic_ns()
            try:
                intent=policy.act(observation)
            finally:
                act_finished_host_ns=time.monotonic_ns()
            elapsed=time.perf_counter_ns()-began
            calls+=1;compute_ns+=elapsed;max_compute_ns=max(max_compute_ns,elapsed)
            mode=getattr(policy,"mode",None)
            if mode!=last_mode:
                rig._record("policy_transition", mode=mode, observation_sequence=observation.sequence, board_ns=observation.available_ns)
                last_mode=mode
            if intent is None and getattr(policy, 'fault_reason', None):
                raise ValidationError(f'host policy fault: {policy.fault_reason}')
            if intent is not None:
                if not isinstance(intent,Acceleration) or abs(intent.acceleration_m_s2)>max_acceleration:
                    raise ValidationError('policy output exceeds commissioning bound or is not Acceleration')
                sequence+=1
                request=ActionRequest(observation.session_id,observation.clock_id,sequence,observation.sequence,
                                      observation.available_ns,observation.available_ns+ttl_ms*1_000_000,
                                      intent.acceleration_m_s2)
                stage='step'
                rig.step(request)
                previous_sent_sequence=observation.sequence
                previous_sent_board_ns=observation.available_ns
            remaining=deadline-time.monotonic()
            if remaining<=0:
                break
            stage='wait'
            try:
                rig.wait_state(after_tick=observation.sequence,timeout=min(remaining,.1))
            except TimeoutError:
                if time.monotonic()>=deadline:
                    break
                raise
    except BaseException as exc:
        failure=exc
        rig._record('policy_failure',reason=str(exc),stage=stage,timing=_timing_snapshot(policy),
            failed_observation_sequence=getattr(observation,'sequence',None),
            failed_observation_board_ns=getattr(observation,'available_ns',None),
            previous_sent_observation_sequence=previous_sent_sequence,
            previous_sent_observation_board_ns=previous_sent_board_ns,
            fetch_started_host_ns=fetch_started_host_ns,
            fetch_finished_host_ns=fetch_finished_host_ns,
            act_started_host_ns=act_started_host_ns,
            act_finished_host_ns=act_finished_host_ns)
        raise
    finally:
        try:
            rig.stop()
        except Exception as exc:
            rig._record('stop_failure',reason=str(exc))
            if failure is None:
                raise
        finally:
            rig._record('policy_end',calls=calls,mean_compute_ns=compute_ns/calls if calls else None,
                        max_compute_ns=max_compute_ns,timing=_timing_snapshot(policy))


def run_host_controller(rig, *, mode, duration_s=10., max_gap_periods=None):
    """Experimental sensed-executor AUTO/BAL using the stopped live readback.

    The MCU owns observer, pulse generation and the single 1 kHz motion governor.
    Python owns energy shaping, capture, balance and rail-return decisions at the
    telemetry cadence. No calibration, parameter write or rehome is implicit.
    Optional max_gap_periods=2/3 permits bounded 250 Hz observation gaps; the
    default remains strict. All configurations retain the 20 ms anchored lease.
    """
    from dataclasses import fields
    from cartpole_sdk.control.parameters import ControllerParameters
    from cartpole_sdk.policies.executor_feedback import ExecutorFeedbackPolicy
    if mode not in ('auto','bal') or rig.rate_hz not in (100,250):
        raise ValidationError('host controller requires auto/bal and 100 or 250 Hz telemetry')
    rig._duration(duration_s)
    rig._stopped_healthy()
    values=rig.get_parameters()
    if set(values)!={f.name for f in fields(ControllerParameters)}:
        raise ValidationError('host controller requires the complete v6 live parameter catalog')
    parameters=ControllerParameters(**values)
    channels=(Channel('cart','position','step','cart-step-home'),
              Channel('cart','velocity','step','cart-step-home'),
              Channel('pole','angle','magnetic_filtered','encoder-zero'),
              Channel('pole','angular_velocity','magnetic','encoder-zero'),
              Channel('cart','acceleration','step','cart-step-home'),
              Channel('cart','brake_direction','step','cart-step-home'))
    rig._record('host_controller',mode=mode,rate_hz=rig.rate_hz,max_gap_periods=max_gap_periods,
                controller_parameters=values,policy='ExecutorFeedbackPolicy',
                raw_safety_channel=dict(entity='pole',quantity='angle',source='magnetic',frame_id='encoder-zero'),
                hardware_validated=False)
    run_policy(rig,lambda selected:ExecutorFeedbackPolicy(selected,rate_hz=rig.rate_hz,
               parameters=parameters,start_mode=mode,max_gap_periods=max_gap_periods,
               raw_angle_channel=Channel('pole','angle','magnetic','encoder-zero')),channels=channels,
               observation_provider=rig.control_observation,duration_s=duration_s,
               ttl_ms=20,max_acceleration=max(parameters.amax_s,parameters.amax_b))
