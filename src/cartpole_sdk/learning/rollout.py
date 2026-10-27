"""Seeded SDK-policy episodes against the provisional governed MuJoCo plant."""
from dataclasses import asdict
import math
import numpy as np
from cartpole_sdk.contracts import ActionRequest, Channel, ValidationError
from cartpole_sdk.policies import HostRatePolicy, HostPolicyParameters
from cartpole_sdk.sim.config import SimulationConfig, SensorConfig, integer
from cartpole_sdk.sim.executor import ExecutorSimulation

OBJECTIVE = ('time mean of cos(theta) - .3*(x/rail)^2 - .001*omega^2 - .0002*raw_accel^2 '
             '+ 2 if abs(theta)<.15 and abs(omega)<1 and abs(x)<.12; '
             'uncompleted horizon reward=-3; + final continuous balance seconds/horizon')
SUCCESS = 'last >=1 s continuously abs(theta)<.15 rad, abs(omega)<1 rad/s, abs(x)<.12 m; full horizon; no fault'


def initial_state(seed, start):
    integer(seed, 'seed')
    if start not in ('hanging', 'upright'):
        raise ValidationError('start must be hanging or upright')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0 if start == 'hanging' else 1]))
    return (float(rng.uniform(-.015,.015)), float(rng.uniform(-.025,.025)),
            float((math.pi if start == 'hanging' else 0.) + rng.uniform(-.07,.07)),
            float(rng.uniform(-.15,.15)))


def simulation_channels(frame_id='sim:upright+x'):
    return tuple(Channel('cart' if q in ('position','velocity') else 'pole', q,
                         'simulation', frame_id)
        for q in ('position','velocity','angle','angular_velocity'))


def rollout(parameters, *, seed, start, config, sensors, trajectory=False):
    if config.control_steps not in (4,10) or config.physics_dt_ns != 1_000_000:
        raise ValidationError('host policy evaluation requires 100 or 250 Hz with 1 ms physics')
    b = ExecutorSimulation(config, sensors)
    state = initial_state(seed, start)
    observation = b.reset(seed=seed, state=state)
    p = HostRatePolicy(simulation_channels(sensors.frame_id), rate_hz=1000//config.control_steps,
                       parameters=parameters)
    p.reset(b.session)
    score, dwell, balanced, peak_x, peak_speed, saturation_steps, rail_braking_steps = 0., 0., 0, abs(state[0]), abs(state[1]), 0, 0
    steps, fault, trace = 0, None, []
    dt = config.control_steps/1000
    try:
        for i in range(config.max_steps):
            intent = p.act(observation)
            if intent is None:
                fault = p.fault_reason or 'policy returned no request'
                break
            request = ActionRequest(observation.session_id, observation.clock_id, i,
                observation.sequence, observation.available_ns,
                observation.available_ns + min(50_000_000, 2*config.control_steps*1_000_000),
                intent.acceleration_m_s2)
            observation = b.step(request)
            steps += 1
            t = b.ground_truth
            is_balanced = abs(t.theta)<.15 and abs(t.omega)<1 and abs(t.x)<.12
            balanced += int(is_balanced)
            dwell = dwell+dt if is_balanced else 0.
            peak_x, peak_speed = max(peak_x, abs(t.x)), max(peak_speed, abs(t.v))
            score += (math.cos(t.theta) - .3*(t.x/config.rail_half_m)**2 - .001*t.omega**2
                      - .0002*intent.acceleration_m_s2**2 + 2*int(is_balanced))
            saturation_steps += int(b.diagnostics.force_saturated)
            rail_braking_steps += int(b.diagnostics.rail_braking)
            if trajectory:
                trace.append(dict(**asdict(t), raw_acceleration_m_s2=intent.acceleration_m_s2,
                    executed_acceleration_m_s2=b.diagnostics.executed_acceleration_m_s2,
                    policy_mode=p.mode, rail_braking=b.diagnostics.rail_braking))
            if b.terminated:
                fault = b.fault_reason
                break
            if b.truncated: break
        final = asdict(b.ground_truth)
    finally:
        b.close()
    horizon_s = config.max_steps*dt
    score = (score-3*(config.max_steps-steps))/config.max_steps + dwell/horizon_s
    result = dict(seed=seed, start=start, initial_state=list(state), final_state=final,
        score=float(score), success=bool(dwell>=1-1e-9 and fault is None and steps==config.max_steps),
        final_balance_dwell_s=dwell, balanced_fraction=balanced/config.max_steps,
        max_abs_x_m=peak_x, max_abs_speed_m_s=peak_speed, steps=steps, fault_reason=fault,
        force_saturated_control_steps=saturation_steps, rail_braking_control_steps=rail_braking_steps)
    if trajectory: result['trajectory'] = trace
    return result


def evaluate(parameters_or_artifact, *, seeds=(1001,1002,1003,1004),
             starts=('hanging','upright'), config=None, sensors=None, trajectory=False):
    """Held-out artifact evaluation rejects overlapping training seeds.

    Passing handwritten parameters evaluates a baseline with no training split.
    Evaluation never updates weights or feeds a reward to the optimizer.
    """
    artifact = parameters_or_artifact if isinstance(parameters_or_artifact,dict) else None
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValidationError('provide distinct nonempty evaluation seeds')
    for seed in seeds: integer(seed,'seed')
    if not starts or any(s not in ('hanging','upright') for s in starts):
        raise ValidationError('provide hanging/upright initial-state families')
    if artifact is not None:
        from .artifact import validate_artifact
        validate_artifact(artifact)
        if set(seeds) & set(artifact['training']['training_seeds']):
            raise ValidationError('held-out evaluation seeds overlap training seeds')
        parameters = HostPolicyParameters(**artifact['parameters'])
        config = config or SimulationConfig(**artifact['training']['simulation_config'])
        sensors = sensors or SensorConfig(**artifact['training']['sensor_config'])
    else:
        parameters = parameters_or_artifact
    if not isinstance(parameters,HostPolicyParameters): raise ValidationError('expected policy parameters or artifact')
    config = config or SimulationConfig(control_steps=10,max_steps=1000)
    sensors = sensors or SensorConfig(sample_every_steps=config.control_steps)
    episodes = [rollout(parameters,seed=seed,start=start,config=config,sensors=sensors,trajectory=trajectory)
                for seed in seeds for start in starts]
    return dict(model_status='provisional; hardware unvalidated', action_semantics=ExecutorSimulation.action_semantics,
        objective=OBJECTIVE, success_definition=SUCCESS, simulation_config=asdict(config), sensor_config=asdict(sensors),
        seeds=list(seeds), episodes=episodes, mean_score=float(np.mean([e['score'] for e in episodes])),
        successes=sum(e['success'] for e in episodes), episodes_count=len(episodes))
