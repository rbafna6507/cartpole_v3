"""Cross-entropy episodic policy search over a disclosed handwritten prior.

No expert action targets are used. Candidate gains receive only mean episodic
reward on the fixed training split. This is model-free parameter-search RL in a
provisional simulator, not a learned dynamics model or hardware-trained policy.
"""
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import sys
import numpy as np
from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.policies import HostPolicyParameters
from cartpole_sdk.sim.config import SimulationConfig, SensorConfig, integer
from cartpole_sdk.sim.executor import ExecutorSimulation
from cartpole_sdk.sim.model import model_xml
from .rollout import evaluate, OBJECTIVE, SUCCESS

# Log-space positive weights. Fixed capture guards/limits remain disclosed priors.
SEARCH_BOUNDS = {
    'energy_gain':(.5,12.), 'swing_position_gain':(5.,100.),
    'swing_velocity_gain':(.2,12.), 'phase_softness':(.3,5.),
    'theta_gain':(12.,40.), 'omega_gain':(.8,5.),
    'position_gain':(.5,10.), 'velocity_gain':(.5,10.),
    'kick_acceleration_m_s2':(.2,4.), 'kick_duration_s':(.06,.3),
}


def provenance(config):
    root = Path(__file__).resolve().parents[1]
    names = ('contracts.py','policies/__init__.py','policies/host.py','learning/__init__.py','learning/search.py','learning/rollout.py',
             'learning/artifact.py','sim/executor.py','sim/backend.py','sim/config.py','sim/model.py',
             'control/motion.py','control/parameters.py','control/tuned_parameters.json')
    return dict(command=list(sys.argv), python=platform.python_version(),
        dependencies={name:version(name) for name in ('numpy','mujoco')},
        code_sha256={name:sha256((root/name).read_bytes()).hexdigest() for name in names},
        model_sha256=sha256(model_xml(config).encode()).hexdigest(),
        model_status='provisional, not physically identified or accepted',
        prior='handwritten energy shaping and linear balance feedback initialized from saved gains; all structure disclosed',
        actuator='1 kHz binary32 governor; perfect acceleration servo with force cap; no motor/slip/ISR/disabled coasting model')


def train(*, seed=20261013, training_seeds=(11,12,13), generations=4, population=10,
          config=None, sensors=None, progress=None):
    for value,name in ((seed,'seed'),(generations,'generations'),(population,'population')):
        integer(value,name,positive=name!='seed')
    if population<4 or generations>100 or population>1000:
        raise ValidationError('require population 4..1000 and generations 1..100')
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValidationError('provide distinct nonempty training seeds')
    for trial_seed in training_seeds: integer(trial_seed,'training seed')
    config = config or SimulationConfig(control_steps=10,max_steps=1000)
    sensors = sensors or SensorConfig(sample_every_steps=config.control_steps)
    if config.control_steps not in (4,10): raise ValidationError('train at 100 or 250 Hz')
    training_provenance = provenance(config)
    rng = np.random.default_rng(seed)
    baseline = HostPolicyParameters()
    names = list(SEARCH_BOUNDS)
    lower,upper = np.log(np.array(list(SEARCH_BOUNDS.values()))).T
    mean = np.log([getattr(baseline,name) for name in names])
    std = np.full(len(names),.28)
    best = baseline
    baseline_evaluation = evaluate(baseline,seeds=training_seeds,config=config,sensors=sensors)
    best_score = baseline_evaluation['mean_score']
    history = []
    for generation in range(generations):
        vectors = np.clip(rng.normal(mean,std,(population-1,len(names))),lower,upper)
        candidates = [best]+[replace(baseline,**dict(zip(names,np.exp(vector).tolist()))) for vector in vectors]
        scores = [best_score]+[evaluate(p,seeds=training_seeds,config=config,sensors=sensors)['mean_score']
                              for p in candidates[1:]]
        elite_indices = np.argsort(-np.array(scores),kind='stable')[:max(2,population//4)]
        elite_vectors = np.log([[getattr(candidates[i],name) for name in names] for i in elite_indices])
        mean = .3*mean+.7*np.mean(elite_vectors,axis=0)
        std = np.maximum(.06,.3*std+.7*np.std(elite_vectors,axis=0))
        winner = int(elite_indices[0])
        best,best_score = candidates[winner],float(scores[winner])
        entry = dict(generation=generation+1,best_score=best_score,candidate_scores=[float(s) for s in scores],
                     elite_indices=[int(i) for i in elite_indices],parameters=asdict(best))
        history.append(entry)
        if progress is not None: progress(entry)
    return dict(schema=1,policy_family='structured_energy_linear_v1',parameters=asdict(best),
        rate_hz=1000//config.control_steps, action_semantics=ExecutorSimulation.action_semantics,
        hardware_accepted=False, provenance=training_provenance, training=dict(
            algorithm='cross_entropy_policy_search', seed=seed, training_seeds=list(training_seeds),
            generations=generations,population=population,search_bounds=SEARCH_BOUNDS,
            objective=OBJECTIVE,success_definition=SUCCESS,simulation_config=asdict(config),sensor_config=asdict(sensors),
            baseline_parameters=asdict(baseline),pretraining_score=baseline_evaluation['mean_score'],
            pretraining_episodes=baseline_evaluation['episodes'],history=history,best_score=best_score))
