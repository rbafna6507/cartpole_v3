"""Local native viewer for numeric actors and the saved reference controller.

Learned mode uses the actual 100 Hz RigCartpoleEnv and numeric SDK actor. Reference
mode uses the same fitted plant with the 1 kHz ReferencePolicy, saved gains and
ideal sensors; its output is already governed, so no second governor is applied.
Imports create no window and no hardware connection. GUI state is display-only.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from queue import SimpleQueue
import time

from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.learning.model_experiment import digest, read_json, validate_experiment
from cartpole_sdk.learning.rollout import initial_state
from cartpole_sdk.policies.neural import NeuralPolicy, load_neural_policy
from cartpole_sdk.policies.numeric_backend import NumpyActorBackend
from cartpole_sdk.policies.rig_observation import contract_from_config, simulation_channels
from .backend import MujocoBackend
from .config import integer
from .rig_gym_env import RigCartpoleEnv
from .viewer import ViewerSession, copy_display_state, _draw_rail, _positive_duration

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACTOR = ROOT/'policies/working.json'
DEFAULT_EXPERIMENT = ROOT/'evidence/policy-recovery-20261021/combined-next/protocol.json'


class NumericViewerPolicy(NeuralPolicy):
    def __init__(self, actor, *, channels):
        super().__init__(actor, channels=channels)
        self.numeric = NumpyActorBackend(actor)

    def predict_encoded(self, encoded):
        return self.numeric.predict_encoded(encoded)


class LearnedViewerSession(ViewerSession):
    period_s = .01
    controller_name = 'learned actor / 100 Hz'

    def __init__(self, env, actor, *, start, seed):
        self.env = env
        self.backend = env.backend.plant
        self.policy = NumericViewerPolicy(actor, channels=env.channels)
        self.sensor_description = env.observation_config.angle_source+' / trained sensor model'
        self.seed = seed
        self.start = start
        self.initial_state = initial_state(seed, start)
        self._commands = SimpleQueue()
        self.exit_requested = False
        self.reset()

    @property
    def decisions(self):
        return self._action_sequence

    def reset(self):
        self.env.reset(seed=self.seed, options={'state': self.initial_state})
        self.policy.reset(self.env.session)
        self.paused = False
        self.end_reason = self.fault_reason = None
        self._action_sequence = 0
        if self.env.terminated:
            self._halt('plant_fault', self.env.fault_reason)

    def advance(self, steps):
        integer(steps, 'steps')
        if self.paused or self.end_reason or self.exit_requested:
            return 0
        completed = 0
        for _ in range(steps):
            intent = self.policy.act(self.env.sdk_observation)
            if intent is None:
                self._halt('policy_fault', self.policy.fault_reason)
                break
            try:
                self.env.step([intent.acceleration_m_s2])
            except (ValueError, RuntimeError) as exc:
                self._halt('plant_fault', str(exc))
                break
            self._action_sequence += 1
            completed += 1
            if self.env.terminated or self.env.truncated:
                self._halt('plant_fault' if self.env.terminated else 'horizon', self.env.fault_reason)
                break
        return completed

    def close(self):
        self.env.close()


class ReferenceViewerSession(ViewerSession):
    period_s = .001
    controller_name = 'traditional reference / 1000 Hz'
    sensor_description = 'ideal simulation sensors'

    @property
    def decisions(self):
        return self._action_sequence

    def close(self):
        self.backend.close()


def create_session(*, actor_path=DEFAULT_ACTOR, experiment_path=DEFAULT_EXPERIMENT,
                   controller='learned', start='hanging', seed=98400, seconds=25.):
    integer(seed, 'seed')
    if controller not in ('learned', 'reference') or start not in ('hanging', 'upright'):
        raise ValidationError('choose learned/reference controller and hanging/upright start')
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or not .01 <= seconds <= 60
            or not math.isclose(seconds*100, round(seconds*100), abs_tol=1e-8, rel_tol=0.)):
        raise ValidationError('seconds must be .01..60 in whole 100 Hz decisions')
    document, _ = read_json(experiment_path)
    experiment = document.get('experiment', document)
    config, parameters, observations, _ = validate_experiment(experiment)
    identity = deepcopy(experiment)
    identity['admission']['model'].pop('source_path', None)
    if controller == 'learned':
        verified = load_neural_policy(actor_path, channels=simulation_channels(angle_source=observations.angle_source))
        actor = verified.actor
        if actor['experiment_sha256'] != digest(identity):
            raise ValidationError('actor experiment hash differs from selected experiment')
        if actor['model_content_sha256'] != experiment['admission']['model']['content_sha256']:
            raise ValidationError('actor model differs from selected experiment')
        if actor['executor_parameters'] != asdict(parameters):
            raise ValidationError('actor executor snapshot differs from selected experiment')
        if actor['observation_contract'] != contract_from_config(asdict(observations)):
            raise ValidationError('actor observation contract differs from selected experiment')
        scale = min(config.max_acceleration_m_s2, max(parameters.amax_s, parameters.amax_b))
        if actor['action_scale_m_s2'] != scale:
            raise ValidationError('actor action scale differs from selected experiment')
        env = RigCartpoleEnv(replace(config, max_steps=round(seconds*100)),
            parameters=parameters, observation_config=observations, ttl_ms=experiment['ttl_ms'])
        try:
            session = LearnedViewerSession(env, actor, start=start, seed=seed)
        except BaseException:
            env.close()
            raise
    else:
        backend = MujocoBackend(replace(config, control_steps=1, max_steps=round(seconds*1000)))
        try:
            session = ReferenceViewerSession(backend, start_mode='auto' if start == 'hanging' else 'upright',
                seed=seed, state=initial_state(seed, start), parameters=parameters)
            session.start = start
        except BaseException:
            backend.close()
            raise
    session.experiment = experiment
    return session


def summary(session):
    return dict(controller=session.controller_name, start=session.start, seed=session.seed,
        sensors=session.sensor_description, decisions=session.decisions,
        end_reason=session.end_reason, fault_reason=session.fault_reason,
        ground_truth=asdict(session.backend.ground_truth), hardware_accepted=False,
        actor_sha256=getattr(session.policy, 'actor', {}).get('content_sha256'))


def _overlay(viewer, session):
    import mujoco
    truth = session.backend.ground_truth
    left = 'Controller\nStart / seed\nSensors\nSimulation time\nPole angle\nCart position\nStatus\nControls'
    right = (f'{session.controller_name}\n{session.start} / {session.seed}\n{session.sensor_description}\n'
             f'{truth.time_ns/1e9:.2f} s\n{math.degrees(truth.theta):.2f} deg\n{truth.x:.3f} m\n'
             f'{session.end_reason or ("paused" if session.paused else "running")}\nSpace: pause | R: reset | Esc: exit')
    if session.fault_reason:
        left += '\nFault'; right += '\n'+session.fault_reason
    # set_texts can wait on the render thread: never call it with viewer.lock held.
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))


def run_viewer(session, *, duration=None):
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_xml_string(session.backend.model_xml)
    data = mujoco.MjData(model)
    copy_display_state(session, model, data)
    with mujoco.viewer.launch_passive(model, data, key_callback=session.key_callback,
            show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [0., 0., 0.]
            viewer.cam.distance = 1.1
            viewer.cam.azimuth, viewer.cam.elevation = 90., -12.
            _draw_rail(viewer, session.backend.config.rail_half_m)
        _overlay(viewer, session)
        viewer.sync()
        started = previous = time.monotonic()
        accumulator = 0.
        last_status = None
        while viewer.is_running() and not session.exit_requested:
            now = time.monotonic()
            if duration is not None and now-started >= duration:
                break
            before, paused = session.decisions, session.paused
            session.apply_pending_commands()
            if session.decisions < before or session.paused != paused or session.paused or session.end_reason:
                accumulator = 0.
            else:
                accumulator += min(now-previous, .1)
            previous = now
            ticks = int((accumulator+1e-12)/session.period_s)
            session.advance(ticks)
            accumulator -= ticks*session.period_s
            with viewer.lock():
                copy_display_state(session, model, data)
            _overlay(viewer, session)
            viewer.sync()
            status = (session.end_reason, session.fault_reason)
            if session.end_reason and status != last_status:
                print(json.dumps(summary(session)), flush=True)
                print('Episode frozen; press R to reset.', flush=True)
            last_status = status
            time.sleep(max(0., 1/60-(time.monotonic()-now)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', choices=('learned', 'reference'), default='learned')
    parser.add_argument('--actor', type=Path, default=DEFAULT_ACTOR)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT,
                        help='Exact training experiment JSON or protocol containing experiment.')
    parser.add_argument('--start', choices=('hanging', 'upright'), default='hanging')
    parser.add_argument('--seed', type=int, default=98400)
    parser.add_argument('--seconds', type=float, default=25., help='Simulation horizon, .01..60 (default 25).')
    parser.add_argument('--duration', type=_positive_duration, help='Close window after this many wall-clock seconds.')
    parser.add_argument('--headless', action='store_true', help='Run same episode without opening a window.')
    args = parser.parse_args(argv)
    try:
        session = create_session(actor_path=args.actor, experiment_path=args.experiment,
            controller=args.controller, start=args.start, seed=args.seed, seconds=args.seconds)
        try:
            if args.headless:
                session.advance(round(args.seconds/session.period_s))
            else:
                print('Local simulation only. Space: pause/resume | R: reset same seed | Esc: close.', flush=True)
                print('Horizon/fault freezes until R; native mouse physics edits affect display only.', flush=True)
                run_viewer(session, duration=args.duration)
            print(json.dumps(summary(session)), flush=True)
        finally:
            session.close()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'policy viewer: {exc}\n')
    return 0


def launch():
    """Use MuJoCo's main-thread launcher on macOS; headless/help need no GUI."""
    import os
    import sys
    if sys.platform == 'darwin' and not any(a in sys.argv[1:] for a in ('--help', '-h', '--headless')):
        launcher = Path(sys.executable).with_name('mjpython')
        if not launcher.is_file():
            raise RuntimeError('Install the sim extra to obtain mjpython')
        os.execv(str(launcher), [str(launcher), '-m', __name__, *sys.argv[1:]])
    return main()


if __name__ == '__main__':
    raise SystemExit(main())
