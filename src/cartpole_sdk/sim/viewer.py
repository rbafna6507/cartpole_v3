"""Native passive viewer for the real SDK reference policy and MuJoCo plant.

Importing this module creates no window. The GUI receives a display-only mirror:
its native reset, keyframes, perturbations and option edits cannot change the
backend's authoritative controller, sensor clock or plant configuration.
"""
import argparse
import math
from queue import Empty, SimpleQueue
import time

from cartpole_sdk import ActionRequest, Channel, ValidationError
from cartpole_sdk.control.policy import ReferencePolicy
from .backend import MujocoBackend
from .config import SimulationConfig, integer


class ViewerSession:
    """Headless, fixed-cadence loop; all commands are applied on its owner thread."""

    def __init__(self, backend: MujocoBackend, *, start_mode: str = 'auto', seed: int = 0,
                 state: tuple[float, float, float, float] | None = None, parameters=None):
        if backend.config.physics_dt_ns * backend.config.control_steps != ReferencePolicy.period_ns:
            raise ValidationError('reference viewer requires a 1 ms control interval')
        if backend.config.max_action_ttl_ns < ReferencePolicy.period_ns:
            raise ValidationError('control interval exceeds the configured action lifetime limit')
        self.backend = backend
        self.policy = ReferencePolicy(
            Channel('pole', 'angle', 'simulation', backend.sensors.frame_id),
            Channel('cart', 'position', 'simulation', backend.sensors.frame_id),
            start_mode=start_mode, parameters=parameters)
        self.seed = seed
        self.initial_state = state if state is not None else (
            (0., 0., math.pi, 0.) if start_mode == 'auto' else (0., 0., .04, 0.))
        self._commands = SimpleQueue()
        self.exit_requested = False
        self.reset()

    def reset(self):
        """Explicitly reset the plant, source stream and controller together."""
        self.backend.reset(seed=self.seed, state=self.initial_state)
        self.policy.reset(self.backend.session)
        self.paused = False
        self.end_reason = self.fault_reason = None
        self._action_sequence = 0

    def key_callback(self, keycode: int):
        # MuJoCo invokes this on its UI thread; never mutate the plant here.
        if keycode in (32, ord('R'), ord('r'), 256):
            self._commands.put(keycode)

    def apply_pending_commands(self):
        while True:
            try:
                key = self._commands.get_nowait()
            except Empty:
                break
            if key == 32:
                self.paused = not self.paused
            elif key in (ord('R'), ord('r')):
                self.reset()
            elif key == 256:  # GLFW_KEY_ESCAPE
                self.exit_requested = True

    def _halt(self, reason, fault=None):
        self.end_reason, self.fault_reason = reason, fault
        self.backend.data.qfrc_applied[:] = 0.
        self.backend.data.xfrc_applied[:] = 0.

    def advance(self, steps: int) -> int:
        """Run each 1 ms policy tick, even when several ticks share one frame."""
        integer(steps, 'steps')
        completed = 0
        if self.paused or self.end_reason or self.exit_requested:
            return completed
        for _ in range(steps):
            observation = self.backend.observation
            intent = self.policy.act(observation)
            if self.policy.fault_reason:
                self._halt('policy_fault', self.policy.fault_reason)
                break
            request = ActionRequest(self.backend.session.session_id, self.backend.session.clock_id,
                self._action_sequence, observation.sequence, observation.available_ns,
                observation.available_ns + self.policy.period_ns, intent.acceleration_m_s2)
            self._action_sequence += 1
            try:
                self.backend.step(request)
            except RuntimeError as exc:
                self._halt('plant_fault', str(exc))
                break
            completed += 1
            if self.backend.terminated:
                self._halt('rail_limit')
                break
            if self.backend.truncated:
                self._halt('horizon')
                break
        return completed


def copy_display_state(session: ViewerSession, model, data):
    """Overwrite all native GUI state edits from the authoritative SDK plant."""
    import mujoco
    mujoco.mj_copyData(data, model, session.backend.data)
    mujoco.mj_forward(model, data)


def _draw_rail(viewer, half_length):
    """Add visual rail and end markers without changing the physics model."""
    import mujoco
    import numpy as np
    scene = viewer.user_scn
    scene.ngeom = 0
    for size, position, color in (
        ([half_length, .006, .006], [0., 0., -.035], [.5, .6, .65, 1.]),
        ([.005, .026, .035], [-half_length, 0., -.035], [.95, .65, .2, 1.]),
        ([.005, .026, .035], [half_length, 0., -.035], [.95, .65, .2, 1.]),
    ):
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
            np.array(size), np.array(position), np.eye(3).reshape(9), np.array(color, dtype=np.float32))
        scene.ngeom += 1


def _set_overlay(viewer, session):
    import mujoco
    mode = session.policy.last_output.mode if session.policy.last_output else 'starting'
    status = session.end_reason or ('paused' if session.paused else 'running')
    truth = session.backend.ground_truth
    left = 'SDK reference\nSource\nController mode\nSimulation time\nStatus\nControls\nModel'
    right = (f'{session.policy.start_mode}\nsimulation\n{mode}\n{truth.time_ns / 1e9:.3f} s\n'
             f'{status}\nSpace: pause | R: reset | Esc: exit\nprovisional surrogate')
    if session.fault_reason:
        left += '\nFault'
        right += f'\n{session.fault_reason}'
    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                     mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))


def run_viewer(*, start_mode='auto', seed=0, duration=None):
    """Open the native window; 30 s episodes remain frozen until explicit R."""
    import mujoco
    import mujoco.viewer

    backend = MujocoBackend(SimulationConfig(max_steps=30_000))
    try:
        session = ViewerSession(backend, start_mode=start_mode, seed=seed)
        display_model = mujoco.MjModel.from_xml_string(backend.model_xml)
        display_data = mujoco.MjData(display_model)
        copy_display_state(session, display_model, display_data)
        print('MuJoCo SDK simulation; simulation channels; provisional surrogate.', flush=True)
        print('Space: pause/resume | R: reset same seed | Esc/window close: exit.\n'
              'Rail, fault or 30 s horizon freezes until R.', flush=True)
        with mujoco.viewer.launch_passive(display_model, display_data,
                key_callback=session.key_callback, show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = [0., 0., 0.]
                viewer.cam.distance = .8
                viewer.cam.azimuth = 90.
                viewer.cam.elevation = -12.
                _draw_rail(viewer, backend.config.rail_half_m)
            # Overlay submission may wait for the render thread. Never hold
            # the viewer mutex while that thread consumes a pending overlay.
            _set_overlay(viewer, session)
            viewer.sync()
            started = previous = time.monotonic()
            accumulator = 0.
            last_status = None
            while viewer.is_running() and not session.exit_requested:
                frame_started = time.monotonic()
                if duration is not None and frame_started - started >= duration:
                    break
                was_sequence = backend.observation.sequence
                was_paused = session.paused
                session.apply_pending_commands()
                # A reset or pause transition discards elapsed paused/reset wall time.
                changed = backend.observation.sequence < was_sequence or session.paused != was_paused
                if changed or session.paused or session.end_reason:
                    accumulator = 0.
                else:
                    # Bound catch-up after a window drag/stall. Simulate every retained
                    # tick; lag slows wall playback rather than skipping policy samples.
                    accumulator += min(frame_started - previous, .1)
                previous = frame_started
                ticks = int((accumulator + 1e-12) / .001)
                session.advance(ticks)
                accumulator -= ticks * .001
                with viewer.lock():
                    copy_display_state(session, display_model, display_data)
                _set_overlay(viewer, session)
                viewer.sync()  # sync acquires its own lock; do not nest it above.
                status = (session.end_reason, session.fault_reason)
                if session.end_reason and status != last_status:
                    print(f'Frozen: {session.end_reason}' +
                          (f': {session.fault_reason}' if session.fault_reason else '') +
                          '. Press R to reset.', flush=True)
                last_status = status
                time.sleep(max(0., 1. / 60. - (time.monotonic() - frame_started)))
    finally:
        backend.close()


def _positive_duration(value):
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError('duration must be a positive finite number of seconds')
    return duration


def main(argv=None):
    parser = argparse.ArgumentParser(description='Interactive SDK reference controller in MuJoCo.')
    parser.add_argument('--start', choices=('auto', 'upright'), default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--duration', type=_positive_duration, help='close after this many wall seconds')
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    try:
        run_viewer(start_mode=args.start, seed=args.seed, duration=args.duration)
    except RuntimeError as exc:
        if 'mjpython' in str(exc):
            parser.exit(2, 'macOS native viewer requires mjpython. Run:\n'
                        '  .venv/bin/mjpython examples/view_simulation.py\n')
        raise
    return 0


def launch():
    """Console entry point selecting MuJoCo's macOS main-thread launcher."""
    import os
    from pathlib import Path
    import sys
    if sys.platform == 'darwin':
        launcher = Path(sys.executable).with_name('mjpython')
        if not launcher.is_file():
            raise RuntimeError('Install the simulation extra to obtain mjpython')
        os.execv(str(launcher), [str(launcher), '-m', __name__, *sys.argv[1:]])
    return main()


if __name__ == '__main__':
    raise SystemExit(main())
