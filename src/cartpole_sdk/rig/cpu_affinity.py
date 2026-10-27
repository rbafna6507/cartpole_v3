"""Reversible Linux camera/control CPU partitioning; standard library only."""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


_CLEANUP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class AffinityError(RuntimeError):
    pass


def _diagnostic(message, *, output=None):
    """A closed SSH pipe or PTY must not interrupt signaling or restoration."""
    try:
        print(message, file=sys.stderr if output is None else output, flush=True)
    except (OSError, ValueError):
        pass


def parse_cpus(value, *, available):
    cpus = set()
    for part in value.split(','):
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', part):
            raise AffinityError('CPU list must contain nonnegative numbers/ranges, for example 0-3,5')
        bounds = [int(item) for item in part.split('-')]
        first, last = bounds[0], bounds[-1]
        # Bound before expanding, including adversarially large ranges.
        if first > last or not available or last > max(available):
            raise AffinityError('CPU list exceeds available CPUs or contains a reversed range')
        cpus.update(range(first, last + 1))
    if not cpus or not cpus <= available:
        raise AffinityError('CPU list must be nonempty and contained in available CPUs')
    return cpus


def validate_partition(camera, control, available):
    if not control or not control <= available or camera is not None and (not camera or not camera <= available):
        raise AffinityError('CPU sets must be nonempty and contained in available CPUs')
    if camera is not None and camera & control:
        raise AffinityError('camera and control CPU sets must be disjoint')


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start: int
    parent: int


@dataclass(frozen=True)
class ThreadIdentity:
    pid: int
    process_start: int
    tid: int
    start: int


class ProcFS:
    def __init__(self, root='/proc'):
        self.root = Path(root)

    @staticmethod
    def _stat(path):
        # comm may itself contain spaces and closing parentheses.
        tail = path.read_text().rsplit(') ', 1)[1].split()
        return int(tail[1]), int(tail[19])  # PPID and starttime (fields 4/22).

    def processes(self):
        result = {}
        for directory in self.root.iterdir():
            if not directory.name.isdigit():
                continue
            try:
                parent, start = self._stat(directory / 'stat')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue  # An unrelated process may exit or be inaccessible.
            pid = int(directory.name)
            result[pid] = ProcessIdentity(pid, start, parent)
        return result

    def threads(self, process):
        try:
            directories = list((self.root / str(process.pid) / 'task').iterdir())
        except (FileNotFoundError, ProcessLookupError):
            return []
        result = []
        for directory in directories:
            if not directory.name.isdigit():
                continue
            try:
                _, start = self._stat(directory / 'stat')
            except (FileNotFoundError, ProcessLookupError):
                continue
            result.append(ThreadIdentity(process.pid, process.start, int(directory.name), start))
        return result

    def alive(self, thread):
        try:
            _, process_start = self._stat(self.root / str(thread.pid) / 'stat')
            _, thread_start = self._stat(self.root / str(thread.pid) / 'task' / str(thread.tid) / 'stat')
        except (FileNotFoundError, ProcessLookupError):
            return False
        return process_start == thread.process_start and thread_start == thread.start

    def get_affinity(self, thread):
        if not self.alive(thread):
            raise ProcessLookupError('camera process/thread exited or identity changed')
        cpus = os.sched_getaffinity(thread.tid)
        if not self.alive(thread):
            raise ProcessLookupError('camera process/thread identity changed during affinity read')
        return set(cpus)

    def set_affinity(self, thread, cpus):
        if not self.alive(thread):
            raise ProcessLookupError('camera process/thread exited or identity changed')
        os.sched_setaffinity(thread.tid, cpus)
        if self.get_affinity(thread) != set(cpus):
            raise AffinityError('affinity readback differs for camera thread '+str(thread.tid))


@contextmanager
def _ignore_cleanup_signals():
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in _CLEANUP_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


class CameraAffinity:
    """Snapshot existing threads, track descendants, and restore by identity."""
    def __init__(self, pid, cpus, *, proc=None):
        self.pid, self.cpus = pid, set(cpus)
        self.proc = ProcFS() if proc is None else proc
        self.root = None
        self.process_masks = {}  # (PID, starttime) -> original process mask.
        self.thread_masks = {}  # Complete thread identity -> original mask.

    def _discover(self, *, initial=False):
        processes = self.proc.processes()
        if initial:
            self.root = processes.get(self.pid)
            if self.root is None:
                raise AffinityError('camera MainPID is no longer running')
        seeds = set(self.process_masks)
        if self.root is not None:
            seeds.add((self.root.pid, self.root.start))
        queue = [process for process in processes.values() if (process.pid, process.start) in seeds]
        members = set()
        while queue:
            process = queue.pop(0)
            key = (process.pid, process.start)
            if key in members:
                continue
            members.add(key)
            if process.pid == os.getpid():
                raise AffinityError('camera service tree includes the launcher; refusing to change parent affinity')
            queue.extend(item for item in processes.values() if item.parent == process.pid and (item.pid, item.start) not in members)
            threads = self.proc.threads(process)
            if not threads:
                continue
            if key not in self.process_masks:
                if initial:
                    leader = next((thread for thread in threads if thread.tid == process.pid), threads[0])
                    self.process_masks[key] = self.proc.get_affinity(leader)
                else:
                    parent = processes.get(process.parent)
                    parent_key = None if parent is None else (parent.pid, parent.start)
                    original = self.process_masks.get(parent_key)
                    if original is None:
                        raise AffinityError('cannot identify original affinity of new camera descendant '+str(process.pid))
                    self.process_masks[key] = set(original)
            for thread in threads:
                if thread not in self.thread_masks:
                    try:
                        self.thread_masks[thread] = (self.proc.get_affinity(thread) if initial
                                                    else set(self.process_masks[key]))
                    except ProcessLookupError:
                        continue
        if initial and not self.thread_masks:
            raise AffinityError('camera MainPID has no live readable threads')

    def refresh(self):
        self._discover()
        for thread in self.thread_masks:
            try:
                if self.proc.alive(thread) and self.proc.get_affinity(thread) != self.cpus:
                    self.proc.set_affinity(thread, self.cpus)
            except ProcessLookupError:
                continue

    def __enter__(self):
        try:
            # Capture every existing thread before the first mutation.
            self._discover(initial=True)
            for _ in range(5):
                count = len(self.thread_masks)
                self.refresh()
                self._discover()
                if len(self.thread_masks) == count:
                    return self
            raise AffinityError('camera thread tree did not stabilize before command launch')
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def restore(self):
        if not self.thread_masks:
            return
        for _ in range(5):
            errors, changed = [], False
            try:
                self._discover()
            except Exception as error:
                errors.append('camera discovery: '+str(error))
            for thread, original in list(self.thread_masks.items()):
                try:
                    if self.proc.alive(thread) and self.proc.get_affinity(thread) != original:
                        self.proc.set_affinity(thread, original)
                        changed = True
                except ProcessLookupError:
                    continue
                except Exception as error:
                    errors.append('thread '+str(thread.tid)+': '+str(error))
            if errors:
                raise AffinityError('camera affinity restoration failed: '+'; '.join(errors))
            if not changed:
                return
        raise AffinityError('camera affinity restoration did not stabilize')

    def __exit__(self, exc_type, error, traceback):
        with _ignore_cleanup_signals():
            try:
                self.restore()
            except Exception as restoration:
                if error is not None:
                    raise AffinityError(str(error)+'; '+str(restoration)) from error
                raise


def service_main_pid(service):
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.service', service) or service.startswith('-'):
        raise AffinityError('camera service must be an explicit systemd .service unit name')
    try:
        result = subprocess.run(['systemctl', '--user', 'show', '--property=MainPID', '--value', service],
                                check=True, capture_output=True, text=True, timeout=5)
        pid = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise AffinityError('cannot read camera service MainPID: '+str(error)) from error
    if pid <= 0:
        raise AffinityError('camera service is not running; use --no-camera only for intentional control-only operation')
    return pid


def _stop_child(child, output):
    if child.poll() is not None:
        return
    with _ignore_cleanup_signals():
        for sig, timeout in ((signal.SIGINT, 10.), (signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
            _diagnostic('control child: sending '+signal.Signals(sig).name+'; waiting for exit', output=output)
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue
        raise AffinityError('control child did not exit after SIGKILL; camera affinity will still be restored')


def _forward_interrupt(child, output):
    if child.poll() is None:
        _diagnostic('SIGINT forwarded to control child; waiting for its normal exit', output=output)
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass


def run_control(command, cpus, *, camera=None, output=None):
    """Set affinity only in a separate child; retain parent mask and terminal IO."""
    output = sys.stderr if output is None else output
    if not command:
        raise AffinityError('a control command after -- is required')
    child = None
    try:
        # Avoid losing the child handle to a terminal signal during Popen.
        try:
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, set(_CLEANUP_SIGNALS))
            try:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--_child',
                                          ','.join(str(cpu) for cpu in sorted(cpus)), '--', *command],
                                         start_new_session=True)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
        except KeyboardInterrupt:
            if child is None:
                raise
            _forward_interrupt(child, output)
        while True:
            try:
                if camera is not None:
                    camera.refresh()
                try:
                    return child.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            except KeyboardInterrupt:
                # The interactive console stops its policy and returns to its
                # prompt. SIGINT must not start a launcher termination timer.
                _forward_interrupt(child, output)
    except BaseException:
        if child is not None:
            _stop_child(child, output)
        raise


class _Terminated(BaseException):
    def __init__(self, signum=signal.SIGTERM):
        self.signum = signum


def _terminate(signum, frame):
    raise _Terminated(signum)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    camera_options = parser.add_mutually_exclusive_group(required=True)
    camera_options.add_argument('--camera-service', help='current user systemd camera service to temporarily restrict')
    camera_options.add_argument('--no-camera', action='store_true', help='explicit control-only operation')
    parser.add_argument('--camera-cpus', help='camera CPUs, e.g. 0-3; required with --camera-service')
    parser.add_argument('--control-cpus', required=True, help='disjoint control CPUs, e.g. 4-5')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='-- COMMAND [ARG ...]')
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a control command after -- is required')
    if bool(args.camera_service) != bool(args.camera_cpus):
        parser.error('--camera-service requires --camera-cpus; --no-camera does not accept --camera-cpus')
    try:
        if not sys.platform.startswith('linux') or not hasattr(os, 'sched_getaffinity'):
            raise AffinityError('jetson-control requires Linux CPU affinity and /proc')
        available = set(os.sched_getaffinity(0))
        control = parse_cpus(args.control_cpus, available=available)
        camera_cpus = None if args.no_camera else parse_cpus(args.camera_cpus, available=available)
        validate_partition(camera_cpus, control, available)
        camera = None if args.no_camera else CameraAffinity(service_main_pid(args.camera_service), camera_cpus)
        previous_handlers = {sig: signal.signal(sig, _terminate) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            with nullcontext() if camera is None else camera:
                _diagnostic('control CPUs '+str(sorted(control))+('; camera CPUs '+str(sorted(camera_cpus)) if camera else '; explicit no-camera mode'))
                code = run_control(command, control, camera=camera)
            if camera is not None:
                _diagnostic('camera affinity restored')
            return code if code >= 0 else 128-code
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    except KeyboardInterrupt:
        _diagnostic('interrupted before control supervision; camera cleanup completed')
        return 130
    except _Terminated as error:
        _diagnostic('termination requested; control shutdown and camera cleanup completed')
        return 128+error.signum
    except (AffinityError, OSError, subprocess.SubprocessError) as error:
        _diagnostic('jetson-control: '+str(error))
        return 1


def _child(argv):
    try:
        cpus = parse_cpus(argv[0], available=set(os.sched_getaffinity(0)))
        if argv[1:2] != ['--'] or len(argv) < 3:
            raise AffinityError('invalid internal control command')
        os.sched_setaffinity(0, cpus)
        if set(os.sched_getaffinity(0)) != cpus:
            raise AffinityError('control child affinity readback mismatch')
        signal.pthread_sigmask(signal.SIG_UNBLOCK, set(_CLEANUP_SIGNALS))
        os.execvp(argv[2], argv[2:])
    except (AffinityError, OSError) as error:
        _diagnostic('jetson-control child: '+str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(_child(sys.argv[2:]) if sys.argv[1:2] == ['--_child'] else main())
