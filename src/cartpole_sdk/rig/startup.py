"""Console-only startup profiles. Importing the SDK never changes board tuning."""
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
import shlex
from .parameters import read_profile, TUNABLE_FIRMWARE, TUNABLE_FIRMWARES
from cartpole_sdk.contracts import ValidationError

DEFAULT_PROFILE_NAME = 'tuned-20261016-bw10'
_DEFAULT_COMMANDS = (None, 'auto', 'bal', 'python-auto', 'python-bal')
_MOTION_COMMANDS = frozenset(('auto','bal','balance','manual','velocity','home','rehome','python-auto','python-auto-trial','python-bal'))


class StartupParameters:
    """Apply once while stopped; leave faulted sessions available for diagnosis."""
    def __init__(self, *, command=None, profile=None, keep_board=False):
        self.custom = profile
        self.selected = not keep_board and (profile is not None or command in _DEFAULT_COMMANDS)
        self.pending = self.selected
        self.path = None
        self.label = str(profile) if profile is not None else DEFAULT_PROFILE_NAME
        self._resources = ExitStack()

    def __enter__(self):
        try:
            if self.selected:
                self.path = (Path(self.custom) if self.custom is not None else
                    self._resources.enter_context(as_file(files('cartpole_sdk.control').joinpath(DEFAULT_PROFILE_NAME+'.json'))))
                read_profile(self.path)  # Validate before opening a serial device.
            return self
        except BaseException:
            self._resources.close()
            raise

    def __exit__(self, *exc):
        return self._resources.__exit__(*exc)

    def apply(self, rig, output, *, defer_unhealthy=False):
        if not self.pending:
            return
        state = rig.state
        if defer_unhealthy and (rig.error is not None or state.fault or not state.healthy):
            print(f'startup profile {self.label} pending: clear the stopped fault first; motion requires a verified profile load', file=output)
            return
        if self.custom is None and rig.get_info().firmware not in TUNABLE_FIRMWARES:
            raise ValidationError(f'default tuned profile requires {TUNABLE_FIRMWARE}; use --keep-board-params for other firmware')
        values = rig.load_parameters(self.path)
        self.pending = False
        print(f'startup profile {self.label} applied and verified ({len(values)} settings); executor remains stopped', file=output)

    def before_command(self, rig, line, output):
        words = shlex.split(line)
        if words and words[0] in _MOTION_COMMANDS:
            self.apply(rig, output)

    def after_command(self, rig, line, output):
        words = shlex.split(line)
        if not words:
            return
        if words[0]=='load':
            self.pending = False  # An explicit successful operator load wins.
        elif words[0]=='clear':
            self.apply(rig, output, defer_unhealthy=True)
