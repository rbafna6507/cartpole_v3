"""Console wiring with scripted time, plus real PTY transport integration."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import tempfile
import time
import unittest
from cartpole_sdk.contracts import Acceleration, Channel
from cartpole_sdk.rig import Rig, RigError
from cartpole_sdk.control.parameters import ControllerParameters, gains
from cartpole_sdk.control.reference import balance
from cartpole_sdk.rig.protocol import State
from cartpole_sdk.rig.console import main, execute_command
from cartpole_sdk.rig.policy import run_policy
from test_rig import Firmware, FDTransport

CHANNELS=(Channel('pole','angle','magnetic','encoder-zero'), Channel('cart','position','step','cart-step-home'))

class FailingPolicy:
    def reset(self, session): self.session=session
    def act(self, observation): raise ValueError('policy failed')

class BoundedPolicy:
    def reset(self, session): pass
    def act(self, observation): return Acceleration(.1)

class PolicyAndConsoleTests(unittest.TestCase):
    def setUp(self):self.fw=Firmware()
    def tearDown(self):self.fw.close()
    def rig(self, port=None, **kwargs):
        return Rig(port or self.fw.port,transport_factory=FDTransport,command_timeout=.2,**kwargs)
    def test_noninteractive_auto_records_and_stops_finally(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            code=main(['--port',self.fw.port,'--keep-board-params','--command','auto','--duration','.04','--switches','off','--center-confirmed','--log-dir',directory],rig_factory=self.rig)
            self.assertEqual(code,0)
            records=[json.loads(s) for p in __import__('pathlib').Path(directory).glob('*.jsonl') for s in p.read_text().splitlines()]
            self.assertTrue(any(r['type']=='command' and 'MODE' in r['payload'] for r in records))
            self.assertEqual(self.fw.mode,0)
            self.assertEqual([c for c in self.fw.commands if c[1]!='STATUS'][-1][1],'STOP')
    def test_interactive_operator_commands_use_board_reference(self):
        with self.rig() as rig:
            output=io.StringIO()
            execute_command(rig,'switches off',output=output)
            execute_command(rig,'center',output=output)
            execute_command(rig,'auto .2',output=output)
            self.assertIn('board reference',output.getvalue())
            execute_command(rig,'stop',output=output)
            self.assertFalse(execute_command(rig,'quit',output=output))
    def test_noninteractive_home_waits_for_physical_routine_completion(self):
        self.fw.home_complete_ms=50
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            code=main(['--port',self.fw.port,'--command','home','--switches','on','--duration','.3','--log-dir',directory],rig_factory=self.rig)
        self.assertEqual(code,0)
        self.assertGreaterEqual(self.fw.ticks()-self.fw.home_started,50)
        self.assertEqual(self.fw.mode,0)

    def test_status_decodes_installed_cart_and_unused_open_inputs(self):
        self.fw.flags=201
        with self.rig() as rig:
            output=io.StringIO()
            execute_command(rig,'status',output=output)
            data=json.loads(output.getvalue())
            self.assertEqual(data['switch_inputs'],{
                '1_left':'closed', '2_right':'closed', '3':'open', '4':'open'})
            self.assertTrue(data['switch_checking_enabled'])
            self.assertTrue(data['encoder_healthy'])
            self.assertFalse(data['homed'])
            self.assertFalse(data['drivers_enabled'])

    def test_policy_exception_stops_in_finally(self):
        with self.rig() as rig:
            with self.assertRaises(ValueError):
                run_policy(rig,lambda channels:FailingPolicy(),channels=CHANNELS,duration_s=.2)
            self.assertFalse(self.fw.flags&2)
            self.assertTrue(any(c[1]=='ARM' for c in self.fw.commands))
    def test_bounded_external_policy_records_requests_and_does_not_select_reference(self):
        with self.rig() as rig:
            run_policy(rig,lambda channels:BoundedPolicy(),channels=CHANNELS,duration_s=.045)
            self.assertTrue(any(c[1]=='A' for c in self.fw.commands))
            self.assertFalse(any(c[1]=='MODE' for c in self.fw.commands))
            self.assertFalse(self.fw.flags&2)
            self.assertTrue(any(r['type']=='action_request' for r in rig.records))
    def test_policy_source_selection_failure_stops_before_arm(self):
        with self.rig() as rig:
            with self.assertRaises(ValueError):
                run_policy(rig,lambda channels:BoundedPolicy(),channels=(Channel('pole','angle','vision','missing'),),duration_s=.2)
            self.assertFalse(any(c[1]=='ARM' for c in self.fw.commands))
    def test_cli_rejects_policy_without_explicit_sources_before_connect(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--port',self.fw.port,'--command','policy-zero'])
        self.assertFalse(self.fw.commands)
    def test_policy_output_above_commissioning_bound_stops(self):
        class Excessive(BoundedPolicy):
            def act(self, observation): return Acceleration(1.)
        with self.rig() as rig:
            with self.assertRaises(ValueError):
                run_policy(rig,lambda channels:Excessive(),channels=CHANNELS,duration_s=.2)
            self.assertFalse(any(c[1]=='A' for c in self.fw.commands))
            self.assertFalse(self.fw.flags&2)


class ScriptedClock:
    """Advance board/host time only when the test delivers another snapshot."""
    now_ns = 0
    def monotonic(self): return self.now_ns/1e9
    def monotonic_ns(self): return self.now_ns
    perf_counter_ns = staticmethod(time.perf_counter_ns)


class ScriptedConsoleRig(Rig):
    """Replace device scheduling, retaining real observations and Rig.step guards.

    Other PTY tests cover transport and real lease expiry. This success-path
    wiring test must not require the test host to meet a 5–15 ms scheduling
    window under unrelated test workload and host scheduling jitter.
    """
    def __init__(self, port, *, clock, intervals=(10,), **kwargs):
        super().__init__(port, **kwargs)
        self.clock = clock
        self.intervals = iter(intervals)
        self.parameters = ControllerParameters()
        self.commands = []
        self._info = SimpleNamespace(firmware='executor-v6-measuredrail')
        self._state = State(sid=self._sid, tick=0, us=0, mode=0, flags=205,
            theta=.02, angle_hat=.02, omega=.4, x=.01, velocity=0., acceleration=0.,
            demand=0., last_seq=0, fault=0, misses=0, dropped=0, brake_direction=0,
            received_ns=0)
        self._history[0] = self._state
        self._lease_until_ns = None

    def connect(self):
        self._connected = True
        return self

    def get_info(self):
        self._require()
        return self._info

    def get_parameters(self):
        self._stopped_healthy()
        return asdict(self.parameters)

    def get_parameter(self, name): return self.get_parameters()[name]

    def _command_state(self, command, *args, predicate, allow_error=False):
        self._require(allow_error=allow_error)
        self.commands.append(['V2', command, str(self._sid), *map(str, args)])
        if command in ('ARM', 'MODE'):
            self._state = replace(self._state, mode=12 if command=='ARM' else 3,
                                  flags=self._state.flags|2)
            self._lease_until_ns = self.clock.now_ns+20_000_000
        elif command=='STOP':
            self._state = replace(self._state, mode=0, flags=self._state.flags&~2)
            self._lease_until_ns = None
        else:
            raise AssertionError(f'unexpected scripted command {command}')
        self._history[self._state.unwrapped_tick] = self._state
        if not predicate(self._state): raise RigError('scripted state confirmation failed')

    def _send(self, payload):
        self._require()
        parts = payload.split()
        if parts[1]!='A': raise AssertionError(f'unexpected scripted payload {payload}')
        self.commands.append(parts)
        # Match the board's lease anchored to the observed tick, not receipt.
        self._lease_until_ns = (int(parts[4])+int(parts[5]))*1_000_000

    def wait_state(self, after_tick=None, timeout=1):
        self.clock.now_ns += next(self.intervals, 10)*1_000_000
        if self._lease_until_ns is not None and self.clock.now_ns>=self._lease_until_ns:
            raise RigError('scripted action lease expired')
        tick = self.clock.now_ns//1_000_000
        self._state = replace(self._state, tick=tick, us=tick*1000,
            unwrapped_tick=tick, unwrapped_us=tick*1000, received_ns=self.clock.now_ns)
        self._history[tick] = self._state
        return self._state


class PythonConsoleTests(unittest.TestCase):
    def setUp(self):
        from test_rig_parameters import ParameterFirmware
        self.fw=ParameterFirmware()
    def tearDown(self): self.fw.close()
    def rig(self, port=None, **kwargs):
        return Rig(port or self.fw.port, transport_factory=FDTransport, command_timeout=.2, **kwargs)

    def scripted_balance(self, intervals=(10,)):
        clock = ScriptedClock()
        connected = []
        def factory(port, **kwargs):
            rig = ScriptedConsoleRig(port, clock=clock, intervals=intervals, **kwargs)
            connected.append(rig)
            return rig
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as output, \
                patch('cartpole_sdk.rig.policy.time', clock), \
                patch('cartpole_sdk.rig.client.time', clock), \
                patch('cartpole_sdk.rig.console.time', clock):
            code=main(['--port','scripted-console','--rate','100','--command','python-bal',
                '--source','magnetic-step','--duration','.08','--keep-board-params',
                '--no-vision-telemetry','--log-dir',directory],rig_factory=factory)
        return code, output.getvalue(), connected[0]

    def test_python_balance_uses_external_action_and_stops_without_native_mode(self):
        code, output, rig = self.scripted_balance()
        self.assertEqual(code,0,output)
        actions=[record for record in rig.records if record['type']=='action_request']
        self.assertGreaterEqual(len(actions),3)
        expected=balance(gains(rig.parameters,True),.02,.4,.01,0.)
        for record in actions:
            self.assertAlmostEqual(record['request']['acceleration_m_s2'],expected)
        starts=[record for record in rig.records if record['type']=='policy_start']
        self.assertEqual(starts[0]['policy'],'ExecutorFeedbackPolicy')
        self.assertEqual(len(starts[0]['channels']),6)
        self.assertTrue(any(c[1]=='ARM' for c in rig.commands))
        self.assertTrue(any(c[1]=='A' for c in rig.commands))
        self.assertFalse(any(c[1]=='MODE' for c in rig.commands))
        self.assertEqual(rig.commands[-1][1],'STOP')
        self.assertFalse(rig.state.enabled)

    def test_policy_failure_identifies_rejected_and_previous_sent_snapshots(self):
        # 16 ms violates the policy's 100 Hz cadence but is inside the 20 ms
        # action lease, isolating rejection of this newly delivered frame.
        code, output, rig = self.scripted_balance(intervals=(10,16))
        self.assertEqual(code,1)
        self.assertIn('cadence',output)
        failures=[record for record in rig.records if record['type']=='policy_failure']
        self.assertEqual(len(failures),1)
        failure=failures[0]
        self.assertEqual(failure['stage'],'act')
        self.assertEqual(failure['failed_observation_sequence'],26)
        self.assertEqual(failure['failed_observation_board_ns'],26_000_000)
        self.assertEqual(failure['previous_sent_observation_sequence'],10)
        self.assertEqual(failure['previous_sent_observation_board_ns'],10_000_000)
        self.assertEqual(failure['fetch_started_host_ns'],26_000_000)
        self.assertEqual(failure['fetch_finished_host_ns'],26_000_000)
        self.assertEqual(failure['act_started_host_ns'],26_000_000)
        self.assertEqual(failure['act_finished_host_ns'],26_000_000)
        self.assertEqual(rig.commands[-1][1],'STOP')
        self.assertFalse(rig.state.enabled)

    def test_fetch_failure_does_not_mislabel_the_previous_observation(self):
        original=ScriptedConsoleRig.control_observation
        def fail_after_first_action(rig):
            if rig.clock.now_ns: raise RuntimeError('scripted fetch failed')
            return original(rig)
        with patch.object(ScriptedConsoleRig,'control_observation',fail_after_first_action):
            code, output, rig=self.scripted_balance()
        self.assertEqual(code,1)
        self.assertIn('scripted fetch failed',output)
        failure=next(record for record in rig.records if record['type']=='policy_failure')
        self.assertEqual(failure['stage'],'fetch')
        self.assertIsNone(failure['failed_observation_sequence'])
        self.assertIsNone(failure['failed_observation_board_ns'])
        self.assertEqual(failure['previous_sent_observation_sequence'],0)
        self.assertEqual(failure['previous_sent_observation_board_ns'],0)
        self.assertEqual(failure['fetch_started_host_ns'],10_000_000)
        self.assertIsNone(failure['fetch_finished_host_ns'])
        self.assertIsNone(failure['act_started_host_ns'])
        self.assertIsNone(failure['act_finished_host_ns'])
        self.assertEqual(rig.commands[-1][1],'STOP')

    def test_wait_failure_retains_the_successfully_sent_observation(self):
        code, output, rig=self.scripted_balance(intervals=(25,))
        self.assertEqual(code,1)
        self.assertIn('scripted action lease expired',output)
        failure=next(record for record in rig.records if record['type']=='policy_failure')
        self.assertEqual(failure['stage'],'wait')
        self.assertEqual(failure['failed_observation_sequence'],0)
        self.assertEqual(failure['previous_sent_observation_sequence'],0)
        self.assertEqual(failure['previous_sent_observation_board_ns'],0)
        self.assertEqual(rig.commands[-1][1],'STOP')

    def test_python_balance_from_hanging_faults_and_stops_without_action(self):
        self.fw.state=lambda **kwargs: __import__('test_rig').Firmware.state(self.fw, theta='3.14', **kwargs).replace(' 0.19 ', ' 3.14 ')
        with self.rig() as rig:
            with self.assertRaises(ValueError):
                execute_command(rig,'python-bal .1',output=io.StringIO())
            self.assertFalse(self.fw.flags&2)
            self.assertFalse(any(c[1]=='A' for c in self.fw.commands))
            self.assertTrue(any(c[1]=='STOP' for c in self.fw.commands))
            self.assertFalse(any(c[1]=='ARM' for c in self.fw.commands))

    def test_unsupported_host_rate_does_not_open_serial(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--port',self.fw.port,'--command','python-auto','--source','magnetic-step','--rate','500'],rig_factory=self.rig)
        self.assertFalse(self.fw.commands)

class PolicyPreflightTests(unittest.TestCase):
    setUp=PolicyAndConsoleTests.setUp
    tearDown=PolicyAndConsoleTests.tearDown
    rig=PolicyAndConsoleTests.rig
    def test_explicit_preflight_rejects_before_arm(self):
        class RejectsStart(BoundedPolicy):
            def validate_start(self, observation):
                raise ValueError('unsuitable starting pose')
        with self.rig() as rig:
            with self.assertRaisesRegex(ValueError,'starting pose'):
                run_policy(rig,lambda channels:RejectsStart(),channels=CHANNELS,duration_s=.2)
            self.assertFalse(any(c[1]=='ARM' for c in self.fw.commands))
            self.assertFalse(self.fw.flags&2)
