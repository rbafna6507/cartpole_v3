"""Stopped live tuning and verified profiles through the real PTY client."""
from dataclasses import asdict, fields
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from cartpole_sdk.contracts import ValidationError
from cartpole_sdk.control import ControllerParameters
from cartpole_sdk.rig import Rig, RigError
from cartpole_sdk.rig import protocol
from cartpole_sdk.rig.console import execute_command, main
from test_rig import Firmware, FDTransport


class ParameterFirmware(Firmware):
    def __init__(self):
        self.values = asdict(ControllerParameters())
        self.writes = []
        self.reject = None
        self.wrong_name = False
        self.only_wrong_name = False
        self.override = None
        self.drop_write_reply = None
        self.span = 0
        super().__init__()

    def command(self, payload):
        parts = payload.split()
        if parts[1] == 'INFO':
            self.commands.append(parts)
            self.send(f'V2 INFO {self.sid} executor-v6-measuredrail 270 1 32 0 1 {self.span} 0.26 768 768 1 2300 0 0')
            return
        if parts[1] != 'PARAM':
            return super().command(payload)
        self.commands.append(parts)
        name = parts[3]
        if self.reject == name:
            self.send(f'V2 ERR {self.sid} parameter');return
        if len(parts) == 5:
            value = struct.unpack('f', struct.pack('f', float(parts[4])))[0]
            proposed = dict(self.values, **{name:value})
            if not 0 < proposed['spin_resume_rad_s'] < proposed['spin_trip_rad_s']:
                self.send(f'V2 ERR {self.sid} value');return
            self.values[name] = value
            self.writes.append((name, value))
            if self.drop_write_reply == name:
                return
        if self.wrong_name or self.only_wrong_name:
            other = 'pw' if name != 'pw' else 'ke'
            self.send(f'V2 PARAM {self.sid} {other} 9')
            if self.only_wrong_name:return
        actual = self.values[name] if self.override is None else self.override
        self.send(f'V2 PARAM {self.sid} {name} {actual:.9g}')


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.fw = ParameterFirmware()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.fw.close)

    def rig(self, port=None, **kwargs):
        return Rig(port or self.fw.port, transport_factory=FDTransport, command_timeout=.1, **kwargs)

    def profile(self, parameters, **extra):
        path = Path(self.tmp.name)/'profile.json'
        path.write_text(json.dumps(dict(schema_version=1, parameters=parameters, **extra)))
        return path

    def test_parameter_reply_parser_is_strict(self):
        self.assertEqual(protocol.parse_parameter('V2 PARAM 7 pc1 -2.5'), ('pc1', -2.5))
        for payload in ('V2 PARAM 0 ke 3', 'V2 PARAM 7 ke 3 extra', 'V2 PARAM 7 ke',
                        'V2 PARAM 7 unknown 3', 'V2 PARAM 7 ke nan', 'V2 PARAM 7 ke inf',
                        'V2 PARAM 7 ke 1e999', 'V2 PARAM 7 ke 0x1', 'V2 PARAM 7  ke 3'):
            with self.subTest(payload=payload), self.assertRaises(protocol.ProtocolError):
                protocol.parse_parameter(payload)

    def test_get_reads_current_board_value_and_matches_response_name(self):
        self.fw.wrong_name = True
        with self.rig() as rig:
            self.fw.values['ke'] = 3.25
            self.assertEqual(rig.get_parameter('ke'), 3.25)
            self.fw.values['ke'] = 4.75
            self.assertEqual(rig.get_parameter('ke'), 4.75)
            self.assertFalse(self.fw.writes)
            self.assertFalse(any(c[1] in ('ARM', 'MODE') for c in self.fw.commands))

    def test_wrong_parameter_reply_cannot_complete_request(self):
        self.fw.only_wrong_name = True
        with self.rig() as rig:
            with self.assertRaises(RigError):rig.get_parameter('ke')
            self.assertIsNotNone(rig.error)

    def test_all_parameters_are_live_reads_of_catalog(self):
        self.fw.values['leff'] = .13
        with self.rig() as rig:
            values = rig.get_parameters()
            self.assertEqual(set(values), {f.name for f in fields(ControllerParameters)})
            self.assertEqual(len(values), 41)
            self.assertAlmostEqual(values['leff'], .13)
            self.assertEqual({c[3] for c in self.fw.commands if c[1]=='PARAM'}, set(values))

    def test_set_accepts_float32_readback_and_returns_actual_value(self):
        with self.rig() as rig:
            actual = rig.set_parameter('leff', .123456789)
            self.assertAlmostEqual(actual, .123456791, places=9)
            self.assertEqual(rig.get_parameter('leff'), actual)
            self.assertEqual(self.fw.writes[0][0], 'leff')

    def test_higher_jerk_can_be_verified_and_saved_while_stopped(self):
        self.assertEqual(ControllerParameters().jmax, 150)
        path = Path(self.tmp.name)/'higher-jerk.json'
        with self.rig() as rig:
            for value in (300, 600):
                self.assertEqual(rig.set_parameter('jmax', value), value)
                self.assertEqual(rig.get_parameter('jmax'), value)
            rig.save_parameters(path)
            self.assertEqual(json.loads(path.read_text())['parameters']['jmax'], 600)
            rig.set_parameter('jmax', 150)
            self.assertEqual(rig.load_parameters(path)['jmax'], 600)
            self.assertFalse(any(c[1] in ('ARM', 'MODE', 'HOME') for c in self.fw.commands))

    def test_higher_jerk_rejects_invalid_values_and_active_writes(self):
        with self.rig() as rig:
            for value in (0, 10000.1, float('nan'), float('inf')):
                with self.subTest(value=value), self.assertRaises(ValidationError):
                    rig.set_parameter('jmax', value)
            self.assertFalse(self.fw.writes)
            rig.start('auto', .5)
            with self.assertRaisesRegex(RigError, '(?i)stop'):
                rig.set_parameter('jmax', 600)
            self.assertFalse(self.fw.writes)

    def test_older_firmware_rejection_of_higher_jerk_is_reported(self):
        self.fw.reject = 'jmax'
        with self.rig() as rig:
            with self.assertRaises(RigError):
                rig.set_parameter('jmax', 300)
            self.assertIsNotNone(rig.error)
            self.assertEqual(self.fw.values['jmax'], 150)
            self.assertFalse(self.fw.writes)

    def test_set_rejects_mismatched_actual_value_and_requests_stop(self):
        self.fw.override = 2.0
        with self.rig() as rig:
            with self.assertRaisesRegex(RigError, '(?i)(readback|confirm|actual)'):
                rig.set_parameter('ke', 3.0)
            self.assertIsNotNone(rig.error)
            self.assertFalse(rig._active)

    def test_validation_happens_before_parameter_writes(self):
        with self.rig() as rig:
            for name, value in (('unknown', 1), ('ke other', 1), ('ke', True),
                                ('ke', '3'), ('ke', float('nan')), ('ke', float('inf')),
                                ('ke', 1e100), ('leff', 0), ('vmax', 2.68), ('bw', 101)):
                with self.subTest(name=name, value=value), self.assertRaises(ValidationError):
                    rig.set_parameter(name, value)
            self.assertFalse(self.fw.writes)
            self.assertFalse(any(c[1]=='PARAM' for c in self.fw.commands))

    def test_reads_and_writes_require_stopped_executor(self):
        with self.rig() as rig:
            rig.start('auto', .5)
            for operation in (lambda:rig.get_parameter('ke'), rig.get_parameters,
                              lambda:rig.set_parameter('ke', 3)):
                with self.assertRaisesRegex(RigError, '(?i)stop'):operation()
            self.assertFalse(any(c[1]=='PARAM' for c in self.fw.commands))

    def test_faulted_get_works_but_write_and_calibration_are_rejected(self):
        with self.rig() as rig:
            self.fw.mode = 4;self.fw.fault = 4
            rig._snapshot(allow_error=True)
            self.assertEqual(rig.get_parameter('ke'), 4.0)
            with self.assertRaises(RigError):rig.set_parameter('ke', 3)
            self.assertFalse(self.fw.writes)
        self.fw.mode=0;self.fw.fault=0;self.fw.flags|=256
        with self.rig() as rig:
            with self.assertRaisesRegex(RigError, '(?i)calibrat'):rig.set_parameter('ke', 3)
            self.assertFalse(self.fw.writes)

    def test_unsupported_firmware_error_does_not_claim_success(self):
        self.fw.reject = 'ke'
        with self.rig() as rig:
            with self.assertRaises(RigError):rig.get_parameter('ke')
            self.assertIsNotNone(rig.error)

    def test_save_uses_verified_live_values_and_does_not_change_board(self):
        self.fw.values['ke'] = 3.125
        path = Path(self.tmp.name)/'saved.json'
        with self.rig() as rig:
            rig.save_parameters(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved['schema_version'], 1)
            self.assertEqual(saved['parameters']['ke'], 3.125)
            self.assertEqual(len(saved['parameters']), 41)
            self.assertFalse(self.fw.writes)

    def test_load_validates_every_value_before_first_write(self):
        with self.rig() as rig:
            for values in ({'ke':3,'unknown':2}, {'ke':3,'leff':0}, {'ke':3,'vmax':2.68},
                           {'ke':3,'bw':float('nan')}, {'ke':True}, {}):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    rig.load_parameters(self.profile(values))
            self.assertFalse(self.fw.writes)

    def test_load_rejects_schema_and_duplicate_keys_before_writing(self):
        path = Path(self.tmp.name)/'invalid.json'
        with self.rig() as rig:
            for text in ('{"schema_version":2,"parameters":{"ke":3}}',
                         '{"schema_version":true,"parameters":{"ke":3}}',
                         '{"schema_version":1,"parameters":{"ke":3,"ke":4}}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValidationError):rig.load_parameters(path)
            self.assertFalse(self.fw.writes)

    def test_partial_profile_load_reads_back_changes_and_never_starts_motion(self):
        with self.rig() as rig:
            loaded = rig.load_parameters(self.profile({'ke':3.25,'kpx':35}))
            self.assertEqual(loaded, {'ke':3.25,'kpx':35.0})
            self.assertEqual(rig.get_parameter('ke'), 3.25)
            self.assertEqual(self.fw.values['pw'], 4.5)
            self.assertFalse(any(c[1] in ('ARM','MODE','HOME','CENTER') for c in self.fw.commands))

    def test_partial_load_failure_reports_confirmed_and_uncertain_keys(self):
        self.fw.drop_write_reply = 'kpx'
        with self.rig() as rig:
            with self.assertRaises(RigError) as caught:
                rig.load_parameters(self.profile({'ke':3.25,'kpx':35}))
            message = str(caught.exception)
            self.assertIn('ke', message)
            self.assertIn('kpx', message)
            self.assertRegex(message, '(?i)(partial|changed)')
            self.assertRegex(message, '(?i)(uncertain|may have|unconfirmed)')
            self.assertEqual(self.fw.values['ke'], 3.25)
            self.assertEqual(self.fw.values['kpx'], 35)
            self.assertFalse(rig._active)

    def test_console_get_set_params_save_and_load_show_readback(self):
        path = Path(self.tmp.name)/'console profile.json'
        with self.rig() as rig:
            output=io.StringIO()
            for line in ('set ke 3.25', 'get ke', 'params', f'save "{path}"',
                         'set ke 4', f'load "{path}"'):
                self.assertTrue(execute_command(rig,line,output=output))
            self.assertIn('3.25', output.getvalue())
            self.assertIn('ke', output.getvalue())
            self.assertEqual(rig.get_parameter('ke'), 3.25)
            self.assertTrue(path.is_file())

    def test_cli_set_and_get_do_not_start_motion(self):
        for command, extra in (('set',['--value','3.25']), ('get',[])):
            output=io.StringIO()
            result=main(['--port',self.fw.port,'--log-dir',self.tmp.name,
                         '--command',command,'--param','ke',*extra],rig_factory=self.rig,output=output)
            self.assertEqual(result,0,output.getvalue())
            self.assertIn('3.25',output.getvalue())
        self.assertFalse(any(c[1] in ('ARM','MODE') for c in self.fw.commands))

    def test_live_bounds_reject_before_writes(self):
        bad = {'leff':(.019,1.01), 'pw':(.09,30.1), 'bal_pw':(.9,20.1),
               'pz':(.04,5.1), 'pc1':(-50.1,0), 'pc2':(-50.1,-.09),
               'vmax_s':(0,2.68), 'vmax_b':(0,2.68), 'approach_v':(0,2.68),
               'catch_v':(0,2.68), 'amax_s':(0,200.1), 'amax_b':(0,200.1),
               'jmax':(0,10000.1), 'amax_m':(0,200.1), 'jmax_m':(0,10000.1),
               'vman':(-.01,.151), 'ke':(-.01,100.1), 'kpx':(-.01,1000.1),
               'kdx':(-.01,100.1), 'phase_soft':(0,50.1), 'catch_a':(.04,1.21),
               'catch_r':(.09,30.1), 'giveup':(.09,2.51), 'bw':(.09,100.1),
               'approach_angle':(.64,1.51), 'catch_da':(0,200.1),
               'spin_trip_rad_s':(0,150.1), 'spin_resume_rad_s':(0,150.1),
               'rail':(.02,.376)}
        with self.rig() as rig:
            for name, values in bad.items():
                for value in values:
                    with self.subTest(name=name,value=value),self.assertRaises(ValidationError):
                        rig.set_parameter(name,value)
            self.assertFalse(self.fw.writes)

    def test_load_validates_spin_pair_and_orders_lower_then_higher_pairs(self):
        with self.rig() as rig:
            with self.assertRaises(ValidationError):
                rig.load_parameters(self.profile({'ke':3,'spin_trip_rad_s':5}))
            self.assertFalse(self.fw.writes)
            rig.load_parameters(self.profile({'spin_trip_rad_s':5,'spin_resume_rad_s':2}))
            self.assertEqual([name for name,_ in self.fw.writes],['spin_resume_rad_s','spin_trip_rad_s'])
            self.fw.writes.clear()
            rig.load_parameters(self.profile({'spin_resume_rad_s':10,'spin_trip_rad_s':20}))
            self.assertEqual([name for name,_ in self.fw.writes],['spin_trip_rad_s','spin_resume_rad_s'])

    def test_live_measured_rail_cap_checked_before_any_profile_write(self):
        self.fw.span = 10668
        with self.rig() as rig:
            with self.assertRaises(ValidationError):rig.set_parameter('rail', .201)
            with self.assertRaises(ValidationError):
                rig.load_parameters(self.profile({'ke':3,'rail':.201}))
            self.assertFalse(self.fw.writes)
            self.assertAlmostEqual(rig.set_parameter('rail', .19), .19, places=7)

    def test_unhomed_rail_uses_hard_cap_not_old_measured_span(self):
        self.fw.span=10668
        self.fw.flags=1
        with self.rig() as rig:
            self.assertAlmostEqual(rig.set_parameter('rail', .2), .2, places=7)

    def test_explicit_unchanged_rail_profile_stores_the_preference(self):
        with self.rig() as rig:
            rig.load_parameters(self.profile({'rail':.26}))
            self.assertEqual(len(self.fw.writes), 1)
            self.assertEqual(self.fw.writes[0][0], 'rail')

    def test_partial_failure_reports_confirmed_rail_preference(self):
        self.fw.drop_write_reply='kpx'
        with self.rig() as rig:
            with self.assertRaises(RigError) as caught:
                rig.load_parameters(self.profile({'rail':.26,'kpx':35}))
            self.assertIn('rail', caught.exception.changed_parameters)
            self.assertEqual(caught.exception.unconfirmed_parameter, 'kpx')

    def test_invalid_measured_span_rejects_before_profile_write(self):
        self.fw.span=10666
        with self.rig() as rig:
            with self.assertRaises(ValidationError):
                rig.load_parameters(self.profile({'ke':3,'rail':.19}))
            self.assertFalse(self.fw.writes)

    def test_measured_rail_accepts_wider_profile_without_arming(self):
        self.fw.span=16509
        with self.rig() as rig:
            self.assertAlmostEqual(rig.set_parameter('rail', .3090375), .3090375, places=7)
            self.assertAlmostEqual(rig.load_parameters(self.profile({'rail':.30}))['rail'], .30, places=7)
            self.fw.writes.clear()
            with self.assertRaises(ValidationError):
                rig.load_parameters(self.profile({'ke':3,'rail':struct.unpack('f', struct.pack('I', struct.unpack('I', struct.pack('f', .3090375))[0]+1))[0]}))
            self.assertFalse(self.fw.writes)
            self.assertFalse(any(c[1] in ('ARM','MODE','HOME') for c in self.fw.commands))

    def test_unreferenced_rail_cannot_expand_using_stale_span(self):
        self.fw.span=20000
        self.fw.flags=1
        with self.rig() as rig:
            with self.assertRaises(ValidationError):rig.set_parameter('rail', .30)
            self.assertFalse(self.fw.writes)

    def test_set_readback_must_match_the_requested_float32(self):
        self.fw.override=3.000000238418579
        with self.rig() as rig:
            with self.assertRaisesRegex(RigError, 'readback'):
                rig.set_parameter('ke',3.0)

    def test_cli_home_allows_75_seconds_but_other_motion_stays_at_60(self):
        self.fw.flags|=8
        self.fw.home_complete_ms=5
        output=io.StringIO()
        self.assertEqual(main(['--port',self.fw.port,'--log-dir',self.tmp.name,
                               '--command','home','--duration','75'],
                              rig_factory=self.rig,output=output),0,output.getvalue())
        for command,duration in (('home','76'),('auto','61'),('manual','61')):
            with self.subTest(command=command),self.assertRaises(SystemExit):
                main(['--port',self.fw.port,'--command',command,'--duration',duration],
                     rig_factory=self.rig,output=io.StringIO())

    def test_subnormal_and_nonzero_underflow_values_are_rejected_before_writes(self):
        with self.rig() as rig:
            for value in (1e-39, 1e-50, -1e-50):
                with self.subTest(value=value), self.assertRaises(ValidationError):
                    rig.set_parameter('ke', value)
            self.assertFalse(self.fw.writes)
            self.assertFalse(any(c[1]=='PARAM' for c in self.fw.commands))

    def test_late_subnormal_profile_value_prevents_all_earlier_writes(self):
        with self.rig() as rig:
            for values in ({'ke':3,'jmax':1e-39}, {'kpx':35,'ke':1e-50}):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    rig.load_parameters(self.profile(values))
            self.assertFalse(self.fw.writes)

    def test_intentional_zero_parameter_is_still_supported(self):
        with self.rig() as rig:
            self.assertEqual(rig.set_parameter('ke', 0), 0)
