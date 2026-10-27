import json
import tempfile
import unittest
from pathlib import Path

from cartpole_sdk.adapters import import_console, import_vision

FIXTURES = Path(__file__).parent / 'fixtures'
HEADER = 't_ms,mode,theta,theta_dot,x,v,accel,energy,z1,z2,agc,loop_us\n'


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'input'

    def console(self, times):
        self.path.write_text(HEADER + ''.join(f'{t},0,0.25,-0.4,0.01,0.2,0,0,0,0,128,100\n' for t in times))
        return list(import_console(self.path, 'run', 'firmware:raw'))

    def vision_row(self, **updates):
        row = dict(t=2.0, seq=4, angle_valid=True, cart_valid=True, calibrated=True,
                   theta_rad=0.25, omega_rad_s=-0.4, cart_x_m=0.01, cart_v_m_s=0.2,
                   omega_effective_t=1.96, cart_velocity_effective_t=1.95,
                   status='measured', velocity_status='measured', cart_source='arm pivot')
        row.update(updates)
        self.path.write_text(json.dumps(row) + '\n')
        return import_vision(self.path, 'run', 'vision:calibration-1')

    def test_real_console_preserves_unknown_quality_and_emission_clock(self):
        session, rows = next(import_console(FIXTURES / 'console-real.csv', 'run', 'firmware:raw'))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0].available_ns, 555643000000)
        self.assertEqual(rows[0].availability_basis, 'board_emission_proxy')
        by_quantity = {s.quantity: s for s in rows[0].samples}
        self.assertEqual(by_quantity['angle'].value, 2.5771)
        self.assertEqual(by_quantity['angle'].source, 'magnetic')
        self.assertEqual(by_quantity['angular_velocity'].kind, 'estimated')
        self.assertEqual(by_quantity['position'].source, 'step')
        for sample in rows[0].samples:
            self.assertEqual(sample.validity, 'unknown')
            self.assertIsNotNone(sample.reason)
            self.assertIsNone(sample.acquired_ns)
            self.assertIsNone(sample.received_ns)
            self.assertIsNone(sample.effective_ns)
            self.assertEqual(sample.frame_id, 'firmware:raw')
        self.assertIn('board', session.clock_id)

    def test_console_unwraps_uint32_wrap_but_splits_smaller_reset(self):
        segments = self.console([4294967290, 3, 20, 2, 12])
        self.assertEqual([s.session_id for s, _ in segments], ['run', 'run:segment-1'])
        self.assertEqual([o.available_ns for o in segments[0][1]], [4294967290000000, 4294967299000000, 4294967316000000])
        self.assertEqual([o.available_ns for o in segments[1][1]], [2000000, 12000000])
        self.assertEqual(segments[1][1][0].sequence, 0)
        self.assertNotEqual(segments[0][0].clock_id, segments[1][0].clock_id)

    def test_console_equal_times_keep_distinct_monotonic_sequences(self):
        _, rows = self.console([10, 10])[0]
        self.assertEqual([r.sequence for r in rows], [0, 1])

    def test_console_rejects_missing_extra_duplicate_columns_and_bad_numbers(self):
        for text in (HEADER.replace('theta_dot,', ''), HEADER.replace('mode,', 't_ms,'), HEADER.replace('loop_us', 'loop_us,extra')):
            self.path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'line 1'):
                list(import_console(self.path, 'run', 'raw'))
        for time in ('-1', '4294967296', '1.2', 'nan'):
            with self.assertRaisesRegex(ValueError, 'line 2'):
                self.console([time])
        for row in ('1,0,nan,-0.4,0.01,0.2,0,0,0,0,128,100', '1,0,1', '1,0,.2,.3,.4,.5,0,0,0,0,128,100,0'):
            self.path.write_text(HEADER + row + '\n')
            with self.assertRaisesRegex(ValueError, 'line 2'):
                list(import_console(self.path, 'run', 'raw'))

    def test_real_vision_latency_and_derivative_warmup(self):
        session, rows = import_vision(FIXTURES / 'vision-real.jsonl', 'run', 'vision:epoch-1')
        self.assertEqual([r.sequence for r in rows], [1, 4, 5, 6])
        self.assertEqual(rows[0].available_ns, 21271159494956)
        self.assertEqual(rows[0].availability_basis, 'host_read_plus_result_latency')
        angle, omega, position, velocity = rows[0].samples
        self.assertEqual(angle.validity, 'valid')
        self.assertIsNone(angle.acquired_ns)
        self.assertEqual(angle.received_ns, 21271124301446)
        self.assertEqual(angle.effective_ns, angle.received_ns)
        self.assertEqual(omega.validity, 'invalid')
        self.assertEqual(omega.reason, 'warming up')
        self.assertEqual(position.validity, 'invalid')
        self.assertEqual(rows[3].samples[1].validity, 'valid')
        self.assertLess(rows[3].samples[1].effective_ns, rows[3].samples[0].effective_ns)
        self.assertIn('host', session.clock_id)

    def test_vision_proxy_does_not_claim_camera_acquisition(self):
        _, rows = self.vision_row()
        self.assertEqual(rows[0].available_ns, 2000000000)
        self.assertEqual(rows[0].availability_basis, 'host_read_proxy')
        self.assertEqual(rows[0].samples[1].effective_ns, 1960000000)
        self.assertTrue(all(s.acquired_ns is None for s in rows[0].samples))

    def test_vision_invalid_numeric_remnants_stay_invalid(self):
        _, rows = self.vision_row(angle_valid=False, cart_valid=False, status='camera stale', velocity_status='camera stale')
        for sample in rows[0].samples:
            self.assertEqual(sample.validity, 'invalid')
            self.assertIsNotNone(sample.value)
            self.assertIn('stale', sample.reason)
        _, rows = self.vision_row(calibrated=False)
        self.assertEqual(rows[0].samples[0].validity, 'invalid')
        self.assertEqual(rows[0].samples[1].validity, 'invalid')

    def test_vision_independent_cart_and_missing_effective_time(self):
        _, rows = self.vision_row(angle_valid=False, status='no arm measurement', cart_source='template', omega_effective_t=None, cart_velocity_effective_t=None)
        self.assertEqual(rows[0].samples[2].validity, 'valid')
        self.assertEqual(rows[0].samples[3].validity, 'unknown')
        self.assertIsNone(rows[0].samples[3].effective_ns)

    def test_vision_requires_frame_identity_and_rejects_bad_input(self):
        self.vision_row()
        for frame in ('', None):
            with self.assertRaises(ValueError):
                import_vision(self.path, 'run', frame)
        for updates in (dict(t=True), dict(seq=True), dict(t=-1), dict(angle_valid=1), dict(theta_rad=float('inf')),
                        dict(host_read_to_result_ms=-1), dict(omega_effective_t=3)):
            with self.assertRaisesRegex(ValueError, 'line 1'):
                self.vision_row(**updates)
        self.path.write_text('{"t":2,"t":3}\n')
        with self.assertRaisesRegex(ValueError, 'line 1'):
            import_vision(self.path, 'run', 'raw')

    def test_vision_rejects_sequence_and_time_regression(self):
        self.vision_row()
        row = json.loads(self.path.read_text())
        for updates in (dict(seq=4, t=3), dict(seq=5, t=1), dict(seq=5, t=2)):
            other = dict(row, **updates)
            self.path.write_text(json.dumps(row)+'\n'+json.dumps(other)+'\n')
            with self.assertRaisesRegex(ValueError, 'line 2'):
                import_vision(self.path, 'run', 'raw')

    def test_vision_unknown_fields_and_missing_status_are_rejected(self):
        for updates in (dict(unrecognized_calibration_epoch='oops'), dict(status=None)):
            with self.assertRaisesRegex(ValueError, 'line 1'):
                self.vision_row(**updates)
        self.vision_row()
        row = json.loads(self.path.read_text())
        del row['seq']
        self.path.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError, 'line 1'):
            import_vision(self.path, 'run', 'epoch')

    def test_vision_legacy_derivative_status_stays_unknown(self):
        _, rows = self.vision_row(velocity_status=None)
        self.assertEqual(rows[0].samples[1].validity, 'unknown')
        self.assertIsNotNone(rows[0].samples[1].value)
        self.assertIn('status', rows[0].samples[1].reason)

    def test_real_vision_missing_pose_retains_invalid_samples(self):
        _, rows = import_vision(FIXTURES / 'vision-invalid-real.jsonl', 'run', 'vision:epoch-1')
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1].samples[0].validity, 'invalid')
        self.assertEqual(rows[1].samples[0].reason, 'no arm measurement')
        self.assertIsNone(rows[1].samples[0].value)
        self.assertEqual(rows[1].availability_basis, 'host_read_proxy')
        self.assertIsNone(rows[1].samples[0].acquired_ns)

    def test_vision_spatial_extrapolation_is_labeled_without_temporal_prediction(self):
        _, rows = self.vision_row(cart_extrapolated=True)
        for sample in rows[0].samples[2:]:
            self.assertIn('spatial_calibration_extrapolation', sample.method)
            self.assertEqual(sample.kind, 'estimated')
            self.assertEqual(sample.validity, 'valid')
        with self.assertRaisesRegex(ValueError, 'line 1'):
            self.vision_row(cart_extrapolated=1)

    def test_real_console_backward_clock_jump_splits_segments(self):
        segments = list(import_console(FIXTURES / 'console-reset-real.csv', 'run', 'firmware:raw'))
        self.assertEqual(len(segments), 2)
        self.assertEqual([len(rows) for _, rows in segments], [4, 2])
        self.assertEqual(segments[0][1][-1].available_ns, 8767000000)
        self.assertEqual(segments[1][1][0].available_ns, 1099000000)
        self.assertEqual(segments[1][0].session_id, 'run:segment-1')

    def test_console_clock_fraction_cannot_round_into_valid_uint32(self):
        for time in ('4294967295.0000000001', '1e-9999'):
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, 'line 2'):
                self.console([time])

    def test_vision_invalid_utf8_has_jsonl_line_context(self):
        self.vision_row()
        self.path.write_bytes(self.path.read_bytes() + b'\xff\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            import_vision(self.path, 'run', 'epoch')

    def test_vision_detectable_calibration_epoch_change_is_rejected(self):
        self.vision_row()
        row = json.loads(self.path.read_text())
        reset = dict(row, seq=5, t=3, calibrated=False)
        self.path.write_text(json.dumps(row)+'\n'+json.dumps(reset)+'\n')
        with self.assertRaisesRegex(ValueError, 'line 2.*calibration epoch'):
            import_vision(self.path, 'run', 'epoch')
        startup = dict(row, calibrated=False)
        calibrated = dict(row, seq=5, t=3)
        self.path.write_text(json.dumps(startup)+'\n'+json.dumps(calibrated)+'\n')
        _, rows = import_vision(self.path, 'run', 'epoch')
        self.assertEqual(rows[0].samples[0].validity, 'invalid')
        self.assertEqual(rows[1].samples[0].validity, 'valid')
