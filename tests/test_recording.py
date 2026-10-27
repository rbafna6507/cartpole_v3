import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cartpole_sdk.contracts import Session, Sample, Observation
from cartpole_sdk.recording import write_recording, read_recording, strict_json_loads, canonical_json


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'record.jsonl'
        self.session = Session('run', 'clock')
        self.sample = Sample('pole', 'angle', 'vision', 0.2, 'rad', 'camera:epoch-1', 'pose', 'valid', None, 'measured', None, 10, None)
        self.obs = Observation('run', 'clock', 2, 20, 'result', (self.sample,))

    def wire(self):
        write_recording(self.path, self.session, (self.obs,))
        return [json.loads(s) for s in self.path.read_text().splitlines()]

    def rewrite(self, rows):
        self.path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')

    def test_round_trip_and_deterministic_bytes(self):
        write_recording(self.path, self.session, iter((self.obs,)))
        self.assertEqual(read_recording(self.path), (self.session, (self.obs,)))
        other = self.path.with_name('other.jsonl')
        write_recording(other, self.session, (self.obs,))
        self.assertEqual(self.path.read_bytes(), other.read_bytes())
        self.assertEqual(len(self.path.read_text().splitlines()), 3)

    def test_observation_payload_preserves_all_fields_and_independent_containers(self):
        from dataclasses import asdict
        from cartpole_sdk.portal.observation_payload import observation_payload
        for sample in (self.sample,replace(self.sample,value=None,validity='invalid',reason='occluded'),
                       replace(self.sample,kind='predicted',effective_ns=30)):
            observation=replace(self.obs,samples=(sample,))
            payload=observation_payload(observation)
            self.assertEqual(payload,asdict(observation))
            self.assertEqual(canonical_json(payload),canonical_json(asdict(observation)))
            payload['samples'][0]['value']=99
            self.assertEqual(observation_payload(observation),asdict(observation))

    def test_empty_recording_still_needs_footer(self):
        write_recording(self.path, self.session, ())
        self.assertEqual(read_recording(self.path), (self.session, ()))
        self.path.write_text(self.path.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'line.*footer'):
            read_recording(self.path)

    def test_exclusive_creation_preserves_existing_file(self):
        self.path.write_text('keep')
        with self.assertRaises(FileExistsError):
            write_recording(self.path, self.session, (self.obs,))
        self.assertEqual(self.path.read_text(), 'keep')

    def test_writer_removes_its_partial_file_on_iteration_failure(self):
        def broken():
            yield self.obs
            raise RuntimeError('iterator failed')
        with self.assertRaisesRegex(RuntimeError, 'iterator failed'):
            write_recording(self.path, self.session, broken())
        self.assertFalse(self.path.exists())

    def test_writer_rejects_order_session_and_topology(self):
        cases = [(self.obs, self.obs), (self.obs, replace(self.obs, sequence=3, available_ns=19)),
                 (replace(self.obs, session_id='other'),),
                 (replace(self.obs, samples=(replace(self.sample, entity='elbow'),)),)]
        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, 'line'):
                    write_recording(self.path, self.session, rows)
                self.assertFalse(self.path.exists())

    def test_reader_rejects_missing_wrong_and_trailing_footer(self):
        rows = self.wire()
        for bad in (rows[:-1], rows[:-1] + [dict(rows[-1], count=2)], rows + [rows[-1]], [rows[1], rows[0], rows[-1]]):
            self.rewrite(bad)
            with self.assertRaisesRegex(ValueError, 'line'):
                read_recording(self.path)

    def test_reader_rejects_unknown_fields_version_and_bad_nested_types(self):
        rows = self.wire()
        cases = []
        for index, field in ((0, 'version'), (0, 'extra'), (1, 'extra'), (2, 'extra')):
            modified = json.loads(json.dumps(rows))
            modified[index][field] = 2
            cases.append(modified)
        for field, value in [('sequence', True), ('available_ns', -1), ('session_id', 'wrong')]:
            modified = json.loads(json.dumps(rows))
            modified[1]['observation'][field] = value
            cases.append(modified)
        for field, value in [('value', float('nan')), ('effective_ns', True), ('entity', 'extra'), ('extra', 1)]:
            modified = json.loads(json.dumps(rows))
            modified[1]['observation']['samples'][0][field] = value
            cases.append(modified)
        modified = json.loads(json.dumps(rows))
        modified[0]['session']['topology']['joints'] = 'pole'
        cases.append(modified)
        for bad in cases:
            self.rewrite(bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'line'):
                read_recording(self.path)

    def test_reader_rejects_duplicate_json_keys_blank_lines_and_malformed_json(self):
        rows = self.wire()
        original = self.path.read_text()
        for text in (original.replace('"sequence":2', '"sequence":2,"sequence":3'), '\n'+original, original+'{bad\n'):
            self.path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'line'):
                read_recording(self.path)

    def test_json_helpers_reject_nonfinite_and_duplicate_keys(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                strict_json_loads(text)
        with self.assertRaises(ValueError):
            canonical_json({'a': float('inf')})
        self.assertEqual(canonical_json({'b': 2, 'a': 1}), '{"a":1,"b":2}')

    def test_reader_rejects_duplicate_channels_nested_unknowns_and_bool_footer(self):
        rows = self.wire()
        for variant in range(5):
            changed = json.loads(json.dumps(rows))
            if variant == 0:
                changed[1]['observation']['samples'] *= 2
            elif variant == 1:
                changed[0]['session']['unexpected'] = 'ignored?'
            elif variant == 2:
                changed[0]['session']['topology']['unexpected'] = 'ignored?'
            elif variant == 3:
                changed[0]['version'] = True
            else:
                changed[2]['count'] = True
            self.rewrite(changed)
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'line'):
                read_recording(self.path)

    def test_invalid_utf8_has_jsonl_line_context(self):
        self.wire()
        first = self.path.read_bytes().splitlines(keepends=True)[0]
        self.path.write_bytes(first + b'\xff\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            read_recording(self.path)
