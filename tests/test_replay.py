"""Replay must preserve recorded inputs and never impersonate a closed-loop rig."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cartpole_sdk import Acceleration, ActionLimits, Channel, Session, ValidationError
from cartpole_sdk.replay import replay, write_trace
from cartpole_sdk.policies import AngularDampingPolicy
from test_contracts import observation, sample


class CountingPolicy:
    def __init__(self):
        self.resets = 0
        self.seen = []

    def reset(self, session):
        self.resets += 1
        self.seen = []

    def act(self, obs):
        self.seen.append(obs)
        return None if len(self.seen) == 1 else Acceleration(0)


class ReplayTests(unittest.TestCase):
    session = Session('run-1', 'host-monotonic')
    limits = ActionLimits(2, 20, 30)

    def observations(self):
        return (observation(), observation(sequence=9, available_ns=120))

    def test_reset_once_each_run_and_no_intent_differs_from_zero(self):
        policy = CountingPolicy()
        steps = replay(self.session, self.observations(), policy, self.limits)
        self.assertEqual(policy.resets, 1)
        self.assertEqual([s.status for s in steps], ['no_request', 'requested'])
        self.assertIsNone(steps[0].action)
        self.assertEqual(steps[1].action.acceleration_m_s2, 0)
        self.assertEqual(steps[1].action.observation_sequence, 9)
        self.assertEqual(steps[1].action.sequence, 0)
        self.assertEqual(steps[1].action.expires_at_ns, 140)
        self.assertEqual(replay(self.session, self.observations(), policy, self.limits), steps)
        self.assertEqual(policy.resets, 2)
        self.assertEqual(policy.seen, list(self.observations()))

    def test_uses_recorded_availability_without_sleep_or_wall_clock(self):
        with patch('time.sleep', side_effect=AssertionError('must not sleep')), \
             patch('time.monotonic_ns', side_effect=AssertionError('must use recording clock')):
            steps = replay(self.session, self.observations(), CountingPolicy(), self.limits)
        self.assertEqual([s.at_ns for s in steps], [100, 120])

    def test_invalid_stream_rejected_before_any_policy_call(self):
        for rows in ((observation(), observation()),
                     (observation(), observation(sequence=8, available_ns=99)),
                     (observation(session_id='other'),)):
            policy = CountingPolicy()
            with self.subTest(rows=rows), self.assertRaises(ValidationError):
                replay(self.session, rows, policy, self.limits)
            self.assertEqual(policy.resets, 0)

    def test_bad_policy_result_or_exception_is_not_converted_to_zero(self):
        class Bad(CountingPolicy):
            def act(self, obs):
                return 0
        with self.assertRaises(ValidationError):
            replay(self.session, self.observations(), Bad(), self.limits)
        class Broken(CountingPolicy):
            def act(self, obs):
                raise RuntimeError('policy bug')
        with self.assertRaisesRegex(RuntimeError, 'policy bug'):
            replay(self.session, self.observations(), Broken(), self.limits)

    def test_runner_validates_intent_and_requested_ttl(self):
        class Oversized(CountingPolicy):
            def act(self, obs):
                return Acceleration(3)
        with self.assertRaises(ValidationError):
            replay(self.session, self.observations(), Oversized(), self.limits)
        for ttl in (True, 0, -1, 21):
            with self.subTest(ttl=ttl), self.assertRaises(ValidationError):
                replay(self.session, self.observations(), CountingPolicy(), self.limits, ttl_ns=ttl)

    def test_action_sequence_is_monotonic_across_observation_gaps(self):
        class Zero(CountingPolicy):
            def act(self, obs):
                return Acceleration(0)
        steps = replay(self.session, self.observations(), Zero(), self.limits, ttl_ns=10)
        self.assertEqual([s.action.sequence for s in steps], [0, 1])
        self.assertEqual(steps[1].action.expires_at_ns, 130)

    def test_trace_bytes_deterministic_and_output_exclusive(self):
        steps = replay(self.session, self.observations(), CountingPolicy(), self.limits)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.jsonl', Path(tmp)/'b.jsonl'
            write_trace(a, self.session, steps)
            write_trace(b, self.session, steps)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            rows = [json.loads(line) for line in a.read_text().splitlines()]
            self.assertEqual(rows[0]['mode'], 'open_loop_replay')
            self.assertEqual(rows[-1], {'type':'footer', 'count':2})
            with self.assertRaises(FileExistsError):
                write_trace(a, self.session, steps)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_trace_failure_removes_partial_file(self):
        steps = replay(self.session, self.observations(), CountingPolicy(), self.limits)
        def broken():
            yield steps[0]
            raise RuntimeError('read failed')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'partial.jsonl'
            with self.assertRaisesRegex(RuntimeError, 'read failed'):
                write_trace(path, self.session, broken())
            self.assertFalse(path.exists())

    def test_trace_rejects_wrong_session_and_duplicate_observation(self):
        steps = replay(self.session, self.observations(), CountingPolicy(), self.limits)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.jsonl'
            for bad in ((steps[0], steps[0]), (replace(steps[1], action=replace(steps[1].action, session_id='wrong')),)):
                with self.subTest(bad=bad), self.assertRaises(ValidationError):
                    write_trace(path, self.session, bad)
                self.assertFalse(path.exists())


class ExamplePolicyTests(unittest.TestCase):
    channel = Channel('pole', 'angular_velocity', 'vision', 'cal-1')

    def obs(self, **changes):
        return observation(samples=(sample(quantity='angular_velocity', unit='rad/s',
            frame_id='cal-1', value=4, **changes),))

    def test_explicit_angular_channel_generates_bounded_illustrative_acceleration(self):
        p = AngularDampingPolicy(self.channel, gain=0.1, max_acceleration=0.25, max_age_ns=10)
        p.reset(Session('run-1', 'host-monotonic'))
        self.assertEqual(p.act(self.obs()).acceleration_m_s2, -0.25)
        p = AngularDampingPolicy(self.channel, gain=0.01, max_acceleration=0.25, max_age_ns=10)
        self.assertAlmostEqual(p.act(self.obs()).acceleration_m_s2, -0.04)

    def test_missing_unknown_or_stale_input_means_no_request(self):
        p = AngularDampingPolicy(self.channel, max_age_ns=10)
        for obs in (observation(), self.obs(validity='unknown', reason='no health'),
                    replace(self.obs(), available_ns=101)):
            with self.subTest(obs=obs):
                self.assertIsNone(p.act(obs))

    def test_wrong_entity_quantity_and_bad_policy_configuration_rejected(self):
        with self.assertRaises(ValidationError):
            AngularDampingPolicy(Channel('pole', 'angle', 'vision', 'cal-1'))
        for kwargs in ({'gain':float('nan')}, {'gain':-1}, {'max_acceleration':0}, {'max_age_ns':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                AngularDampingPolicy(self.channel, **kwargs)
        p = AngularDampingPolicy(Channel('other', 'angular_velocity', 'vision', 'cal-1'))
        with self.assertRaises(ValidationError):
            p.reset(Session('run-1', 'host-monotonic'))
