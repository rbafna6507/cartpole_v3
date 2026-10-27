"""Contract failures that would otherwise pass bad observations into a policy."""
import dataclasses
import math
import unittest
from cartpole_sdk.contracts import (
    Acceleration, ActionLimits, ActionRequest, Channel, Observation, Sample,
    SelectionError, Session, Topology, ValidationError, select_channels,
    validate_action, validate_observation,
)


def sample(**changes):
    args = dict(entity='pole', quantity='angle', source='vision', value=0.2,
                unit='rad', frame_id='vision-cal-1', method='marker', validity='valid',
                reason=None, kind='measured', acquired_ns=None, effective_ns=90,
                received_ns=95)
    return Sample(**(args | changes))


def observation(**changes):
    return Observation(**(dict(session_id='run-1', clock_id='host-monotonic',
        sequence=7, available_ns=100, availability_basis='host-result',
        samples=(sample(),)) | changes))


def request(**changes):
    return ActionRequest(**(dict(session_id='run-1', clock_id='host-monotonic',
        sequence=2, observation_sequence=7, issued_at_ns=110,
        expires_at_ns=130, acceleration_m_s2=1.5) | changes))


class ObservationTests(unittest.TestCase):
    def test_multi_joint_topology_and_distinct_sensor_frames(self):
        session = Session('run-1', 'host-monotonic', Topology('cart', ('pole', 'pole2', 'pole3')))
        obs = observation(samples=(sample(), sample(source='magnetic', frame_id='encoder-zero'),
                                   sample(entity='pole3')))
        validate_observation(session, obs)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            obs.sequence = 8

    def test_reject_unknown_entity_and_incompatible_quantity(self):
        session = Session('run-1', 'host-monotonic')
        for s in (sample(entity='missing'), sample(entity='cart'),
                  sample(quantity='position', unit='m')):
            with self.subTest(s=s), self.assertRaises(ValidationError):
                validate_observation(session, observation(samples=(s,)))

    def test_reject_mismatched_session_or_clock(self):
        for session in (Session('other', 'host-monotonic'), Session('run-1', 'board')):
            with self.subTest(session=session), self.assertRaises(ValidationError):
                validate_observation(session, observation())

    def test_topology_names_unique_and_immutable(self):
        for args in (('cart', ('cart',)), ('cart', ('pole', 'pole')), ('cart', ['pole']), ('', ())):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                Topology(*args)

    def test_reject_duplicate_channels_but_keep_other_sources(self):
        with self.assertRaises(ValidationError):
            observation(samples=(sample(), sample(method='other-method')))
        self.assertEqual(len(observation(samples=(sample(), sample(source='fused'))).samples), 2)

    def test_units_and_vocabularies_are_checked(self):
        for changes in ({'unit':'degrees'}, {'quantity':'unknown'}, {'source':'auto'},
                        {'validity':'maybe'}, {'kind':'guess'}, {'frame_id':''}, {'method':''}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                sample(**changes)

    def test_bool_nonfinite_and_missing_valid_values_rejected(self):
        for value in (True, math.nan, math.inf, -math.inf, '0.2', None, 10**500):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                sample(value=value)

    def test_invalid_and_unknown_require_explanation(self):
        for validity in ('invalid', 'unknown'):
            with self.subTest(validity=validity):
                self.assertIsNone(sample(value=None, validity=validity, reason='not observed').value)
                with self.assertRaises(ValidationError):
                    sample(validity=validity)

    def test_integer_nonnegative_times_and_sequences(self):
        for value in (True, -1, 1.5, '1'):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    observation(sequence=value)
                with self.assertRaises(ValidationError):
                    sample(effective_ns=value)
        with self.assertRaises(ValidationError):
            observation(samples=[sample()])

    def test_acquisition_receipt_availability_order(self):
        self.assertEqual(sample(acquired_ns=80, effective_ns=85).effective_ns, 85)
        with self.assertRaises(ValidationError):
            sample(acquired_ns=96)
        with self.assertRaises(ValidationError):
            observation(samples=(sample(received_ns=101),))
        with self.assertRaises(ValidationError):
            observation(samples=(sample(effective_ns=101),))
        observation(samples=(sample(effective_ns=101, kind='predicted'),))


class SelectionTests(unittest.TestCase):
    channel = Channel('pole', 'angle', 'vision', 'vision-cal-1')

    def select(self, obs, **kwargs):
        return select_channels(obs, (self.channel,), **(dict(now_ns=100, max_age_ns=10) | kwargs))

    def test_source_frame_explicit_and_requested_order_preserved(self):
        magnetic = sample(source='magnetic', value=0.7, frame_id='encoder-zero')
        obs = observation(samples=(magnetic, sample()))
        selected = self.select(obs)
        self.assertEqual([s.value for s in selected], [0.2])

    def test_no_fallback_to_other_frame_or_source(self):
        for s in (sample(source='magnetic'), sample(frame_id='new-calibration')):
            with self.subTest(s=s), self.assertRaises(SelectionError):
                self.select(observation(samples=(s,)))

    def test_valid_number_is_not_enough_for_selection(self):
        for s in (sample(validity='invalid', reason='lost'), sample(validity='unknown', reason='no health'),
                  sample(effective_ns=None), sample(effective_ns=101, kind='predicted')):
            with self.subTest(s=s), self.assertRaises(SelectionError):
                self.select(observation(samples=(s,)))

    def test_age_is_based_on_effective_time_not_arrival_time(self):
        with self.assertRaises(SelectionError):
            self.select(observation(), now_ns=101)
        self.assertEqual(self.select(observation(), now_ns=100)[0].effective_ns, 90)

    def test_future_observation_rejected(self):
        with self.assertRaises(SelectionError):
            self.select(observation(), now_ns=99)

    def test_selector_configuration_rejects_bool_negative_and_duplicate_requests(self):
        for kwargs in ({'max_age_ns':-1}, {'now_ns':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                self.select(observation(), **kwargs)
        with self.assertRaises(ValidationError):
            select_channels(observation(), (self.channel, self.channel), now_ns=100, max_age_ns=10)


class ActionTests(unittest.TestCase):
    limits = ActionLimits(max_abs_acceleration=2.0, max_ttl_ns=20, max_observation_age_ns=30)

    def validate(self, req=None, **kwargs):
        return validate_action(req or request(), observation(), self.limits,
                               **(dict(now_ns=120, previous_sequence=1) | kwargs))

    def test_valid_request_has_explicit_units_and_zero_is_an_intent(self):
        self.validate()
        self.validate(request(acceleration_m_s2=0))
        self.assertEqual(Acceleration(0).acceleration_m_s2, 0)

    def test_exact_deadline_expired_and_future_issue_rejected(self):
        for now in (130, 131, 109):
            with self.subTest(now=now), self.assertRaises(ValidationError):
                self.validate(now_ns=now)
        self.validate(now_ns=129)

    def test_magnitude_and_lifetime_limits(self):
        for changes in ({'acceleration_m_s2':2.01}, {'acceleration_m_s2':-2.01}, {'expires_at_ns':131}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.validate(request(**changes))
        self.validate(request(acceleration_m_s2=-2.0))

    def test_action_must_match_observation_session_clock_and_sequence(self):
        for changes in ({'session_id':'other'}, {'clock_id':'board'}, {'observation_sequence':6}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.validate(request(**changes))

    def test_sequence_reuse_and_old_or_unavailable_observation_rejected(self):
        with self.assertRaises(ValidationError):
            self.validate(previous_sequence=2)
        with self.assertRaises(ValidationError):
            self.validate(request(issued_at_ns=99, expires_at_ns=119), now_ns=110)
        limits = ActionLimits(2, 100, 30)
        with self.assertRaises(ValidationError):
            validate_action(request(expires_at_ns=200), observation(), limits, now_ns=131)

    def test_constructor_rejects_malformed_acceleration_request_and_limits(self):
        for value in (True, '1', math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Acceleration(value)
        for changes in ({'sequence':True}, {'expires_at_ns':110}, {'expires_at_ns':109}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                request(**changes)
        for args in ((0, 20, 30), (2, 0, 30), (2, 20, -1), (True, 20, 30)):
            with self.subTest(args=args), self.assertRaises(ValidationError):
                ActionLimits(*args)
