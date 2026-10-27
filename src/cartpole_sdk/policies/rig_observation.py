"""Shared, dependency-free encoder for rig SDK and governed Gym policy inputs.

Only explicit channels are read. This module imports no simulator or device
client; it can run on the policy host. The encoder preserves validity and age,
using finite zeros for missing values. The trusted runner/policy must still
enforce freshness and reject unusable channels before sending hardware actions.
"""
import math

from cartpole_sdk.contracts import Channel, Observation, ValidationError, select_channels

QUANTITIES = ('position', 'velocity', 'angle', 'angular_velocity', 'acceleration', 'brake_direction')
STATE_FIELDS = ('x_m', 'commanded_v_m_s', 'sin_angle', 'cos_angle',
                'filtered_omega_rad_s', 'commanded_a_m_s2', 'brake_direction')
SCHEMA = 'cartpole-rig-observation-v1'
ANGLE_SOURCES = ('raw', 'filtered', 'vision', 'fused')


def observation_contract(*, angle_source='raw', angle_frame_id='encoder-zero',
                         vision_max_age_ns=150_000_000):
    """Explicit inference contract, independent of simulator dependencies.

    Camera effective-time tolerance never changes the 20 ms cart/command lease.
    Fused angle uses the common SDK complementary estimator; its rate remains
    the aligned magnetic estimate, explicitly declared by its channel source.
    """
    if angle_source not in ANGLE_SOURCES:
        raise ValidationError('angle_source must be raw, filtered, vision or fused')
    if not isinstance(angle_frame_id, str) or not angle_frame_id.strip():
        raise ValidationError('explicit angle_frame_id required')
    if angle_source in ('raw', 'filtered') and angle_frame_id != 'encoder-zero':
        raise ValidationError('encoder input requires encoder-zero frame')
    if type(vision_max_age_ns) is not int or not 1 <= vision_max_age_ns <= 500_000_000:
        raise ValidationError('vision_max_age_ns must be 1..500000000')
    return dict(schema='cartpole-policy-observation-contract-v1', angle_source=angle_source,
                angle_frame_id=angle_frame_id, vision_max_age_ns=vision_max_age_ns)


def validate_observation_contract(value):
    if not isinstance(value, dict) or set(value) != {
            'schema', 'angle_source', 'angle_frame_id', 'vision_max_age_ns'}:
        raise ValidationError('complete observation contract required')
    expected = observation_contract(**{k: v for k, v in value.items() if k != 'schema'})
    if value != expected:
        raise ValidationError('unsupported observation contract')
    return expected


def contract_from_config(config):
    """Extract the runtime contract from a complete experiment sensor config."""
    return observation_contract(angle_source=config['angle_source'],
        angle_frame_id=config.get('angle_frame_id', 'encoder-zero'),
        vision_max_age_ns=config.get('vision_max_age_ns', 150_000_000))


def rig_channels(*, angle_source='raw', angle_frame_id='encoder-zero'):
    """Explicit channels from Rig.control_observation(), retaining raw by default."""
    observation_contract(angle_source=angle_source, angle_frame_id=angle_frame_id)
    source = dict(raw='magnetic', filtered='magnetic_filtered', vision='vision', fused='fused')[angle_source]
    return tuple(Channel(entity, quantity, source, frame) for entity, quantity, source, frame in (
        ('cart', 'position', 'step', 'cart-step-home'),
        ('cart', 'velocity', 'step', 'cart-step-home'),
        ('pole', 'angle', source, angle_frame_id),
        ('pole', 'angular_velocity', 'vision' if angle_source == 'vision' else 'magnetic', angle_frame_id),
        ('cart', 'acceleration', 'step', 'cart-step-home'),
        ('cart', 'brake_direction', 'step', 'cart-step-home'),
    ))


def simulation_channels(*, angle_source='raw'):
    if angle_source not in ANGLE_SOURCES:
        raise ValidationError('unsupported simulation angle source')
    angular_frame = 'sim:upright+x' + (':'+angle_source if angle_source in ('vision', 'fused') else '')
    return tuple(Channel('pole' if q in ('angle', 'angular_velocity') else 'cart',
                         q, 'simulation', angular_frame if q in ('angle', 'angular_velocity') else 'sim:upright+x') for q in QUANTITIES)


def channel_age_limits(channels, *, contract):
    contract = validate_observation_contract(contract)
    hardware = rig_channels(angle_source=contract['angle_source'], angle_frame_id=contract['angle_frame_id'])
    simulated = simulation_channels(angle_source=contract['angle_source'])
    if channels not in (hardware, simulated):
        raise ValidationError('selected source/frame channels differ from observation contract')
    return tuple(contract['vision_max_age_ns'] if contract['angle_source'] == 'vision' and i in (2, 3)
                 else 20_000_000 for i in range(6))


def select_rig_observation(observation, channels, *, contract):
    """Fail closed using source-specific effective age and exact channel identity."""
    limits = channel_age_limits(channels, contract=contract)
    samples = tuple(select_channels(observation, (c,), now_ns=observation.available_ns,
                                   max_age_ns=age)[0] for c, age in zip(channels, limits))
    synchronous = [s for i, s in enumerate(samples) if not (
        contract['angle_source'] == 'vision' and i in (2, 3))]
    if max(s.effective_ns for s in synchronous)-min(s.effective_ns for s in synchronous) > 10_000_000:
        raise ValidationError('sensor timestamp skew exceeds one control period')
    if samples[-1].value not in (-1, 0, 1):
        raise ValidationError('invalid brake direction')
    return samples


def encode_rig_observation(observation, channels):
    """Return list-valued state/valid/age_s in the fixed seven-feature schema.

    No inference about source/frame is made. Missing channels are unknown;
    invalid samples preserve their age when known. Age is in the observation's
    declared clock, not host receipt time. Raw angle and filtered angle are
    separate channel selections with the same numeric encoding.
    """
    if not isinstance(observation, Observation):
        raise ValidationError('expected SDK Observation')
    if (type(channels) is not tuple or len(channels) != len(QUANTITIES) or
            not all(isinstance(c, Channel) for c in channels) or
            tuple(c.quantity for c in channels) != QUANTITIES):
        raise ValidationError('choose explicit x, velocity, angle, rate, acceleration, brake channels')
    available = {s.channel: s for s in observation.samples}
    state, valid, ages = [], [], []
    for index, transform in ((0, float), (1, float), (2, math.sin), (2, math.cos),
                             (3, float), (4, float), (5, float)):
        sample = available.get(channels[index])
        usable = sample is not None and sample.validity == 'valid' and sample.value is not None
        state.append(transform(sample.value) if usable else 0.)
        valid.append(int(usable))
        ages.append(-1. if sample is None or sample.effective_ns is None else
                    (observation.available_ns-sample.effective_ns)/1e9)
    return {'state': state, 'valid': valid, 'age_s': ages}
