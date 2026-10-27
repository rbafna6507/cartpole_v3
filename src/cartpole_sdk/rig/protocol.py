"""Pinned V2 ASCII framing. Parsing has no transport or actuation side effects."""
from dataclasses import dataclass
from binascii import crc_hqx
import math
import re
from .parameters import PARAMETER_NAMES

MAX_LINE = 255
UINT32 = 0xFFFFFFFF

class ProtocolError(ValueError):
    """Invalid, damaged or unordered executor traffic."""

def crc16(data: bytes) -> int:
    # CCITT polynomial 0x1021, initial 0xFFFF, no reflection/final XOR.
    # Keep the wire checksum unchanged while avoiding per-bit Python work.
    return crc_hqx(data, 0xFFFF)

def encode(payload: str) -> bytes:
    try:
        body = payload.encode('ascii')
    except (UnicodeEncodeError, AttributeError) as exc:
        raise ProtocolError('payload must be ASCII') from exc
    if not body or any(c < 32 or c > 126 for c in body) or b'*' in body or b'@' in body:
        raise ProtocolError('invalid payload characters')
    line = b'@' + body + f'*{crc16(body):04X}\n'.encode('ascii')
    if len(line) > MAX_LINE:
        raise ProtocolError('oversize frame')
    return line

def decode(line: bytes) -> str:
    if not isinstance(line, bytes) or len(line) > MAX_LINE or not line.endswith(b'\n'):
        raise ProtocolError('invalid frame length or terminator')
    match = re.fullmatch(rb'@([\x20-\x7e]+)\*([0-9A-Fa-f]{4})\n', line)
    if match is None:
        raise ProtocolError('invalid framing')
    body, checksum = match.groups()
    if b'*' in body or b'@' in body or crc16(body) != int(checksum, 16):
        raise ProtocolError('CRC or payload error')
    return body.decode('ascii')

def uint32(token: str) -> int:
    if not re.fullmatch(r'[0-9]+', token) or len(token) > 10 or int(token) > UINT32:
        raise ProtocolError('invalid uint32')
    return int(token)

def unwrap32(raw: int, previous: int | None) -> int:
    if previous is None:
        return raw
    delta = (raw - (previous & UINT32)) & UINT32
    if delta >= 0x80000000:
        raise ProtocolError('out-of-order board counter')
    return previous + delta

@dataclass(frozen=True, slots=True)
class State:
    sid: int
    tick: int
    us: int
    mode: int
    flags: int
    theta: float
    angle_hat: float
    omega: float
    x: float
    velocity: float
    acceleration: float
    demand: float
    last_seq: int
    fault: int
    misses: int
    dropped: int
    brake_direction: int
    received_ns: int
    unwrapped_tick: int = 0
    unwrapped_us: int = 0

    @property
    def enabled(self):
        return bool(self.flags & 2)

    @property
    def homed(self):
        return bool(self.flags & 4)

    @property
    def healthy(self):
        return bool(self.flags & 1) and self.fault == 0

    @property
    def switch_profile(self):
        return bool(self.flags & 8)

def parse_state(payload: str, *, received_ns: int) -> State:
    parts = payload.split(' ')
    if len(parts) != 19 or parts[:2] != ['V2', 'S']:
        raise ProtocolError('invalid state field count')
    tokens = parts[2:]
    integers = [uint32(t) for t in tokens[:5]]
    if integers[0] == 0 or integers[3] > 15:
        raise ProtocolError('invalid session or mode')
    if not all(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',t) for t in tokens[5:12]):
        raise ProtocolError('invalid real grammar')
    try:
        values = [float(t) for t in tokens[5:12]]
    except ValueError as exc:
        raise ProtocolError('invalid real value') from exc
    if not all(math.isfinite(v) for v in values):
        raise ProtocolError('nonfinite state')
    tail = [uint32(t) for t in tokens[12:16]]
    if tokens[16] not in ('-1', '0', '1'):
        raise ProtocolError('invalid brake direction')
    return State(*integers, *values, *tail, int(tokens[16]), received_ns)

@dataclass(frozen=True, slots=True)
class ExecutorInfo:
    sid: int
    firmware: str
    encoder_upright_raw: int
    reference_saved: bool
    mag_status: int
    i2c_errors: int
    home_epoch: int
    measured_span_steps: int
    rail_half_m: float
    received_ns: int
    encoder_boot_conf: int | None = None
    encoder_conf: int | None = None
    encoder_conf_valid: bool | None = None
    encoder_raw: int | None = None
    # Time since the value changed, not proof of a fresh sensor transaction.
    encoder_raw_age_us: int | None = None
    encoder_recoveries: int | None = None


def parse_info(payload: str, *, received_ns: int) -> ExecutorInfo:
    parts = payload.split(' ')
    field_counts = {'executor-v1': 11, 'executor-v2-cart2': 11, 'executor-v3-encguard': 17, 'executor-v4-rehome': 17, 'executor-v5-calmodel': 17, 'executor-v6-measuredrail': 17, 'executor-v7-vertical': 17}
    if len(parts) < 4 or parts[:2] != ['V2', 'INFO'] or len(parts) != field_counts.get(parts[3]):
        raise ProtocolError('invalid diagnostic identity or field count')
    sid = uint32(parts[2])
    reference, saved, status, errors, epoch, span = map(uint32, parts[4:10])
    if not sid or reference > 4095 or saved > 1 or status > 255 or span > 20000:
        raise ProtocolError('invalid diagnostic range')
    if not re.fullmatch(r'[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', parts[10]):
        raise ProtocolError('invalid diagnostic rail value')
    rail = float(parts[10])
    if not math.isfinite(rail) or not 0 < rail <= (.375 if parts[3] in ('executor-v6-measuredrail', 'executor-v7-vertical') else .26):
        raise ProtocolError('invalid diagnostic rail bound')
    base = (sid, parts[3], reference, bool(saved), status, errors, epoch, span, rail, received_ns)
    if len(parts) == 11:
        return ExecutorInfo(*base)
    boot_conf, conf = (-1 if token == '-1' else uint32(token) for token in parts[11:13])
    valid, raw, raw_age, recoveries = map(uint32, parts[13:17])
    if boot_conf > 65535 or conf > 65535 or valid > 1 or raw > 4095:
        raise ProtocolError('invalid encoder diagnostic range')
    if valid and (conf < 0 or (conf & 0x3F0F) != 0x0300):
        raise ProtocolError('invalid encoder configuration claim')
    return ExecutorInfo(*base, boot_conf, conf, bool(valid), raw, raw_age, recoveries)

def parse_parameter(payload: str) -> tuple[str, float]:
    parts = payload.split(' ')
    if len(parts) != 5 or parts[:2] != ['V2', 'PARAM']:
        raise ProtocolError('invalid parameter field count')
    if not uint32(parts[2]) or parts[3] not in PARAMETER_NAMES:
        raise ProtocolError('invalid parameter session or name')
    if not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', parts[4]):
        raise ProtocolError('invalid parameter value grammar')
    value = float(parts[4])
    if not math.isfinite(value):
        raise ProtocolError('nonfinite parameter value')
    return parts[3], value

class LineBuffer:
    """Bounded incremental framing; discard through newline after overflow."""
    def __init__(self):
        self._buffer = bytearray()
        self._discarding = False

    @property
    def buffered_bytes(self):
        return len(self._buffer)

    def feed(self, data: bytes, *, raise_overflow=True) -> list[bytes]:
        lines = []
        overflow = False
        for byte in data:
            if self._discarding:
                if byte == 10:
                    self._discarding = False
                continue
            self._buffer.append(byte)
            if len(self._buffer) > MAX_LINE:
                self._buffer.clear()
                self._discarding = byte != 10
                overflow = True
            elif byte == 10:
                lines.append(bytes(self._buffer))
                self._buffer.clear()
        if overflow and raise_overflow:
            raise ProtocolError('oversize line discarded')
        return lines


@dataclass(frozen=True, slots=True)
class ZStatus:
    """Counted vertical positions; refs and bottom_mask are axis bitmasks.

    A bottom bit describes timed homing completion, not a contact sensor.
    Positions without a corresponding reference bit are not physical coordinates.
    """
    sid: int
    phase: int
    refs: int
    left_mm: float
    right_mm: float
    bottom_mask: int
    received_ns: int
    max_mm: float | None = None


def parse_z_status(payload: str, *, received_ns: int) -> ZStatus:
    parts = payload.split(' ')
    if len(parts) not in (8, 9) or parts[:2] != ['V2', 'Z']:
        raise ProtocolError('invalid vertical status field count')
    sid, phase, refs = map(uint32, parts[2:5])
    bottom = uint32(parts[7])
    if not sid or phase > 4 or refs > 3 or bottom > 3:
        raise ProtocolError('invalid vertical status range')
    if not all(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', t) for t in parts[5:7]):
        raise ProtocolError('invalid vertical position grammar')
    left, right = map(float, parts[5:7])
    if not all(math.isfinite(v) for v in (left, right)):
        raise ProtocolError('nonfinite vertical position')
    maximum = None
    if len(parts) == 9:
        if not re.fullmatch(r'[0-9]+(?:\.[0-9]*)?', parts[8]):
            raise ProtocolError('invalid vertical limit grammar')
        maximum = float(parts[8])
        if not math.isfinite(maximum) or not 10 <= maximum <= 200:
            raise ProtocolError('invalid vertical limit range')
    return ZStatus(sid, phase, refs, left, right, bottom, received_ns, maximum)
