"""Strict wire parsing, bounded framing and distinct board clocks."""
import unittest
from cartpole_sdk.rig.protocol import ProtocolError, LineBuffer, crc16, encode, decode, parse_state, unwrap32

STATE = 'V2 S 7 10 10000 0 5 0.2 0.19 1.1 0.01 0 0 0 0 0 0 0 0'

class ProtocolTests(unittest.TestCase):
    def test_crc_ccitt_known_vector(self):
        self.assertEqual(crc16(b'123456789'), 0x29B1)

    def test_frame_roundtrip_and_strict_damage_rejection(self):
        frame = encode(STATE)
        self.assertEqual(decode(frame), STATE)
        for damaged in (frame[:-6]+b'0000\n', b'junk\n', frame+b'x', frame.replace(b'0.2', b'\t.2'), b'@'+b'x'*250+b'*0000\n'):
            with self.subTest(damaged=damaged), self.assertRaises(ProtocolError):
                decode(damaged)

    def test_state_rejects_nonfinite_trailing_signed_integer_and_bad_mode(self):
        for payload in (STATE+' extra', STATE.replace('0.2', 'nan'), STATE.replace(' 10 ', ' -1 '), STATE.replace(' 0 5 ', ' 99 5 '), STATE.replace(' 0.2 ', ' 1e999 '), STATE.replace(' 0.2 ', ' 1_0 ')):
            with self.subTest(payload=payload), self.assertRaises(ProtocolError):
                parse_state(payload, received_ns=1)
        state = parse_state(STATE, received_ns=123)
        self.assertEqual((state.sid, state.tick, state.us, state.received_ns), (7,10,10000,123))
        self.assertEqual(state.acceleration, 0)

    def test_bounded_buffer_drops_entire_oversize_line_then_resynchronizes(self):
        buffer = LineBuffer()
        with self.assertRaises(ProtocolError):
            buffer.feed(b'x'*300)
        self.assertLessEqual(buffer.buffered_bytes, 255)
        self.assertEqual(buffer.feed(b'garbage\n'+encode(STATE)), [encode(STATE)])

    def test_unsigned_rollover_and_out_of_order_rejection(self):
        self.assertEqual(unwrap32(3, 0xFFFFFFFE), 0x100000003)
        with self.assertRaises(ProtocolError):
            unwrap32(9, 10)
        self.assertEqual(unwrap32(10, 10), 10)
