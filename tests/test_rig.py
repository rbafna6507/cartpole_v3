"""Real PTY integration: executor timing and failure behavior without hardware."""
from dataclasses import replace
import json
import os
import pty
import select
import tempfile
import threading
import time
import tty
import termios
import unittest
from cartpole_sdk.contracts import ActionRequest, ValidationError
from cartpole_sdk.rig import Rig, RigError
from cartpole_sdk.rig.protocol import decode, encode

class FDTransport:
    def __init__(self, port):
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        tty.setraw(self.fd)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        self.closed = False
    def read(self, size):
        if not select.select([self.fd], [], [], .005)[0]:
            return b''
        return os.read(self.fd, size)
    def write(self, data):
        if not select.select([], [self.fd], [], .05)[1]:
            raise TimeoutError('write timeout')
        return os.write(self.fd, data)
    def close(self):
        self.closed = True
        os.close(self.fd)

class Firmware:
    """Independent pinned-protocol executor on a PTY with real lease expiry."""
    def __init__(self, *, tick_origin=0, us_origin=0):
        self.master, self.slave = pty.openpty()
        os.set_blocking(self.master, False)
        tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self.sid = 0
        self.tick_origin, self.us_origin = tick_origin, us_origin
        self.origin = time.monotonic()
        self.mode = 0
        self.flags = 5
        self.fault = 0
        self.seq = 0
        self.expiry = None
        self.watchdog = None
        self.duration = None
        self.commands = []
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.quiet = False
        self.ignore = None
        self.rate_hz = 100
        self.last_emitted_tick = 0
        self.publish_commands = True
        self.hello_prefix = b''
        self.home_complete_ms = None
        self.home_started = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    def ticks(self):
        return int((time.monotonic()-self.origin)*1000)
    def send(self, payload):
        # Match firmware STATUS/emitTelemetry: an explicit snapshot restarts the
        # periodic publication interval. Otherwise the PTY invents a short next
        # interval that correctly trips a host policy's cadence guard.
        if payload.startswith('V2 S '):
            self.last_emitted_tick = self.ticks()
        with self.lock:
            try:
                os.write(self.master, encode(payload))
            except BlockingIOError:
                pass
    def state(self, *, sid=None, tick=None, theta='0.2'):
        elapsed = self.ticks()
        return f'V2 S {self.sid if sid is None else sid} {(self.tick_origin+elapsed)&0xffffffff if tick is None else tick} {(self.us_origin+elapsed*1000)&0xffffffff} {self.mode} {self.flags} {theta} 0.19 0.4 0.01 0 0 0 {self.seq} {self.fault} 0 0 0'
    def command(self, payload):
        p = payload.split()
        self.commands.append(p)
        if p[1] == self.ignore:
            return
        if p[1] == 'HELLO':
            if self.hello_prefix:os.write(self.master,self.hello_prefix)
            self.sid = int(p[2]); self.send(f'V2 HELLO {self.sid} 1 1000'); return
        if p[1] == 'INFO':
            self.send(f'V2 INFO {self.sid} executor-v1 270 1 32 0 1 0 0.26'); return
        if p[1] == 'STATUS':
            self.send(self.state()); return
        if p[1] == 'PING':
            self.watchdog = self.ticks()+200; return
        if p[1] in ('A', 'V'):
            seq, observed, ttl = map(int, p[3:6])
            now = (self.tick_origin+self.ticks())&0xffffffff
            age = (now-observed)&0xffffffff
            if int(p[2]) != self.sid or seq <= self.seq or age >20 or not 1<=ttl<=50 or age >=ttl:
                self.send(f'V2 ERR {self.sid} stale'); return
            self.seq = seq; self.expiry = self.ticks()+ttl-age; return
        if p[1] == 'STOP':
            self.mode=4 if self.fault else 0; self.flags &= ~2; self.expiry=None
        if p[1] == 'RATE':self.rate_hz=int(p[3])
        if p[1] == 'CLEAR': self.fault=0; self.mode=0
        if p[1] == 'SWITCHES':
            if p[3]=='1':self.flags |=8
            else:self.flags &=~8
        if p[1] == 'CENTER': self.flags |= 4
        if p[1] == 'HOME':
            self.home_started=self.ticks()
            self.mode=13; self.flags |=2; self.duration=self.ticks()+2000; self.watchdog=self.ticks()+200
        if p[1] in ('ARM', 'MODE'):
            self.mode = 12 if p[1]=='ARM' else {'auto':2,'bal':3,'manual':1}[p[3]]
            self.flags |=2; self.seq=0
            self.expiry=self.ticks()+20 if p[1]=='ARM' else None
            self.duration = self.ticks()+int(p[-1]); self.watchdog=self.ticks()+200
        self.send(f'V2 OK {self.sid} {p[1]}')
        if self.publish_commands:self.send(self.state())
    def run(self):
        buffer=b''
        while not self.done.is_set():
            try:
                if select.select([self.master],[],[],.002)[0]:
                    buffer += os.read(self.master,1024)
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1)
                        if line==b'stop':
                            self.mode=0; self.flags &=~2; self.expiry=None
                        else:
                            self.command(decode(line+b'\n'))
                now=self.ticks()
                if self.flags&2 and any(deadline is not None and now>=deadline for deadline in (self.expiry,self.watchdog,self.duration)):
                    self.flags &=~2; self.mode=0; self.fault=1; self.expiry=None
                if self.mode==13 and self.home_complete_ms is not None and now-self.home_started>=self.home_complete_ms:
                    self.mode=0; self.flags=(self.flags&~2)|4
                if self.sid and self.rate_hz and not self.quiet and now-self.last_emitted_tick>=1000/self.rate_hz:
                    self.send(self.state())
            except OSError:
                break
    def close(self):
        self.done.set(); self.thread.join(.5)
        os.close(self.master); os.close(self.slave)

class RigTests(unittest.TestCase):
    def setUp(self): self.fw=Firmware()
    def tearDown(self): self.fw.close()
    def rig(self, **kwargs):
        return Rig(self.fw.port, transport_factory=FDTransport, command_timeout=.2, **kwargs)
    def wait_until(self, predicate, timeout=.5):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate(): return
            time.sleep(.002)
        self.fail('timed out waiting for condition')
    def test_context_handshake_never_arms_and_closes_stopped(self):
        with self.rig() as rig:
            self.assertFalse(rig.state.enabled)
            self.assertEqual(rig.session.clock_id, f'teensy:{rig.state.sid}')
            self.assertFalse(any(c[1] in ('ARM','MODE') for c in self.fw.commands))
            rig.center(); rig.start('auto', .2)
            self.assertTrue(rig.state.enabled)
        self.assertEqual([c for c in self.fw.commands if c[1]!='STATUS'][-1][1], 'STOP')
        self.assertFalse(self.fw.flags&2)
    def test_exclusive_device_owner_even_for_injected_transport(self):
        with self.rig():
            with self.assertRaises(RigError):
                with self.rig(): pass
    def test_observations_name_measurement_and_estimate_sources(self):
        with self.rig() as rig:
            obs=rig.observation()
            self.assertEqual(obs.available_ns, rig.state.unwrapped_us*1000)
            values={(s.quantity,s.source):s for s in obs.samples}
            self.assertEqual(values['position','step'].kind, 'estimated')
            self.assertEqual(values['velocity','step'].method, 'commanded_velocity')
            self.assertEqual(values['angle','magnetic'].value, .2)
            self.fw.flags &=~1; previous=rig.state.tick
            rig.wait_state(after_tick=previous)
            sample=next(s for s in rig.observation().samples if s.source=='magnetic')
            self.assertEqual(sample.validity,'invalid'); self.assertIsNone(sample.value)
    def test_acceleration_anchor_default_ttl_and_heartbeat_cannot_renew(self):
        with self.rig(heartbeat_interval=.005) as rig:
            rig.arm(.3); rig.set_acceleration(.1)
            self.wait_until(lambda:any(c[1]=='A' for c in self.fw.commands))
            action=next(c for c in self.fw.commands if c[1]=='A')
            self.assertEqual(action[5], '20')
            self.wait_until(lambda: not self.fw.flags&2)
            self.assertTrue(any(c[1]=='PING' for c in self.fw.commands))
            self.wait_until(lambda:rig.error is not None)
            with self.assertRaises(RigError): rig.arm(.1)
            rig.stop(); rig.clear_fault()
            self.assertIsNone(rig.error)
    def test_invalid_values_write_no_action(self):
        with self.rig() as rig:
            rig.arm(.3)
            for value in (float('nan'),float('inf'),True,25.01):
                with self.subTest(value=value), self.assertRaises(ValidationError): rig.set_acceleration(value)
            self.assertFalse(any(c[1]=='A' for c in self.fw.commands))
    def test_stale_host_receipt_does_not_renew_board_anchor(self):
        with self.rig() as rig:
            rig.arm(.3); self.fw.quiet=True; time.sleep(.025)
            with self.assertRaises((ValidationError,RigError)): rig.set_acceleration(.1)
            self.assertFalse(any(c[1]=='A' for c in self.fw.commands))
    def test_cross_clock_sequence_expiry_rejected_before_write(self):
        with self.rig() as rig:
            rig.arm(.3); obs=rig.observation()
            request=ActionRequest(obs.session_id,obs.clock_id,1,obs.sequence,obs.available_ns,obs.available_ns+20_000_000,.1)
            for bad in (replace(request,clock_id='host'),replace(request,session_id='other'),replace(request,observation_sequence=obs.sequence+1000),replace(request,sequence=0),replace(request,expires_at_ns=request.issued_at_ns+60_000_000)):
                with self.subTest(bad=bad), self.assertRaises(ValidationError): rig.step(bad)
            rig.step(request)
            with self.assertRaises(ValidationError): rig.step(request)
            self.wait_until(lambda:any(c[1]=='A' for c in self.fw.commands))
            self.assertEqual(len([c for c in self.fw.commands if c[1]=='A']),1)
    def test_bad_frame_while_active_stops_and_latches(self):
        for damage in (b'@junk*0000\n', b'x'*300+b'\n', encode(self.fw.state(sid=1)), encode(self.fw.state(theta='nan'))):
            with self.subTest(damage=damage):
                with self.rig() as rig:
                    rig.start('auto',.3)
                    os.write(self.fw.master,damage)
                    self.wait_until(lambda:rig.error is not None)
                    self.wait_until(lambda:not self.fw.flags&2)
                    with self.assertRaises(RigError):rig.start('auto',.1)
                self.fw.fault=0
    def test_out_of_order_state_latches_and_stops(self):
        with self.rig() as rig:
            rig.start('auto',.3)
            self.fw.send(self.fw.state(tick=(rig.state.tick-1)&0xffffffff))
            self.wait_until(lambda:rig.error is not None)
            self.wait_until(lambda:not self.fw.flags&2)
    def test_command_timeout_stops_without_reconnect(self):
        with self.rig() as rig:
            self.fw.ignore='MODE'
            with self.assertRaises(RigError):rig.start('auto',.2)
            self.assertIsNotNone(rig.error)
            self.wait_until(lambda:any(c[1]=='STOP' for c in self.fw.commands))
            self.assertEqual(len([c for c in self.fw.commands if c[1]=='HELLO']),1)
    def test_home_requires_switch_profile_and_motion_requires_home(self):
        with self.rig() as rig:
            with self.assertRaises(RigError):rig.home()
            self.fw.flags &=~4
            self.wait_until(lambda:not rig.state.homed)
            with self.assertRaises(RigError):rig.start()
    def test_wait_timeout_and_none_step_never_send_action(self):
        with self.rig() as rig:
            self.fw.quiet=True; time.sleep(.015)
            with self.assertRaises(TimeoutError):rig.wait_state(after_tick=rig.state.tick,timeout=.03)
            rig.step(None)
            self.assertFalse(any(c[1]=='A' for c in self.fw.commands))
    def test_recording_has_states_actions_and_terminal_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.rig(log_dir=directory) as rig:
                rig.start('auto', .2)
                path=rig.log_path
            records=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(records[0]['type'],'session')
            self.assertTrue(any(r['type']=='state' for r in records))
            self.assertTrue(any(r['type']=='command' and r['payload'].split()[1]=='STOP' for r in records))
            self.assertEqual(records[-1]['type'],'termination')
    def test_write_failure_latches_and_requests_stop(self):
        class BrokenActionWrite(FDTransport):
            def write(self,data):
                if b'V2 A ' in data:raise OSError('simulated serial write failure')
                return super().write(data)
        with Rig(self.fw.port,transport_factory=BrokenActionWrite,command_timeout=.2) as rig:
            rig.arm(.2)
            with self.assertRaises(RigError):rig.set_acceleration(.1)
            self.assertIsNotNone(rig.error)
            self.wait_until(lambda:not self.fw.flags&2)

    def test_receipt_older_than_100ms_rejected_even_while_stopped(self):
        with self.rig() as rig:
            self.fw.quiet=True; time.sleep(.12)
            with self.assertRaises(ValidationError):rig.observation()

    def test_old_observation_with_fresh_latest_state_cannot_extend_lease(self):
        with self.rig() as rig:
            rig.arm(.3); old=rig.observation(); rig.set_acceleration(.1,ttl_ms=50)
            time.sleep(.025)
            bad=ActionRequest(old.session_id,old.clock_id,2,old.sequence,old.available_ns,old.available_ns+50_000_000,.1)
            with self.assertRaises(ValidationError):rig.step(bad)
            self.assertEqual(len([c for c in self.fw.commands if c[1]=='A']),1)

    def test_read_failure_stops_and_latches(self):
        class BrokenRead(FDTransport):
            fail=False
            def read(self,size):
                if self.fail:raise OSError('simulated disconnected read')
                return super().read(size)
        with Rig(self.fw.port,transport_factory=BrokenRead,command_timeout=.2) as rig:
            rig.start('auto',.2)
            rig._transport.fail=True
            self.wait_until(lambda:rig.error is not None)
            self.wait_until(lambda:not self.fw.flags&2)

    def test_homing_telemetry_loss_stops_while_heartbeat_is_alive(self):
        with self.rig() as rig:
            rig.configure_switches(True); rig.home()
            self.fw.quiet=True
            self.wait_until(lambda:rig.error is not None)
            self.wait_until(lambda:not self.fw.flags&2)

    def test_unhomed_cart_position_has_no_valid_physical_origin(self):
        with self.rig() as rig:
            self.fw.flags &=~4; rig.wait_state(after_tick=rig.state.tick)
            position=next(s for s in rig.observation().samples if s.quantity=='position')
            self.assertEqual(position.validity,'invalid')
            self.assertIsNone(position.value)

    def test_connect_to_stopped_fault_mode_allows_explicit_clear(self):
        self.fw.fault=1; self.fw.mode=4
        with self.rig() as rig:
            self.assertIsNotNone(rig.error)
            self.assertFalse(rig.state.enabled)
            with self.assertRaises(RigError):rig.start('auto',.1)
            rig.stop(); rig.clear_fault()
            self.assertIsNone(rig.error)
            self.assertEqual(rig.state.mode,0)

    def test_rate_zero_stopped_commands_request_fresh_status(self):
        self.fw.publish_commands=False
        with self.rig(rate_hz=0) as rig:
            time.sleep(.12)
            rig.configure_switches(False); rig.center(); rig.stop(); rig.clear_fault()
            self.assertTrue(rig.state.homed)
            self.assertIsNone(rig.error)
            self.assertGreaterEqual(len([c for c in self.fw.commands if c[1]=='STATUS']),5)
            with self.assertRaises(RigError):rig.start('auto',.1)
            rig.configure_switches(True)
            with self.assertRaises(RigError):rig.home()

    def test_negotiation_discards_partial_old_tail_and_old_session_reply(self):
        self.fw.hello_prefix=b'old trailing*0000\n'+encode('V2 ERR 17 stale')
        with self.rig() as rig:
            self.assertIsNone(rig.error)
            self.assertFalse(rig.state.enabled)
            self.assertTrue(any(r['type']=='negotiation_discard' for r in rig.records))

    def test_switch_profile_configuration_is_stopped_and_explicit(self):
        with self.rig() as rig:
            with self.assertRaises(ValidationError):rig.configure_switches(1)
            rig.configure_switches(True)
            self.assertTrue(rig.state.switch_profile)
            with self.assertRaises(RigError):rig.center()
            rig.configure_switches(False)
            self.assertFalse(rig.state.switch_profile)
            rig.center(); rig.start('manual',.2)
            with self.assertRaises(RigError):rig.configure_switches(True)

    def test_board_tick_and_microsecond_rollovers_are_independent(self):
        class ClockedFirmware(Firmware):
            elapsed_ms=0
            def ticks(self):return self.elapsed_ms

        # Freeze board time through handshake: without an observed pre-wrap
        # sample, the client cannot infer a counter's earlier epoch. Keep real
        # PTY framing, parsing and independent state-clock unwrapping.
        self.fw.close();self.fw=ClockedFirmware(tick_origin=0xfffffff0,us_origin=0xfffff000)
        self.fw.publish_commands=False  # First validated frame comes from STATUS.
        with self.rig(rate_hz=0) as rig:
            previous=rig.observation()
            self.assertEqual((rig.state.unwrapped_tick,rig.state.unwrapped_us),
                             (0xfffffff0,0xfffff000))
            for elapsed_ms,tick,us,unwrapped_tick,unwrapped_us in (
                    (5,0xfffffff5,904,0xfffffff5,0x100000388),
                    (20,4,15904,0x100000004,0x100003e20)):
                self.fw.elapsed_ms=elapsed_ms
                self.fw.send(self.fw.state())
                state=rig.wait_state(after_tick=previous.sequence)
                self.assertEqual((state.tick,state.us),(tick,us))
                self.assertEqual((state.unwrapped_tick,state.unwrapped_us),
                                 (unwrapped_tick,unwrapped_us))
                current=rig.observation()
                self.assertEqual(current.sequence,unwrapped_tick)
                self.assertEqual(current.available_ns,unwrapped_us*1000)
                self.assertGreater(current.sequence,previous.sequence)
                self.assertGreater(current.available_ns,previous.available_ns)
                previous=current
