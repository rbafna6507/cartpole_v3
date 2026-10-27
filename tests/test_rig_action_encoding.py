"""SDK action bytes must survive the production executor's float parser."""
from decimal import Decimal
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from cartpole_sdk.contracts import ActionRequest, Session, ValidationError
from cartpole_sdk.rig import Rig
from cartpole_sdk.rig.protocol import decode, encode

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native_action_parser(tmp_path_factory):
    build = tmp_path_factory.mktemp('action-wire-parser')
    source = build / 'parser.cpp'
    source.write_text(r'''
#include "executor_core.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
int main() {
    executor::Core core;
    core.tick=100;
    core.sample(true,0);
    auto send=[&](const char* payload) {
        char line[256];
        assert(executor::frame(payload,line,sizeof line)>0);
        return core.command(line);
    };
    assert(send("V2 HELLO 7").kind==executor::HELLO);
    assert(send("V2 SWITCHES 7 0").kind==executor::OK);
    assert(send("V2 CENTER 7").kind==executor::OK);
    assert(send("V2 ARM 7 1000").kind==executor::OK);
    std::string line;
    std::getline(std::cin,line);
    const auto reply=core.command(line.c_str());
    std::cout << std::setprecision(17)
              << "{\"accepted\":" << (reply.kind==executor::SILENT ? "true" : "false")
              << ",\"reply\":\"" << reply.text << "\",\"demand\":" << core.demand
              << ",\"last_seq\":" << core.last_seq
              << ",\"action_end\":" << core.action_end << "}\n";
}
''')
    binary = build / 'parser'
    result = subprocess.run([shutil.which('c++') or '/usr/bin/c++', '-std=c++17',
        '-O2', '-fno-fast-math', '-I', str(ROOT/'firmware/teensy_executor'),
        str(source), '-o', str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    def parse(line):
        result = subprocess.run([str(binary)], input=line, capture_output=True, check=True)
        return json.loads(result.stdout)
    return parse


class CapturedTransport:
    def __init__(self):
        self.writes = []

    def write(self, line):
        self.writes.append(line)
        return len(line)


def action_wire(value, api, monkeypatch):
    # Only the physical transport and clock are substituted. The SDK validates
    # the parsed snapshot/request, serializes the action and frames its CRC.
    monkeypatch.setattr('cartpole_sdk.rig.client.time.monotonic_ns', lambda: 1_000_000_000)
    rig = Rig('wire-only')
    rig._sid = 7
    rig.session = Session('7', 'teensy:7')
    rig._connected = rig._handshake = True
    rig._transport = CapturedTransport()
    rig._accept('V2 S 7 100 100000 12 7 3.14 3.14 0 0 0 0 0 0 0 0 0 0', 1_000_000_000)
    if api == 'step':
        observation = rig.observation()
        request = ActionRequest(observation.session_id, observation.clock_id, 1,
            observation.sequence, observation.available_ns,
            observation.available_ns + 20_000_000, value)
        rig.step(request)
    else:
        rig.set_acceleration(value)
    assert len(rig._transport.writes) == 1
    return rig._transport.writes[0]


def test_native_parser_reproduces_pilot_underflow_without_relaxing_parser(native_action_parser):
    accepted = native_action_parser(encode('V2 A 7 629 100 20 1.19172592e-38'))
    assert accepted['accepted'] and accepted['last_seq'] == 629
    for token in ('1.01935274e-38', '-1.01935274e-38', '1e-999'):
        rejected = native_action_parser(encode(f'V2 A 7 630 100 20 {token}'))
        assert not rejected['accepted'] and rejected['reply'] == 'args'
        assert rejected['last_seq'] == 0


@pytest.mark.parametrize('api', ['step', 'set_acceleration'])
@pytest.mark.parametrize('value', [
    0., -0., 1.01935274e-38, -1.01935274e-38,
    1.19172592e-38, -1.19172592e-38,
    math.ulp(0.), -math.ulp(0.), 2.**-126, -(2.**-126),
    math.nextafter(5e-10, 0.), math.nextafter(5e-10, math.inf),
    -math.nextafter(5e-10, 0.), -math.nextafter(5e-10, math.inf),
    .1234567890123, -.1234567890123, 25., -25.,
])
def test_sdk_acceleration_wire_is_native_parseable_with_bounded_quantization(
        api, value, monkeypatch, native_action_parser):
    line = action_wire(value, api, monkeypatch)
    fields = decode(line).split()
    # Check the decimal encoding error separately from downstream float32
    # conversion. Decimal.from_float represents the input float exactly.
    error = abs(Decimal(fields[-1]) - Decimal.from_float(value))
    assert error <= Decimal('0.0000000005')
    parsed = native_action_parser(line)
    assert parsed['accepted'], (value, fields[-1], parsed)
    assert parsed['last_seq'] == 1
    assert parsed['action_end'] == 120  # Same observed tick and 20 ms lease.
    if abs(value) < 5e-10:
        assert parsed['demand'] == 0.
    else:
        # Decimal quantization plus the existing binary32 round-to-nearest.
        assert abs(parsed['demand'] - value) <= 5e-10 + abs(value)*2.**-23


@pytest.mark.parametrize('api', ['step', 'set_acceleration'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 25.000001])
def test_acceleration_encoding_keeps_nonfinite_and_range_rejections(api, value, monkeypatch):
    with pytest.raises(ValidationError):
        action_wire(value, api, monkeypatch)
