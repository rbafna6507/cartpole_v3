"""Export a frozen numeric artifact as one bounded portal create_policy module.

The generated file embeds exact artifact bytes and explicit source selection.
It imports only stdlib and the readonly SDK; no external weight file is needed.
It deliberately retains the saved policy cadence and hardware_accepted=False.
"""
from hashlib import sha256
from pathlib import Path
import re

from cartpole_sdk.contracts import ValidationError
from .artifact import read_artifact

MAX_SOURCE_BYTES = 65_536
# Verify behavior-bearing inference modules, never filenames supplied by an
# artifact. This is compatibility/tamper detection, not an execution sandbox.
RUNTIME_FILES = (
    'contracts.py', 'policies/__init__.py', 'policies/host.py',
    'learning/__init__.py', 'learning/artifact.py',
    'control/parameters.py', 'control/tuned_parameters.json',
)
SOURCES = {
    'simulation': (
        ('cart','position','simulation','sim:upright+x'),
        ('cart','velocity','simulation','sim:upright+x'),
        ('pole','angle','simulation','sim:upright+x'),
        ('pole','angular_velocity','simulation','sim:upright+x'),
    ),
    'magnetic-step': (
        ('cart','position','step','cart-step-home'),
        ('cart','velocity','step','cart-step-home'),
        ('pole','angle','magnetic','encoder-zero'),
        ('pole','angular_velocity','magnetic','encoder-zero'),
    ),
}


def export_policy(artifact_path, output_path, *, source):
    """Export one explicit source mapping; never infer it from observations.

    The caller must stage the exported source through the real portal sandbox.
    Exporting does not approve an acceleration ceiling or physical execution.
    """
    if source not in SOURCES:
        raise ValidationError('source must be simulation or magnetic-step')
    artifact_path = Path(artifact_path)
    artifact = read_artifact(artifact_path)
    raw = artifact_path.read_bytes()
    # Re-validate the exact bytes embedded if the file changed between reads.
    from cartpole_sdk.recording import strict_json_loads
    from .artifact import validate_artifact
    current = validate_artifact(strict_json_loads(raw.decode('utf-8')))
    if current != artifact:
        raise ValidationError('artifact changed during export')
    expected = artifact['provenance'].get('code_sha256', {})
    if not isinstance(expected, dict) or any(
            not isinstance(expected.get(name),str) or re.fullmatch('[0-9a-f]{64}',expected[name]) is None
            for name in RUNTIME_FILES):
        raise ValidationError('artifact lacks supported inference source hashes')
    hashes = {name:expected[name] for name in RUNTIME_FILES}
    digest = sha256(raw).hexdigest()
    code = f'''"""Frozen learned policy for the isolated portal child.

Source: {source}; saved cadence: {artifact['rate_hz']} Hz.
Four explicitly selected SDK channels; raw acceleration demand up to 25 m/s².
The trusted broker and MCU own ceilings, leases and the single 1 kHz governor.
Provisional simulator training only. Hardware acceptance is false.
No external weights or third-party inference packages are loaded.
"""
from hashlib import sha256
from pathlib import Path
import cartpole_sdk
from cartpole_sdk.contracts import Channel, ValidationError
from cartpole_sdk.recording import strict_json_loads
from cartpole_sdk.learning.artifact import LearnedPolicy

SENSOR_SOURCE = {source!r}
RATE_HZ = {artifact['rate_hz']!r}
ACTION_SEMANTICS = {artifact['action_semantics']!r}
ARTIFACT_SHA256 = {digest!r}
RUNTIME_SHA256 = {hashes!r}
CHANNELS = {SOURCES[source]!r}
ARTIFACT_JSON = {raw.decode('utf-8')!r}


def create_policy():
    if sha256(ARTIFACT_JSON.encode('utf-8')).hexdigest() != ARTIFACT_SHA256:
        raise ValidationError('embedded learned artifact digest mismatch')
    root = Path(cartpole_sdk.__file__).resolve().parent
    for name, expected in RUNTIME_SHA256.items():
        if sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValidationError('SDK inference code differs from trained provenance')
    artifact = strict_json_loads(ARTIFACT_JSON)
    policy = LearnedPolicy(artifact, channels=tuple(Channel(*fields) for fields in CHANNELS))
    if policy.rate_hz != RATE_HZ or artifact['action_semantics'] != ACTION_SEMANTICS:
        raise ValidationError('exported policy contract mismatch')
    return policy
'''
    encoded = code.encode('utf-8')
    if len(encoded) > MAX_SOURCE_BYTES:
        raise ValidationError('embedded policy exceeds portal 64 KiB source limit')
    compile(code, str(output_path), 'exec')  # parse only; never import submitted output
    output = Path(output_path)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(encoded)
    return dict(schema=1, source_sha256=sha256(encoded).hexdigest(), source_bytes=len(encoded),
        artifact_sha256=digest, sensor_source=source, rate_hz=artifact['rate_hz'],
        runtime_sha256=hashes, exporter_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        action_semantics=artifact['action_semantics'], hardware_accepted=False)
