"""Schema-preserving portal IPC encoding for immutable SDK observations."""
from dataclasses import asdict, fields
from cartpole_sdk.contracts import Observation, Sample

_OBSERVATION_FIELDS = tuple(f.name for f in fields(Observation) if f.name != 'samples')
_SAMPLE_FIELDS = tuple(f.name for f in fields(Sample))


def observation_payload(observation: Observation) -> dict:
    """Preserve the wire schema without deep-copying immutable scalar leaves.

    Allocate fresh containers; retain generic dataclass behavior for extensions.
    Validation remains at the existing contract and decode boundaries.
    """
    if type(observation) is not Observation or any(type(s) is not Sample for s in observation.samples):
        return asdict(observation)
    result = {name: getattr(observation, name) for name in _OBSERVATION_FIELDS}
    result['samples'] = tuple({name: getattr(sample, name) for name in _SAMPLE_FIELDS}
                              for sample in observation.samples)
    return result

