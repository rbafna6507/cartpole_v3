#!/usr/bin/env python3
"""Run a numeric SDK actor in shadow, or explicitly execute development actions."""
import argparse
import json
from pathlib import Path


def main(argv=None, *, rig_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True, help='Exported actor.json; never a Torch/pickle checkpoint.')
    parser.add_argument('--port', required=True, help='Explicit serial device path.')
    parser.add_argument('--development-only', action='store_true', help='Acknowledge unqualified development actor; does not grant hardware acceptance.')
    parser.add_argument('--execute', action='store_true', help='Explicitly arm and submit actions; omitted means stopped shadow inference.')
    parser.add_argument('--duration', type=float, default=1., help='Bounded run duration in seconds.')
    parser.add_argument('--max-acceleration', type=float, default=.25, help='Executed raw acceleration ceiling in m/s²; excess demands fault instead of clipping.')
    parser.add_argument('--sensing-config', type=Path, help='Required explicit camera calibration, angular alignment and current clock binding for vision/fused actors.')
    parser.add_argument('--preflight-seconds', type=float, default=2., help='Stopped causal sensor warmup deadline.')
    parser.add_argument('--log-dir', type=Path, help='Optional SDK JSONL trace directory.')
    args = parser.parse_args(argv)
    if not args.development_only:
        parser.error('--development-only is required for these unqualified numeric actors')
    try:
        from cartpole_sdk.policies.neural import MAX_BYTES, validate_actor, load_neural_policy
        from cartpole_sdk.policies.rig_observation import observation_contract, rig_channels
        from cartpole_sdk.recording import strict_json_loads
        from cartpole_sdk.rig import Rig
        from cartpole_sdk.rig.neural_runner import run_neural_policy, validate_sensing_config
        def read_json(path, max_bytes):
            with path.open('rb') as stream:
                raw = stream.read(max_bytes+1)
            if len(raw) > max_bytes:
                raise ValueError('JSON input exceeds bounded size')
            result = strict_json_loads(raw.decode('utf-8'))
            if not isinstance(result, dict):
                raise ValueError('JSON object required')
            return result
        actor = validate_actor(read_json(args.actor, MAX_BYTES))
        contract = actor.get('observation_contract') or observation_contract(angle_source=actor['angle_source'])
        channels = rig_channels(angle_source=contract['angle_source'], angle_frame_id=contract['angle_frame_id'])
        policy = load_neural_policy(args.actor, channels=channels)
        if policy.executor_parameters is None:
            raise ValueError('legacy actor lacks complete executor snapshot; re-export before live inference')
        config = None if args.sensing_config is None else read_json(args.sensing_config, 262144)
        if contract['angle_source'] in ('vision', 'fused'):
            validate_sensing_config(config, contract)
        elif config is not None:
            raise ValueError('camera binding supplied for an encoder actor')
        factory = Rig if rig_factory is None else rig_factory
        with factory(args.port, rate_hz=100, log_dir=args.log_dir) as rig:
            result = run_neural_policy(rig, policy, development_only=True, execute=args.execute,
                duration_s=args.duration, max_acceleration=args.max_acceleration,
                sensing_config=config, preflight_s=args.preflight_seconds)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'cartpole numeric inference: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
