"""Offline local trusted-code runner. The portal separately isolates uploaded code."""
import argparse
from hashlib import sha256
from pathlib import Path
from cartpole_sdk import ActionLimits
from cartpole_sdk.developer import load_local_policy, run_profile_policy
from cartpole_sdk.recording import read_recording
from cartpole_sdk.replay import replay, write_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('simulate', 'replay'))
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('policy.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    policy = load_local_policy(args.policy)
    if args.mode == 'simulate':
        result = run_profile_policy(policy, args.output, seconds=args.seconds, seed=args.seed,
                                    policy_sha256=sha256(args.policy.read_bytes()).hexdigest())
        print(result)
    else:
        if args.recording is None:
            parser.error('--recording is required for replay')
        session, observations = read_recording(args.recording)
        steps = replay(session, observations, policy, ActionLimits(25., 20_000_000, 20_000_000))
        write_trace(args.output, session, steps)
        print(f'Replayed {len(steps)} inputs; actions do not alter recorded observations.')


if __name__ == '__main__':
    main()
