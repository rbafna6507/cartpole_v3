# Developer command reference

Run these from the repository root. Use a fresh output path for each run. `uv run`
selects the requested extras; include `--extra rl` for training and `--extra sim`
for simulation. The console is for an operator with a prepared physical rig, not
for remote users to bypass the portal.

| Task | Command |
| --- | --- |
| Install simulation + tests | `uv sync --locked --extra sim` |
| Run regression tests | `uv run --locked --extra sim pytest -q` |
| Simulate starter | `uv run --locked --extra sim python examples/controller_starter/run.py simulate --seconds 5 --output runs/starter-001` |
| Simulate existing policy | `uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy policies/working.json --seconds 25 --output runs/actor-001` |
| Build host physics controller | `uv run --locked python -m cartpole_sdk.control.native_build --source-root .` |
| Simulate physics example | `uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy examples/controller_starter/physics_policy.py --seconds 25 --output runs/physics-001` |
| Replay | `uv run --locked python examples/controller_starter/run.py replay --policy examples/controller_starter/policy.py --recording runs/starter-001/observations.jsonl --output runs/replay-001.jsonl` |
| View learned actor | `uv run --locked --extra sim cartpole-view-policy --actor policies/working.json --start hanging --seconds 25` |
| View reference controller | `uv run --locked --extra sim cartpole-view-policy --controller reference --start hanging --seconds 25` |
| Test viewer without window | Add `--headless` to either viewer command |
| Reproduce system-ID plots | `uv run --locked --extra plots python system_id/reproduce.py --output runs/id-review` |
| Train/export smoke | `uv run --locked --extra rl python examples/controller_starter/train.py --output runs/train-001 --objective controlled-swing-v3 --steps 1024 --checkpoint-every 1024 --seconds 5` |
| Read-only sensor collection | `uv run --locked cartpole-collect --url http://127.0.0.1:3090 --duration 30 --output runs/sensing-001.jsonl` |

`run.py`: `simulate|replay`, `--policy PATH`, `--output PATH` (required),
`--recording PATH` (required for replay), `--seconds` (default 5), `--seed` (default 0).
Numeric JSON and trusted local Python policies are supported. Local Python import
is **not sandboxed**; only execute code you trust on your own computer.

Viewer: `--controller learned|reference`, `--actor PATH`, `--experiment PATH`,
`--start hanging|upright`, `--seed`, `--seconds`, `--duration` (wall-clock auto-close),
`--headless`. The reference viewer uses ideal sensors at 1 kHz; it is not the
100 Hz portable physics example. On macOS the entry point selects MuJoCo's
main-thread launcher. A desktop/display is required for a window.

Training and submission have additional configuration; use their `--help` to see
the exact installed options. The short training command verifies execution/export,
not learning success. For agent submission and authenticated collection, see
[Submission and data](SUBMISSION_AND_DATA.md).

## Operator console, separate from public submission

The current deployed launcher is `cartpole-console` on the Jetson. A source checkout
also supports `uv run cartpole`. Do not open the console while the hardware worker
owns the serial port. With a prepared, homed, calibrated rig:

```text
policy load policies/working.json
policy check swingup
policy swingup 25
```

`auto 25` selects onboard physics AUTO. `python-auto-trial 25` selects the saved
experimental external Python trial, not the new portable example. `python-auto`
retains its separate default pathway. `stop` requests STOP; `quit` closes the
console. `status`, `info`, `params`, `policy status` inspect state. `home` performs
horizontal homing; `upright` records an operator-positioned upright reference.
Do not run those calibration commands merely to bypass a compatibility error.

For a known measured rail difference, `policy rail-tolerance 10` permits up to
10 mm difference in **half-rail** compatibility for that selected policy (allowed
range 1–20 mm). It resets on policy load/unload; actual measured motion bounds and
other safety checks remain. Research actors staged after the controlled-swing
campaign are described in [training status](TRAINING_STATUS.md), not installed as
`working.json`.
