# Portable SDK quickstart

Run from this checkout with Python 3.11+ and uv. These commands were tested on
macOS arm64 with Python 3.12.11 on 2026-10-27. They perform no hardware motion.

Install uv with `pipx install uv` if it is not already available, clone this
repository and enter it:

```sh
git clone https://github.com/rbafna6507/cartpole_v3.git
cd cartpole_v3
```

Allow about 30 minutes:
5 minutes for installation, 5 for the starter/replay, 10 for the physics and
actor examples, and 10 to inspect outputs and edit a controller. Dependency
downloads and short training may take longer on a new machine.

```sh
uv sync --extra sim
uv run --locked --extra sim pytest -q
uv run --locked --extra sim python examples/controller_starter/run.py simulate --seconds 5 --output runs/starter-sim-001
uv run --locked python examples/controller_starter/run.py replay --recording runs/starter-sim-001/observations.jsonl --output runs/starter-replay-001.jsonl
```

The suite passed 423 tests and 284 subtests. Simulation produced 500 observations;
replay consumed all 500. Choose a new output directory for subsequent simulations.
The starter `policy.py` damps cart velocity; it does not swing up or balance.
Replay evaluates actions against fixed recorded observations; it cannot show how
new actions would change the plant.

## Physics controller

A C++17 compiler is required. This explicitly builds the host binding to the
included firmware's swing, capture, and balance equations, then uses the same
100 Hz simulation runner and observation adapter:

```sh
uv run --locked python -m cartpole_sdk.control.native_build --source-root .
uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy examples/controller_starter/physics_policy.py --seconds 25 --output runs/physics-sim-001
```

`physics_policy.py` starts hanging, uses the named profile's parameters, and
switches from energy pumping to balance through the native controller. It is an
offline integration example. It does not reproduce all onboard firmware safety
state machines or validate host deadlines for a physical rig.

## Preserved trained actors

```sh
uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy policies/working.json --seconds 25 --output runs/working-sim-001
uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy policies/1m.json --seconds 25 --output runs/1m-sim-001
```

All three 25-second runs, seed 0, reached 2,500 steps without an environment fault
or absent action. Using observed filtered angle, upright means |wrapped angle|
< 0.2 radians. These results are single-seed smoke checks, not robust qualification:

| Policy | Upright fraction | Final 5 seconds upright | Maximum absolute cart x (m) |
| --- | ---: | ---: | ---: |
| Physics example | 90.92% | 100% | 0.215175 |
| working.json | 83.88% | 100% | 0.294075 |
| 1m.json | 73.00% | 100% | 0.294075 |

Outputs are local ignored `runs/` files. `run.json` records profile, seed, source
hash, termination and model configuration; `observations.jsonl` preserves policy
inputs and `trace.jsonl` records requested actions and executor diagnostics.

## Extend a controller

Implement `create_policy()` returning an object with `reset(session)` and
`act(observation)`. Return `Acceleration` in m/s² or `None` to abstain. Use
`ObservationAdapter` to select explicitly declared sources and retain validity,
age, frame and cadence checks. Clear state on every reset. Test in simulation,
then replay recorded inputs. Do not relabel vision as encoder measurements.

## Short training and export smoke

```sh
uv sync --extra sim --extra rl
uv run --locked --extra rl python examples/controller_starter/train.py --objective controlled-swing-v3 --steps 1024 --seconds 5 --checkpoint-every 1024 --output runs/training-smoke-001
uv run --locked --extra sim python examples/controller_starter/run.py simulate --policy runs/training-smoke-001/actor.json --seconds 5 --output runs/training-export-sim-001
uv run --locked python examples/controller_starter/run.py replay --policy runs/training-smoke-001/actor.json --recording runs/training-export-sim-001/observations.jsonl --output runs/training-export-replay-001.jsonl
```

This tested checkpoint roundtrip and numeric export parity, then loaded the
exported actor for 500-step simulation/replay. It did not train a qualified
controller: zero of four hanging scenarios succeeded and the development
selector exported step 0. `report.json` is the final completion record;
`progress.json` remains the last intermediate snapshot. See [validation](VALIDATION.md).

## View a policy

The following headless viewer command was tested. Remove `--headless` for the
interactive window; the launcher selects `mjpython` on macOS. A display is
required for the graphical path, which was not opened during these checks.

```sh
uv run --locked --extra sim cartpole-view-policy --actor policies/working.json --experiment evidence/policy-recovery-20261021/combined-next/protocol.json --seconds 1 --headless
```

## Submit a custom controller

After configuring a portal, set `CARTPOLE_PORTAL_URL` and `CARTPOLE_SUBMIT_TOKEN`
in your environment. The submission CLI help was checked; no remote submission
was made. Upload a trusted controller file with:

```sh
uv run --locked python examples/controller_starter/submit.py --artifact examples/controller_starter/physics_policy.py --source simulation --seconds 5 --key physics-simulation-001 --recording detailed --collect runs/physics-portal-001.zip
```

The worker must have the matching SDK and an explicitly built native library.
Both starter controller files select 100 Hz for simulation and 250 Hz for magnetic-step
inputs, with source-specific timing guards; synthetic offline tests cover both.
For a prepared physical worker, change `--source` to `magnetic-step` and use a new
key. Physical operation still needs the deployment's approval and safety workflow;
this starter does not validate real-time deadlines or physical transfer. The
simple damping `policy.py` uses the same source-aware cadence guards without a
native build requirement; it remains a damping example, not a balancing controller.
Training and portal operations do not grant hardware acceptance.

See [capabilities and limitations](SDK_CAPABILITIES.md) and the root hardware and
system-identification guides before configuring a physical deployment.
