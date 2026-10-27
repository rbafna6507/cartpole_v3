# One controller, three execution paths

Run these commands from the repository root. `policy.py` is a small cart-damping
integration example, **not a swing-up or balancing controller**. Edit its `act`
method while retaining the explicit observation adapter and per-run reset.
Local simulation/replay executes trusted local Python. Uploaded code uses the
portal worker’s separate sandbox.

```sh
uv run --locked --extra sim python examples/controller_starter/run.py simulate \
  --seconds 5 --output runs/starter-sim-001
uv run --locked python examples/controller_starter/run.py replay \
  --recording runs/starter-sim-001/observations.jsonl \
  --output runs/starter-replay-001.jsonl
```

For an intentionally configured portal server, set `CARTPOLE_PORTAL_URL` and
`CARTPOLE_SUBMIT_TOKEN` in your environment, then upload the same file:

```sh
uv run --locked python examples/controller_starter/submit.py \
  --artifact examples/controller_starter/policy.py --source simulation \
  --seconds 5 --key starter-sim-001 --recording detailed \
  --collect runs/starter-portal-001.zip
```

An updated API and configured simulation worker are required for detailed
collection. Physical runs use `--source magnetic-step`, operator approval and a
healthy prepared rig. A reservation does not bypass those requirements.

`train.py` trains and exports a numeric actor that `run.py --policy actor.json`
and `submit.py --artifact actor.json` can consume. Use `--help` on each script
for its arguments. Full installation, training/export, dataset/replay commands,
source semantics and limitations: [developer guide](../../docs/SDK_QUICKSTART.md).

For the firmware-equation swing/capture/balance example, use `physics_policy.py`
after the explicit native build in the SDK quickstart.
