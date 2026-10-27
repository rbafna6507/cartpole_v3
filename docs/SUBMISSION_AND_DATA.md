# Submit an experiment and learn from its data

Open the [Experiments page](https://cartpole-portal.vercel.app/#experiments).
The video, submission form and overlaid vision/encoder angles share the page.
Use a submission credential supplied by the operator. The frontend is separate
from the Jetson API/worker; a Vercel page does not execute your controller.
Ask the operator for the API origin as well as your token when using the Python
client. Do not use the Vercel static-site origin as a presumed worker API.

## Supported artifacts

| Artifact | Contract |
| --- | --- |
| One `.py` file | UTF-8, at most 64 KiB; exports `create_policy()` |
| `.zip` project | Root `policy.py`; at most 8 MiB uploaded, 32 MiB expanded, 256 entries |
| Numeric actor `.json` | SDK actor schema, at most 1,000,000 bytes; raw/filtered encoder source |

ZIPs cannot contain links, traversal paths or encrypted entries. ZIP support does
not mean arbitrary dependencies can be installed. The execution environment is
fixed: no package installation, network, subprocesses, arbitrary threads, devices
or writable filesystem. Keep policy state in memory. Export neural weights to the
numeric actor format rather than submitting a pickle or training checkpoint.

A controller receives observations and returns acceleration demands. It must
explicitly support its selected sensor source; a simulation channel is not a
physical encoder channel with a different label. The supplied starter demonstrates
source adapters. Physical jobs require operator approval and a healthy, homed rig.
A failed job is not automatically retried on hardware.

## Submit using Python or an agent

From the repository root, with a fresh output filename:

```sh
export CARTPOLE_PORTAL_URL='https://YOUR_OPERATOR_API_ORIGIN'
# Set CARTPOLE_SUBMIT_TOKEN privately in your shell; do not commit it.
uv run python examples/controller_starter/submit.py \
  --artifact examples/controller_starter/policy.py \
  --source simulation --seconds 5 --recording detailed \
  --key my-first-simulation-v1 --collect runs/my-first-result.zip --timeout 180
```

Create `runs/` first if needed. Use `--source magnetic-step` for a physical
submission, then wait for operator approval. A live simulation worker must be
available for remote simulation; local simulation works independently. Use a new
idempotency key when changing any run inputs. Retry the *same* request with the
same key after an ambiguous response instead of accidentally requesting two runs.

The SDK's `PortalClient` exposes `submit`, `get`, `jobs`, `cancel`, `wait`,
`download`, `collect`, and reservation methods. `collect` waits for completed
recording publication and verifies the downloaded bundle's identity and hashes.
Reservation APIs exist but reservations are currently hidden in the UI. Approval,
availability and the worker's configured limits remain authoritative.

## What is in a run bundle?

The ZIP contains `manifest.json` and `telemetry.jsonl`. The manifest identifies the
job/artifact, recording, outcome and diagnostics. JSONL records preserve observations,
actions and lifecycle/safety diagnostics; optional sensing envelopes are captured
separately from the control path and merged into the recording. Inspect the actual
record types and manifest instead of assuming every run contains every sensor.
Failed runs can be especially useful; their data should be analyzed alongside
successful runs.

Detailed recording requests bounded sensor history, not an unlimited lossless
capture. The current recorder polls sensing at up to 30 Hz, with up to 128 samples
per envelope in detailed mode. Original sensor timestamps, cursor/gap metadata and
recording drop counts matter. A stream may be unavailable; camera/vision data is
not guaranteed, and the bundle does not promise raw video or raw images. Hardware
INFO snapshots may include encoder register counts; these are not a full-rate raw
I²C trace. Cart steps remain an estimate of position, not an independent linear
encoder measurement.

Use these data to compare demanded and achieved motion, sensor age, filtering,
angle estimates, capture time, rail usage and failure reasons. They support system
identification and offline evaluation. Training a policy directly from a few runs
is not automatically reliable: retain source/frame identity, account for missing
samples and actuator limits, separate train/validation runs, and avoid treating
recorded replay as new physical feedback.

## Read-only collection on an operator-accessible network

The camera/sensing service exposes a developer stream catalog and bounded NDJSON
history. This is a separate service, not a second owner of the motor serial port.
With an operator-provided reachable sensing-service origin:

```sh
uv run cartpole-collect --url http://127.0.0.1:3090 \
  --duration 30 --max-hz 30 --limit 128 --output runs/sensing.jsonl
uv run cartpole-collect --help
```

The localhost example means the machine running that sensing service (or an
explicit tunnel). It is not the public Vercel API. `--streams` selects catalog IDs;
`--cursors` resumes from recorded cursor metadata. Output files are created anew.
The CLI has duration, record and byte bounds and does not command the rig.

## Execution and safety architecture

The API stores queue metadata in SQLite and immutable artifacts/recordings on the
server filesystem. The worker pulls jobs, downloads and verifies an artifact,
prepares a Linux Bubblewrap/seccomp sandbox, and waits for policy reset before
starting motion authority. Preparation and physical execution have separate
budgets. The sandbox cannot access host credentials, network or the serial device.
The trusted broker and firmware independently enforce limits and stop on faults,
stale commands or expired authority. Kernel/process isolation and firmware guards
reduce risk; this is not a claim that arbitrary code can never exploit the host.

The supplied files are developer tooling, not a production service installer.
Public multi-user operation still needs the operator's authentication, queue,
sandbox commissioning, physical preparation and fault-recovery procedures.
