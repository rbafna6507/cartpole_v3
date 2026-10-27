# SDK capabilities and status

This starter retains the source-explicit SDK, current Teensy firmware source,
portable controller examples, two frozen trained actors, a pinned simulation
profile, and a curated offline regression suite. It does not carry raw rig runs,
private deployment configuration, credentials, or the complete research archive.

| Area | Included capability | Status and limits |
| --- | --- | --- |
| Data contract | Sessions, typed observations/actions, channel source/frame/age validation | Used by simulation, replay and rig interfaces |
| Simulation | MuJoCo plant, sampled encoder observations, delayed commands, motion governor, Gym interface | Tested offline; model error and physical transfer remain |
| Replay | Recording reader/writer and policy action traces | Open-loop evaluation; observations stay fixed |
| Physics controller | Native C++ binding to firmware headers, swing/capture/balance example | Built and tested locally; Python external timing experimental |
| Trained policy | working.json and 1m.json numeric actors, portable source adapter | DEVELOPMENT_ONLY; hardware_accepted=false |
| Physical interfaces | Serial/Ethernet transport, protocol, parameters, calibration and operator APIs | Require independently configured rig; no physical validation performed here |
| Vision and fusion | Source-explicit sensing and optional vision service source | Install vision extra as needed; camera/deployment assets are not configured |
| Training | Training/export modules and profile/model inputs | Requires rl extra; 1024-step checkpoint/export smoke passed; no qualified new controller |
| Portal | Client, server and worker implementation source | Deployment, credentials and hardware access not configured |
| Identification | Identification APIs and examples | Refer to the system_id package documentation for retained inputs and limitations |

## Native firmware versus host Python

The Teensy firmware owns the onboard high-rate executor and its safety state
machines. The Python SDK exposes experiments and host policy pathways; a normal
OS process does not inherit the firmware's real-time scheduling guarantees.
`physics_policy.py` calls the actual firmware equations through an explicitly
built native library, but uses source-aware policy cadence: 100 Hz simulation and 250 Hz magnetic-step.
Both input cadences have synthetic offline tests; physical timing is unqualified.
Numerical reuse is not proof of end-to-end hardware equivalence.

The preserved Oct-27 magnetic-only 250 Hz experiment, if referenced in the
experimental examples, is a separate interface/cadence contract. It has not been
validated by the 100 Hz starter simulations and is not a drop-in replacement.

## Artifact integrity and provenance

Both policy files and all tracked SDK source were copied without modifications.
All 18 runtime file hashes in each actor matched the copied source. The actor
model hash is `37a9b9f00245ffb4784e11b4355206b0e49db784740f0e660c517cefe188484f`;
the associated model is retained in `training/profiles/rl-encoder-20261021/`.
The packaged profile is `rig-calibrated-20261021-v1`, digest
`b3f3bea424c30440acefe9bb29f54bcb8de7b0249152d804815124e68b8e8e0d`.

The single source protocol under
`evidence/policy-recovery-20261021/combined-next/protocol.json` is retained because
the profile pins its bytes. Historical absolute source paths and experiment
references inside hash-bound metadata are provenance, not promises that all
original research inputs are present. They were not rewritten, because doing so
would change the frozen identities. New output belongs in ignored `runs/`.

The portable test selection excludes archive-dependent whole-sketch oracles,
historical raw-run numeric benchmarks, and historical camera fixtures. Included
native tests independently compile the current controller headers; the full v2
research qualification history is not represented by this starter suite.

## Verified here

Fresh `uv sync --extra sim`; 423 pytest cases plus 284 subtests; native C++ binding
build; 5-second starter simulation and 500-observation replay; 25-second physics,
working and 1m actor simulations; frozen actor runtime hashes; 1024-step training/export parity and exported-actor replay. Commands and
single-seed outcomes are in [SDK_QUICKSTART.md](SDK_QUICKSTART.md).
No firmware flash, rig connection, physical motion or hardware
acceptance was performed for these checks.

The simple damping starter also accepts the same explicit 100 Hz simulation /
250 Hz magnetic-step mappings. Offline tests cover repeated-sample rejection,
late fresh-input rejection, fault latching and per-run reset for both sources.
