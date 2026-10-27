# Training status

The October 26 controlled-swing-v3 campaign completed **1,024,000 decisions** and preserved **41 checkpoints**, but did **not** learn a qualified controlled swing-up. All 164 recorded hanging evaluations failed controlled success. These are four repeated development cases across checkpoints, not 164 independent seeds. Reserved confirmation seeds were not evaluated.

The original step-0 baseline won the selection rule. Among trained checkpoints, step **25,600** ranked highest: hanging success 0/4, nominal upright holds 2/2, randomized upright holds 1/2. The final checkpoint has hanging success 0/4, nominal upright holds 2/2, and randomized upright holds 0/2. Zero physical faults does not imply success: rotation constraint violations are recorded separately.

Research numeric exports are in [policies/research/controlled-swing-20261026](../policies/research/controlled-swing-20261026/README.md). Both exports preserve exact original actors and all original weights. Their migrated versions pin the reviewed console rail-tolerance context change in the current runtime, retaining the strict provenance loader and its default 1 mm tolerance. No tolerance override is enabled by the export.

All 41 source checkpoint/actor file hashes, 82 canonical actor/manifest digests, and 41 ZIP CRC checks passed. For each staged actor, every weight matched its original PyTorch checkpoint exactly. Independent deterministic inference comparisons covered 1,296 encoded observations per actor spanning rail positions, velocities, angles, angular rates, accelerations, braking states, and ages. Maximum errors were 3.26e-6 m/s² (step 25,600) and 3.21e-6 m/s² (final), below the 1e-5 tolerance. See per-actor receipts for exact measurements.

These artifacts are **research only**, with no hardware acceptance or controlled-swing qualification. Sampled inference parity does not establish closed-loop performance, sensor-pipeline correctness, transport behavior, Jetson deadlines, or physical safety. No active policy alias, firmware, or service is changed by staging them. Future development should investigate exploration/reward and the history-dependent rotation constraint before assuming more unchanged training will help.

## Staged on the Jetson

The inactive research directory is `~/cartpole-v2-live/candidates/controlled-swing-20261026`.
It contains `step-000025600/actor.json` and `step-001024000/actor.json`, alongside
original exports and migration receipts. SHA256 checks matched all copied files;
both migrated actors then passed strict offline loading against the active Jetson
SDK at 100 Hz. No serial port was opened and neither actor was physically run.
The operator's active release and `policies/working.json` are unchanged.

For local research, use the same starter simulation command with
`--policy policies/research/controlled-swing-20261026/step-000025600/actor.json`.
These checkpoints should not be presented as successful controlled-swing demos.
