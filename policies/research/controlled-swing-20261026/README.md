# Controlled swing research weights

Training completed, but controlled swing-up did not succeed. No checkpoint is qualified for hardware. Step 25,600 is the best actually trained development comparison; step 1,024,000 is the final research state. The original step-0 actor outperformed both under the selection rule.

`actor.original.json` preserves the exact original numeric export. `actor.json` preserves every weight and changes only the runtime pin for the reviewed scoped console rail-tolerance change, plus its content digest. The strict loader remains unchanged. No tolerance override is enabled here. See receipts and `runtime-migration.patch`. These are numeric JSON artifacts, not executable checkpoints.

Parity checks compare exact checkpoint PyTorch weights against migrated numeric inference on deterministic valid observations spanning cart position, velocity, full-angle orientations, angular rates, commanded acceleration, brake direction and sample ages. This establishes sampled numerical agreement only. No rig motion, services, active policy aliases, or firmware were changed.
