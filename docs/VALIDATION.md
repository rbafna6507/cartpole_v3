# Starter validation — 2026-10-27

Environment: macOS arm64, Python 3.12.11, MuJoCo 3.15.0, Gymnasium 1.4.0,
NumPy 2.5.4. Optional RL smoke: Torch 2.14.1, stable-baselines3 2.9.0.

- Fresh sim installation succeeded; curated suite: 423 passed, 284 subtests passed.
- Native controller built from included firmware headers with C++17.
- Physics adapter: synthetic 100 Hz simulation / 250 Hz magnetic-step acceptance,
  repeated/late sample rejection with latched faults, reset, missing-build failure.
- Damping starter: 500 simulation inputs and 500 replay inputs.
- Physics: 2,500 simulation and replay inputs. Preserved working and 1m actors:
  2,500 simulation inputs each. No null actions or environment faults.
- All 18 runtime hashes matched for each preserved actor; 175 source and 23
  firmware files were byte-identical to the source checkout.
- Short controlled-swing-v3 training completed 1,024 steps, saved/reloaded
  checkpoints at steps 0 and 1,024, and passed numeric emission parity for 40
  actions (maximum discrepancy 3.9853e-8 m/s²). The selector reported zero of
  four hanging successes, selected best development step 0, and left
  qualified_for_controlled_swing=false and deployment_candidate=false.
- Newly exported development actor loaded, simulated 500 steps and replayed 500
  inputs. Export is a pipeline smoke test, not an effective trained controller.

Raw smoke outputs remain ignored under runs/. Commands and rollout performance
are in [SDK_QUICKSTART.md](SDK_QUICKSTART.md). No physical hardware motion,
firmware flash, remote submission or graphical window was used.
