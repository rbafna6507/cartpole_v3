# Cartpole: build a controller, test it, run it on a real pendulum

This repository gives you the Python SDK, a governed MuJoCo/Gym simulation,
example controllers, learned policies and measured characterization data for a
remote cart-and-pendulum experiment. Start locally, inspect your results, then
submit the same controller file or numeric policy to the
[experiment portal](https://cartpole-portal.vercel.app/#experiments).

**Start here:** [30-minute SDK quick start](docs/SDK_QUICKSTART.md).
No Jetson, serial device or portal credential is needed for local simulation.

| What you want | Where to go |
| --- | --- |
| Install, simulate, replay, submit | [SDK quick start](docs/SDK_QUICKSTART.md) |
| Find runnable commands and options | [Command reference](docs/COMMANDS.md) |
| Understand the controller, policy and simulation | [How it works](docs/HOW_IT_WORKS.md) |
| Understand observations, sources and primitives | [SDK capabilities](docs/SDK_CAPABILITIES.md) |
| Upload code or a policy; collect run data | [Submission and data](docs/SUBMISSION_AND_DATA.md) |
| Understand measured dynamics and model limitations | [System identification](system_id/README.md) |
| Inspect hardware specifications and assumptions | [hardware.json](hardware.json) |
| Check the new controlled-swing training result | [Training status](docs/TRAINING_STATUS.md) |

## What is ready?

- **`policies/working.json`**: the existing learned actor that has swung up and
  balanced on this physical rig. Its formal acceptance flag remains false; a
  successful demonstration is not a guarantee across hardware or calibrations.
- **`policies/1m.json`**: a separate earlier million-step actor, not the new
  controlled-swing experiment. Do not assume more training means better behavior.
- **Native AUTO**: the established physics-based controller running on the
  microcontroller. The SDK also exposes its controller math for host-side work.
- **Python AUTO trial**: swing-up improved in recent trials, but sustained balance
  and repeatability are still under investigation. It is not a qualified default.
- **Simulation**: useful for learning the API, exercising failure handling and
  comparing designs. It does not certify that a controller will work on the rig.

See the quick start for tested examples and commands. The beginner damping
example illustrates the contract; it is intentionally not a swing-up controller.

## Repository map

`src/` contains the SDK and optional vision service; `examples/` contains runnable
entry points; `firmware/` contains the executor and controller reference;
`policies/` contains numeric actors; `system_id/` contains curated measurements;
`docs/` explains the public developer workflow. Historical hash-bound profile
inputs are retained only where needed to reproduce the supplied policies.

This is a clean developer snapshot of the working v2 system. Operational service
credentials, private settings, raw backup archives and local environments are
excluded. Native binaries are built for your own machine, not copied from a Jetson.

## Licensing

No project license is granted at present. The repository is public for inspection;
public visibility alone does not grant permission to reuse or redistribute its
code. Third-party dependencies retain their own licenses.
