# Saved hardware Python AUTO trial

`python-auto-consistent-lead-20261027.py` is the exact saved October 27 trial
submission. It uses native controller math, a fixed parameter snapshot and a
250 Hz magnetic/step observation contract. It does not run unchanged through the
100 Hz simulation adapter. See the portable `controller_starter/physics_policy.py`
for a source-aware simulation/replay starting point.

The saved trial improved capture but was not consistently reliable: one 12 s
trial captured at 3.884 s and held through the end; a later 25 s repeat captured
at 3.796 s, entered rail recovery at 7.756 s and did not recapture. These are
historical observations, not qualification. Calibration and host timing remain
relevant. This file includes a historical rail/parameter snapshot and disables
cyclic garbage collection on import; import it only in a dedicated experiment
process. It must not be substituted silently for a user's controller.

On the operator console, `python-auto-trial` uses the corresponding maintained
SDK pathway with current validated rig parameters and scoped runtime handling.
The frozen submission preserves the old trial for study. The reference source
for the portable example is the included firmware; native AUTO remains the
established physical controller.
