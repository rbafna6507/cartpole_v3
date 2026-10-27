"""Portal-compatible bridge over the same named RigCartpoleEnv as local work."""
from dataclasses import asdict, replace
import math
from cartpole_sdk.contracts import Session, ValidationError
from cartpole_sdk.policies.rig_observation import select_rig_observation
from .profiles import PROFILE_ID, load_profile, make_profile_env


class ProfileSimulationBridge:
    kind = 'simulation'

    def __init__(self, *, profile_id=PROFILE_ID, seed=0):
        self.profile_id, self.seed = profile_id, seed

    def prepare(self, duration, identifier):
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 60:
            raise ValidationError('duration must be finite in (0,60]')
        self.profile = load_profile(self.profile_id)
        self.env = make_profile_env(self.profile_id, seconds=math.ceil(duration * 100) / 100)
        try:
            self.env.reset(seed=self.seed)
            self.session = Session(identifier, self.env.session.clock_id, self.env.session.topology)
            self.channels = self.env.channels
            self.config = asdict(self.env.config)
            self.steps, self.max_steps = 0, math.ceil(duration * 100)
            self._update()
            return self.selected()
        except BaseException:
            self.env.close()
            raise

    def _update(self):
        self.current = replace(self.env.sdk_observation, session_id=self.session.session_id)

    def selected(self):
        select_rig_observation(self.current, self.channels, contract=self.env.observation_contract)
        return self.current

    def arm(self, duration):
        pass

    def advance(self, value, sequence):
        self.env.step(None if value is None else [value])
        self.steps += 1
        self._update()
        if self.env.terminated:
            raise ValidationError('simulation executor fault: ' + str(self.env.fault_reason))
        return self.selected()

    def complete(self):
        return self.steps >= self.max_steps or self.env.truncated

    def stop(self):
        return True

    def close(self):
        if hasattr(self, 'env'):
            self.env.close()

    def diagnostics(self):
        return dict(profile_id=self.profile_id, profile_sha256=self.profile['content_sha256'],
            sim_config=self.config, observation_config=asdict(self.env.observation_config),
            simulation_time_ns=self.current.available_ns, simulation_fault=self.env.fault_reason,
            action_semantics=self.env.action_semantics,
            executor=asdict(self.env.backend.diagnostics), ground_truth=asdict(self.env.ground_truth),
            physical_acceptance=False)
