"""V3 task constraint: a full angular excursion ends the attempt.

This is a history-dependent task constraint. The feedforward actor retains its
21 observed inputs, and does not receive accumulated angular excursion. It may
fail to learn swing-up; this constraint does not create controller competence.
"""
from .energy_swing import EnergySwingEnv, OBJECTIVE as ENERGY_OBJECTIVE

OBJECTIVE_ID='controlled-swing-v3'
OBJECTIVE=dict(ENERGY_OBJECTIVE,id=OBJECTIVE_ID,
    full_rotation_constraint='terminate when unwrapped max-minus-min angle reaches 2pi',
    constraint_penalty='fixed 3 * configured episode decisions, independent of violation time',
    observability='history-dependent task constraint; original 21 actor inputs unchanged',
    rotations='task termination and qualification; physical fault remains separate')


class ConstrainedSwingEnv(EnergySwingEnv):
    def reset(self,*,seed=None,options=None):
        self.constraint_violation=None;self.constraint_penalty=0.
        return super().reset(seed=seed,options=options)

    def step(self,action):
        observation,reward,terminated,truncated,info=super().step(action)
        if self.swing_metrics.report()['unwrapped_range_rotations']>=1. and self.constraint_violation is None:
            self.constraint_violation='full_rotation_excursion'
            self.constraint_penalty=3.*self.core.config.max_steps
            self.total_reward-=self.constraint_penalty
            reward-=self.constraint_penalty*self.reward_scale
            self._ended=True;terminated,truncated=True,False
            info['episode_result']=self.result()
        return observation,float(reward),terminated,truncated,info

    def result(self,*,hold_seconds=5.):
        result=super().result(hold_seconds=hold_seconds)
        result.update(objective_id=OBJECTIVE_ID,constraint_violation=self.constraint_violation,
                      constraint_penalty=self.constraint_penalty)
        if self.constraint_violation:
            result.update(success=False,hold_success=False,controlled_success=False)
        return result
