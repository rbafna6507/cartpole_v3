"""V2: penalize surplus swing energy, allowing speed required at the bottom."""
import math
from .ppo import TrainingEnv
from .controlled_swing import SwingMetrics

OBJECTIVE_ID='controlled-swing-v2'
OBJECTIVE=dict(id=OBJECTIVE_ID,rail_zone_m=.22,rail_weight=2.,energy_weight=1.,
    normalized_energy='cos(theta) + effective_length_m * omega^2 / (2 * 9.81)',
    free_energy_upper=1.15,horizon='time-limit truncation, no terminal dwell bonus',
    faults='legacy terminal penalty retained',
    qualification='final five-second hold, no faults, unwrapped angle range strictly less than one full turn',
    rotations='diagnostic and qualification only; no spin termination or history reward')


def energy_penalty(*,theta,omega,effective_length,gravity=9.81):
    energy=math.cos(theta)+effective_length*omega**2/(2*gravity)
    return max(energy-1.15,0.)**2


class EnergySwingEnv(TrainingEnv):
    def reset(self,*,seed=None,options=None):
        observation,info=super().reset(seed=seed,options=options)
        self.swing_metrics=SwingMetrics(self.core.ground_truth.theta,min(.22,.75*self.core.config.rail_half_m))
        return observation,info

    def step(self,action):
        before=self.core.ground_truth.time_ns
        observation,reward,terminated,truncated,info=super().step(action)
        truth=self.core.ground_truth;dt=(truth.time_ns-before)/1e9
        penalty=0.
        if dt>0:
            self.swing_metrics.update(truth.theta,truth.x,dt)
            zone=self.swing_metrics.rail_zone
            penalty=2.*(max(abs(truth.x)-zone,0.)/(self.core.config.rail_half_m-zone))**2
            penalty+=energy_penalty(theta=truth.theta,omega=truth.omega,effective_length=self.core.config.effective_length_m)
        if (terminated or truncated) and self.fault is None and self.steps==self.core.config.max_steps:
            penalty+=self.dwell*100
            terminated,truncated=False,True
        self.total_reward-=penalty;reward-=penalty*self.reward_scale
        if self._ended:info['episode_result']=self.result()
        return observation,float(reward),terminated,truncated,info

    def result(self,*,hold_seconds=5.):
        result=dict(super().result(hold_seconds=hold_seconds),objective_id=OBJECTIVE_ID,**self.swing_metrics.report())
        result['controlled_success']=result['hold_success'] and result['unwrapped_range_rotations']<1.
        return result
