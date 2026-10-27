#pragma once
#include <stdint.h>
#include "cart_motion.h"

namespace spin_recovery {
constexpr float trip_rad_s=150.0f, resume_rad_s=10.0f;
constexpr uint32_t centering_timeout_us=10000000UL;
constexpr float center_tolerance=.003f, return_speed=.10f, return_acceleration=.5f;
// Thresholds apply to the magnitude of the existing filtered theta_dot estimate.
// No full-turn qualification, angle gate, or dwell time.
inline bool validLimits(float trip,float resume){return isfinite(trip) && isfinite(resume) && resume>0 && trip>resume;}
inline bool trigger(float omega,float trip=trip_rad_s){return fabsf(omega)>trip;}
enum Phase {NONE=0,BRAKING,CENTERING,WAITING};
struct State {
  Phase phase=NONE;
  uint32_t phase_start=0;
  void reset(){phase=NONE;phase_start=0;}
  void begin(uint32_t now){if(phase==NONE){phase=BRAKING;phase_start=now;}}
  bool timedOut(uint32_t now)const{return phase!=NONE && phase!=WAITING && uint32_t(now-phase_start)>centering_timeout_us;}
  bool update(float x,const cart_motion::State &m,float omega,bool sensor_valid,uint32_t now,float resume=resume_rad_s){
    if(phase==NONE)return false;
    const bool cart_quiet=fabsf(m.velocity)<=.005f && fabsf(m.acceleration)<=.1f;
    const bool centered=fabsf(x)<=center_tolerance && cart_quiet;
    if(phase==BRAKING && cart_quiet)phase=CENTERING;
    if(phase==CENTERING && centered)phase=WAITING;
    if(phase==WAITING && !centered){phase=CENTERING;phase_start=now;}
    if(phase==WAITING && sensor_valid && fabsf(omega)<resume){reset();return true;}
    return false;
  }
  float demand(float x,const cart_motion::State &m,float vmax,float amax,float jerk,float dt)const{
    const float target=phase==BRAKING?0:cart_motion::clamp(-3*x,-fminf(return_speed,vmax),fminf(return_speed,vmax));
    const float cap=phase==BRAKING?amax:fminf(return_acceleration,amax);
    return cart_motion::velocityAccel(m.velocity,target,cap,jerk,dt);
  }
};
}
