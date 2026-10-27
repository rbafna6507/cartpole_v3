#pragma once
#include <stdint.h>
#include "cart_motion.h"

// Return to the existing counted midpoint. Never establishes a new reference.
// The first phase accepts inherited auto velocity/acceleration unchanged and
// brakes through the original governor. Only the settled centering phase uses
// the separately configured centering speed/acceleration/jerk bounds.
namespace saved_center_return {
constexpr uint32_t deadline_ticks=15000;
constexpr float brake_speed=1.5f,brake_acceleration=25.f,brake_jerk=150.f;
constexpr float center_speed=.08f,center_acceleration=.5f,center_jerk=10.f;
// The old fixed centering limits could absorb the small residual state after
// braking. User-selected limits can be much weaker. Preview the first center
// tick and its complete jerk-limited stop in BOTH directions before handing off.
inline bool canCenter(const cart_motion::State& motion,float x,float rail,float dt,
                      float speed,float acceleration,float jerk,float margin){
 if(fabsf(motion.velocity)>.002f || fabsf(motion.acceleration)>.02f ||
    fabsf(motion.velocity)>speed || fabsf(motion.acceleration)>acceleration)return false;
 const float target=cart_motion::clamp(-3*x,-speed,speed);
 const float demand=cart_motion::settledVelocityAccel(motion.velocity,motion.acceleration,target,acceleration,jerk,dt);
 const auto next=cart_motion::advance(motion,x,demand,speed,acceleration,acceleration,jerk,rail,dt,margin);
 if(!isfinite(next.velocity)||!isfinite(next.acceleration))return false;
 for(int direction=-1;direction<=1;direction+=2){
  const float outward_v=direction*next.velocity,outward_a=direction*next.acceleration;
  const float positive_a=fmaxf(0,outward_a);
  // Reserve the remaining velocity gained while ramping acceleration to zero.
  if(outward_v+positive_a*positive_a/(2*jerk)+positive_a*dt>speed)return false;
  const float room=fmaxf(0,rail-margin-direction*(x+next.velocity*dt));
  const float distance=cart_motion::stoppingDistance(outward_v,outward_a,acceleration,jerk);
  if(!isfinite(distance)||distance>room)return false;
 }
 return true;
}
struct State {
 enum Phase:uint8_t {OFF,BRAKING,CENTERING};
 Phase phase=OFF;float demand=0;
 void reset(){phase=OFF;demand=0;}
 void begin(){phase=BRAKING;demand=0;}
 bool settledAtCenter(int32_t error_steps,const cart_motion::State& motion)const{
  return error_steps==0 && fabsf(motion.velocity)<=.0005f && fabsf(motion.acceleration)<=.01f;
 }
 cart_motion::State advance(cart_motion::State motion,float x,float rail,float dt,float configured_speed=brake_speed,float configured_jerk=brake_jerk,
                            float configured_acceleration=brake_acceleration,
                            float return_speed=center_speed,float return_acceleration=center_acceleration,
                            float return_jerk=center_jerk,float rail_margin=.015f){
  if(phase==BRAKING && canCenter(motion,x,rail,dt,fminf(return_speed,configured_speed),return_acceleration,return_jerk,rail_margin))phase=CENTERING;
  const bool braking=phase==BRAKING;
  const float acceleration=braking?configured_acceleration:return_acceleration;
  // Auto may reach states whose stopping envelope requires its higher jerk.
  // Retain its configured braking capability until v/a settle.
  const float jerk=braking?configured_jerk:return_jerk;
  const float speed=braking?configured_speed:fminf(return_speed,configured_speed);
  const float target=braking?0:cart_motion::clamp(-3*x,-speed,speed);
  demand=cart_motion::settledVelocityAccel(motion.velocity,motion.acceleration,target,acceleration,jerk,dt);
  return cart_motion::advance(motion,x,demand,speed,acceleration,acceleration,jerk,rail,dt,rail_margin);
 }
};
}
