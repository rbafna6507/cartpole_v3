#pragma once
#include "cart_motion.h"

// Latched edge recovery; preserves the existing position origin.
namespace rail_recovery {
constexpr float edge_margin = 0.015f;       // ±135 mm on a 300 mm rail
constexpr float inside_hysteresis = 0.005f; // re-enter by 5 mm before releasing
constexpr float return_speed = 0.15f;
constexpr float return_acceleration = 1.5f;
enum Phase { NONE=0, BRAKING, RETURNING };
struct State {
  Phase phase = NONE;
  float elapsed = 0, exit_boundary = 0;
  int side = 0;
  void reset() { phase=NONE; elapsed=exit_boundary=0; side=0; }
  void begin(float x,float rail_half,int brake_direction=0,float margin=edge_margin,float hysteresis=inside_hysteresis) {
    if(phase!=NONE)return;
    phase=BRAKING;elapsed=0;
    side=brake_direction?brake_direction:(x>=0?1:-1);
    exit_boundary=rail_half-margin-hysteresis;
  }
  bool atEdge(float x,float rail_half,float margin=edge_margin) const { return fabsf(x)>=rail_half-margin; }
  bool timedOut(float seconds=8.f) const { return elapsed>seconds; }
  bool update(float x,const cart_motion::State &motion,float dt,float speed=return_speed,float acceleration=return_acceleration) {
    if(phase==NONE)return false;
    elapsed+=dt;
    // Finish the stop, including ramping acceleration out, before returning.
    const bool stopped=fabsf(motion.velocity)<=0.005f &&
        fabsf(motion.acceleration)<=0.1f;
    if(phase==BRAKING && stopped)phase=RETURNING;
    const bool inward=motion.velocity*side < -0.005f ||
        (motion.velocity*side<=0.005f && motion.acceleration*side<=0);
    const bool settled=fabsf(motion.velocity)<=speed+0.005f &&
        fabsf(motion.acceleration)<=acceleration;
    if(phase==RETURNING && inward && settled && motion.brake_direction==0 &&
        fabsf(x)<=exit_boundary){reset();return true;}
    return false;
  }
  float demand(float,const cart_motion::State &motion,
               float vmax,float amax,float jerk,float dt,float speed=return_speed,float acceleration=return_acceleration) const {
    const float target=phase==RETURNING ? -side*fminf(speed,vmax) : 0;
    const float cap=phase==RETURNING ? fminf(acceleration,amax) : amax;
    return cart_motion::settledVelocityAccel(motion.velocity,motion.acceleration,target,cap,jerk,dt);
  }
};
}
