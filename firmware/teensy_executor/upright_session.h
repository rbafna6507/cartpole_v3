#pragma once
#include "spin_recovery.h"

// Manual upright trials never enter swing-up or restart after a failed attempt.
namespace upright_session {
constexpr float pi=3.14159265358979323846f;
constexpr float fall_angle=50*pi/180, start_angle=10*pi/180;
inline bool canStart(float angle,float estimated,float rate,float x,float v,float a,bool healthy){
  return healthy && isfinite(angle) && isfinite(estimated) && isfinite(rate) &&
    fabsf(angle)<=start_angle && fabsf(estimated)<=start_angle &&
    fabsf(rate)<=1 && fabsf(x)<=.03f && fabsf(v)<=.005f && fabsf(a)<=.1f;
}
inline bool fallen(float angle){return fabsf(atan2f(sinf(angle),cosf(angle)))>fall_angle;}
struct Return {
  spin_recovery::State center;
  void reset(){center.reset();}
  void begin(uint32_t now){center.begin(now);}
  bool update(float x,const cart_motion::State &motion,uint32_t now){
    // Suppress automatic swing-up release: this session stops when centered.
    center.update(x,motion,0,false,now);
    return center.phase==spin_recovery::WAITING;
  }
  bool timedOut(uint32_t now)const{return center.timedOut(now);}
  bool braking()const{return center.phase==spin_recovery::BRAKING;}
  float demand(float x,const cart_motion::State &m,float vmax,float amax,float jerk,float dt)const{
    return center.demand(x,m,vmax,amax,jerk,dt);
  }
};
}
