#pragma once
#include <math.h>
#include "cart_motion.h"
#include "cart_mechanics.h"
#include "spin_recovery.h"

// Shared by firmware and nonlinear closed-loop simulation. SI units.
namespace swing_control {
constexpr float kGravity = 9.81f;
constexpr float kPi = 3.14159265358979323846f;
inline float wrap(float x) { return atan2f(sinf(x), cosf(x)); }

struct Parameters {
  float leff = 0.124f;              // 125 mm arm: 20 low-angle cycles in three prior captures
  float pw = 6.0f, pz = 0.85f;
  float bal_pw = 8.0f;             // upright-only trial; auto retains pw
  float pc1 = -3.0f, pc2 = -4.0f;
  float vmax = cart_hardware::default_speed; // 750 RPM = 1.5 m/s on 60T
  // vmax is a hard command ceiling; these are independent policy limits.
  float vmax_s = 1.5f, vmax_b = 1.5f;
  float approach_v = 0.3f, catch_v = 0.3f;
  float approach_angle = 0.8f, catch_da = 4.0f;
  float amax_s = cart_hardware::default_acceleration; // 12,500 RPM/s on 60T
  float amax_b = cart_hardware::default_acceleration; // 25 m/s^2
  float jmax = 150.0f; // user-tuned default, m/s^3
  float amax_m = 0.5f, jmax_m = 10.0f;
  float manual_speed_max = .15f;
  float rail_fraction = .95f, rail_margin = .015f, rail_fault_margin = .010f;
  float rail_hysteresis = .005f, rail_recover_v = .15f, rail_recover_a = 1.5f, rail_recover_s = 8.f;
  float rehome_v = .08f, rehome_a = .5f, rehome_j = 10.f;
  float ke = 4.0f, kpx = 40.0f, kdx = 2.0f;
  float phase_soft = 2.0f;          // rad/s scale of smooth phase feedback
  float catch_a = 0.60f, catch_r = 3.0f, giveup = 0.80f;
  float spin_trip_rad_s=spin_recovery::trip_rad_s, spin_resume_rad_s=spin_recovery::resume_rad_s;
  float rail = 0.150f, bw = 30.0f, vman = 0.05f;
};
struct Gains { float k1, k2, k3, k4; };
inline Gains gains(const Parameters &p) {
  const float b1 = 2*p.pz*p.pw, b0 = p.pw*p.pw;
  const float c1 = -(p.pc1+p.pc2), c0 = p.pc1*p.pc2;
  Gains k;
  k.k3 = -b0*c0*p.leff/kGravity;
  k.k4 = -(b1*c0+b0*c1)*p.leff/kGravity;
  k.k1 = p.leff*k.k3-kGravity-p.leff*(b0+b1*c1+c0);
  k.k2 = p.leff*(k.k4-b1-c1);
  return k;
}
inline Gains uprightGains(const Parameters &p) {
  Parameters upright=p; upright.pw=p.bal_pw; return gains(upright);
}
inline float balance(const Gains &k, float theta, float omega, float x, float v) {
  return -(k.k1*theta+k.k2*omega+k.k3*x+k.k4*v);
}
inline float energy(float theta, float omega, float leff) {
  return 0.5f*leff/kGravity*omega*omega+cosf(theta);
}
inline bool validEstimatorBandwidth(float hz) {
  return hz>=0.1f && hz<=100.0f;
}
inline void resetEstimate(float measured, float &angle, float &rate) {
  angle=wrap(measured); rate=0;
}
// At 1 kHz, bw=30 Hz gives about 9 ms rate phase delay over 1-5 Hz.
// Keep wrapped innovations: differentiating raw counts would spike at rollover.
inline void estimate(float measured, float bandwidth, float dt, float &angle, float &rate) {
  const float w = 2*kPi*bandwidth;
  const float error = wrap(measured-angle);
  angle = wrap(angle+(rate+2*w*error)*dt);
  rate += w*w*error*dt;
}
struct PumpState {
  float elapsed = 0, quiet = 0;
  bool startup_kick = false;
  void reset() { elapsed = quiet = 0; startup_kick = false; }
};
// Changing an unused hard ceiling cannot change the requested motion.
inline float speedLimit(const Parameters &p, bool balancing) {
  return fminf(p.vmax, balancing ? p.vmax_b : p.vmax_s);
}
inline float recoverySpeedLimit(const Parameters &p) {
  return fminf(p.vmax, fmaxf(p.vmax_s,p.vmax_b));
}
// Begin preparing the actuator before the capture gate. Blend energy pumping
// into balance demand on the inbound arc; reduce the planned speed smoothly.
// This changes the target acceleration, never the current velocity/acceleration.
inline float approach(const Parameters &p, float theta, float omega, float x,
                      float v, float acceleration, float pump, float dt,
                      const Gains *feedback=nullptr) {
  const bool inbound=theta*omega<0 || fabsf(theta)<0.15f;
  const float span=fmaxf(0.05f,p.approach_angle-p.catch_a);
  const float weight=inbound?cart_motion::clamp((p.approach_angle-fabsf(theta))/span,0,1):0;
  const float request=(1-weight)*pump+weight*balance(feedback?*feedback:gains(p),theta,omega,x,v);
  const float speed=speedLimit(p,false);
  const float cap=speed*(1-weight)+fminf(speed,p.approach_v)*weight;
  const float upper=cart_motion::settledVelocityAccel(v,acceleration,cap,p.amax_s,p.jmax,dt);
  const float lower=cart_motion::settledVelocityAccel(v,acceleration,-cap,p.amax_s,p.jmax,dt);
  return cart_motion::clamp(request,lower,upper);
}
inline float swing(const Parameters &p, PumpState &s, float theta, float omega,
                   float x, float v, float dt, float acceleration=0,
                   const Gains *feedback=nullptr) {
  // Latch only at a centered, near-rest hanging start. The directly calibrated
  // upright can leave a modest hanging-angle error; do not wait for sensor noise.
  if(s.elapsed==0) s.startup_kick=fabsf(wrap(theta-kPi))<0.30f &&
      fabsf(omega)<0.5f && fabsf(x)<0.05f;
  s.elapsed += dt;
  // Continuous phase feedback replaces the noisy sign relay. Small sensor
  // motion now produces a small demand, not a full +/- energy-pump reversal.
  const float phase = tanhf(omega*cosf(theta)/p.phase_soft);
  float a = p.ke*(energy(theta, omega, p.leff)-1)*phase-p.kpx*x-p.kdx*v;
  if (fabsf(omega)<0.15f && fabsf(wrap(theta-kPi))<0.08f) s.quiet += dt;
  else s.quiet = 0;
  // One 120 ms, 1 m/s^2 request, still subject to the common jerk/rail governor.
  // Never rearm while pumping or on an edge-recovery handoff. Abort the kick if
  // the arm leaves the hanging window or is already moving appreciably.
  if(s.startup_kick && s.elapsed<=0.120f && fabsf(omega)<1.5f &&
     fabsf(wrap(theta-kPi))<0.40f && fabsf(x)<0.05f)
    a=fminf(1.0f,p.amax_s);
  return approach(p,theta,omega,x,v,acceleration,cart_motion::clamp(a,-p.amax_s,p.amax_s),dt,feedback);
}
inline bool canCapture(const Parameters &p, const Gains &k, float theta,
                       float omega, float x, float v, float acceleration=0) {
  const float demand=balance(k,theta,omega,x,v);
  if(fabsf(theta)>=p.catch_a || fabsf(omega)>=p.catch_r ||
     fabsf(x)>=fminf(0.12f,p.rail-2*p.rail_margin) ||
     fabsf(v)>=fminf(p.catch_v,speedLimit(p,true)) ||
     fabsf(demand)>=p.amax_b || fabsf(demand-acceleration)>p.catch_da)return false;
  if(fabsf(theta)>0.20f && theta*omega>0)return false;
  // Bounded 180 ms forecast using the same jerk/speed/rail governor. Reject a
  // capture whose correction immediately relatches rail braking or diverges.
  // This is a feasibility screen, not a guarantee of physical motor tracking.
  const float initial=kGravity/p.leff*theta*theta+omega*omega,dt=0.005f;
  float q=theta,w=omega,xx=x;
  cart_motion::State motion;motion.velocity=v;motion.acceleration=acceleration;
  for(int i=0;i<36;++i){
    const float request=balance(k,q,w,xx,motion.velocity);
    motion=cart_motion::advance(motion,xx,request,speedLimit(p,true),p.amax_b,
                               fmaxf(p.amax_s,p.amax_b),p.jmax,p.rail,dt,p.rail_margin);
    if(motion.brake_direction)return false;
    xx+=motion.velocity*dt;
    const float alpha=(kGravity*sinf(q)-motion.acceleration*cosf(q))/p.leff;
    q=wrap(q+w*dt+0.5f*alpha*dt*dt);w+=alpha*dt;
    if(fabsf(q)>=p.giveup || fabsf(xx)>p.rail-p.rail_margin-p.rail_hysteresis)return false;
  }
  return kGravity/p.leff*q*q+w*w<=initial*1.05f+0.01f;
}

// This observes pendulum response, NOT motor position. It can detect the gross
// "pulses but no pendulum motion" failure seen in the log, not all missed steps.
struct ResponseWatch {
  float elapsed = 0, initial_angle = 0, angle_change = 0, travel = 0;
  bool armed = false;
  void reset(float theta) {
    elapsed = angle_change = travel = 0; initial_angle = theta; armed = true;
  }
  bool fault(float theta, float command_velocity, float dt) {
    if (!armed) return false;
    elapsed += dt;
    angle_change = fmaxf(angle_change, fabsf(wrap(theta-initial_angle)));
    travel += fabsf(command_velocity)*dt;
    if (angle_change>0.03f) { armed = false; return false; }
    return elapsed>2.0f && travel>0.03f;
  }
};
} // namespace swing_control
