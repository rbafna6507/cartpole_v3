#pragma once
#include <math.h>

// SI units throughout. Kept independent of Arduino so the exact motion math
// can be exercised on the host without driving hardware.
namespace cart_motion {
inline float clamp(float x, float lo, float hi) {
  return fminf(hi, fmaxf(lo, x));
}
inline float slew(float current, float target, float delta) {
  return current + clamp(target - current, -delta, delta);
}

// Acceleration target for approaching a velocity without stepping acceleration
// to zero at arrival. Reserve one control tick in the jerk ramp-down distance.
inline float velocityAccel(float velocity, float target, float amax,
                           float jerk, float dt) {
  const float error = target - velocity;
  const float jd = jerk * dt;
  const float a = fminf(amax, sqrtf(jd*jd + 2*jerk*fabsf(error)) - jd);
  return error >= 0 ? a : -a;
}

// Account for acceleration already in progress when approaching a velocity.
// The remaining velocity change while acceleration slews to zero is a*|a|/(2*j).
// Reserve one tick so a discrete update starts the ramp-out before overshooting.
inline float settledVelocityAccel(float velocity, float acceleration, float target,
                                  float amax, float jerk, float dt) {
  const float error=target-velocity;
  const float remaining=acceleration*acceleration/(2*jerk)+fabsf(acceleration)*dt;
  if(acceleration*error>0 && fabsf(error)<=remaining) return 0;
  return velocityAccel(velocity,target,amax,jerk,dt);
}

// Maximum outward distance for a jerk-limited stop ending at v=0 AND a=0.
// Triangular acceleration when possible, otherwise ramp/hold/ramp. Unlike
// stopping only at the first velocity zero, this reserves the final jerk ramp.
inline float stoppingDistance(float velocity, float acceleration,
                              float amax, float jerk) {
  const float v=fmaxf(0,velocity),a=clamp(acceleration,-amax,amax);
  // Already braking too strongly to avoid reversing before acceleration is
  // zero: ramp out immediately and return distance to the first velocity zero.
  if(a<0 && v<a*a/(2*jerk)) {
    const float t=2*v/(-a+sqrtf(fmaxf(0,a*a-2*jerk*v)));
    return fmaxf(0,v*t+0.5f*a*t*t+jerk*t*t*t/6);
  }
  const float peak=fminf(amax,sqrtf(jerk*v+0.5f*a*a));
  const float t1=fmaxf(0,(a+peak)/jerk);
  const float v1=v+a*t1-0.5f*jerk*t1*t1;
  const float hold=peak>0?fmaxf(0,(v1-peak*peak/(2*jerk))/peak):0;
  const float d1=v*t1+0.5f*a*t1*t1-jerk*t1*t1*t1/6;
  const float d2=v1*hold-0.5f*peak*hold*hold;
  const float v2=v1-peak*hold,t3=peak/jerk;
  return fmaxf(0,d1+d2+v2*t3-0.5f*peak*t3*t3+jerk*t3*t3*t3/6);
}

struct State {
  float velocity = 0;
  float acceleration = 0;
  int brake_direction = 0;
};

inline State advance(State s, float x, float requested_accel,
                     float vmax, float drive_amax, float brake_amax,
                     float jerk, float rail_half, float dt, float rail_margin=0.015f) {
  // The caller validates a separate fault boundary outside this normal stop.
  const float boundary = rail_half - rail_margin;
  const float jd = jerk*dt;
  float target = clamp(requested_accel, -drive_amax, drive_amax);
  // Signed bounds also decelerate an inherited velocity above a lower mode cap.
  const float positive = settledVelocityAccel(s.velocity, s.acceleration, vmax, brake_amax, jerk, dt);
  const float negative = settledVelocityAccel(s.velocity, s.acceleration, -vmax, brake_amax, jerk, dt);
  target = clamp(target, negative, positive);
  const float next_a = slew(s.acceleration, target, jd);
  const float next_v = s.velocity + next_a*dt;

  if (s.brake_direction && fabsf(s.velocity)<=0.005f &&
      fabsf(s.acceleration)<=0.1f)
    s.brake_direction = 0;

  // Look ahead one tick before accepting an acceleration command. Test BOTH
  // rail ends; this also covers accelerating outward from rest near an end.
  if (!s.brake_direction) {
    for (int direction = -1; direction <= 1; direction += 2) {
      const float outward_v = next_v*direction;
      const float outward_a = next_a*direction;
      const float room = boundary - direction*x;
      // A late/inherited stop may already lie beyond the normal boundary.
      // Permit a gentle inward departure instead of relatching on residual
      // sub-5 mm/s outward velocity while acceleration is already inward.
      if(room<=0 && outward_v<=0.005f && outward_a<0) continue;
      const float travel = fmaxf(0, outward_v)*dt;
      if ((outward_v > 0 || outward_a > 0) &&
          travel + stoppingDistance(outward_v, outward_a, brake_amax, jerk) >= room) {
        s.brake_direction = direction;
        break;
      }
    }
  }
  // Hold the braking latch until both speed and acceleration are settled.
  if (s.brake_direction)
    target = settledVelocityAccel(s.velocity,s.acceleration,0,brake_amax,jerk,dt);
  // Account for velocity gained while acceleration ramps back to zero.
  // Without this anticipation, low-jerk reversals can overshoot vmax even
  // when the requested acceleration has already changed direction.
  const float candidate_a = slew(s.acceleration, target, jd);
  const float candidate_v = s.velocity + candidate_a*dt;
  if (candidate_a > 0 && candidate_v + candidate_a*candidate_a/(2*jerk) >= vmax)
    target = fminf(target, 0.0f);
  if (candidate_a < 0 && candidate_v - candidate_a*candidate_a/(2*jerk) <= -vmax)
    target = fmaxf(target, 0.0f);
  s.acceleration = slew(s.acceleration, target, jd);
  s.velocity += s.acceleration*dt;
  return s;
}
}  // namespace cart_motion
