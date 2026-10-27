"""Fast, stateless capture feasibility screen for the SDK control loop.

This follows ``swing_control::canCapture`` and ``cart_motion::advance`` over
36 fixed 5 ms steps, using ordinary Python float64 arithmetic. The live motion
governor remains the binary32 implementation in :mod:`.motion`; this forecast
must never supply actuator state. It has no compilation or third-party runtime
dependencies.

This is not native bit parity or a guarantee of physical capture. Float32
rounding and libm differences can change the tick on which a discontinuous
settled-velocity guard switches. Small guard margins below reject numerically
ambiguous decisions, but do not constitute a formal error bound for all gains
and parameters. Conservative rejections are more common with tight speed caps.
Validate the intended controller envelope against native and
physical behavior before relying on this as a capture policy.
"""
from math import atan2, cos, isfinite, sin, sqrt

from .parameters import ControllerParameters, Gains


# An absolute floor plus a relative allowance, in the units of each guard.
# This intentionally admits fewer near-boundary captures than the native gate.
_GUARD_EPS = 1e-6
_DT = .005


def _stopping_distance(velocity, acceleration, amax, jerk):
    """Float64 counterpart of cart_motion::stoppingDistance (SI units)."""
    v = max(0., velocity)
    a = min(amax, max(-amax, acceleration))
    if a < 0 and v < a*a/(2*jerk):
        t = 2*v/(-a+sqrt(max(0., a*a-2*jerk*v)))
        return max(0., v*t+.5*a*t*t+jerk*t*t*t/6)
    peak = min(amax, sqrt(jerk*v+.5*a*a))
    t1 = max(0., (a+peak)/jerk)
    v1 = v+a*t1-.5*jerk*t1*t1
    hold = max(0., (v1-peak*peak/(2*jerk))/peak) if peak > 0 else 0.
    d1 = v*t1+.5*a*t1*t1-jerk*t1*t1*t1/6
    d2 = v1*hold-.5*peak*hold*hold
    v2, t3 = v1-peak*hold, peak/jerk
    return max(0., d1+d2+v2*t3-.5*peak*t3*t3+jerk*t3*t3*t3/6)


def can_capture_fast(
    p: ControllerParameters, k: Gains, theta: float, omega: float,
    x: float, v: float, acceleration: float = 0.,
) -> bool:
    """Return whether a 180 ms governed balance forecast is feasible.

    Inputs use SI units and are read without mutation. Nonfinite state, gains,
    or consumed parameters fail closed. Parameters should otherwise already
    satisfy ``ControllerParameters`` validation. The forecast cadence is always
    5 ms, independent of the caller's 100 Hz or 250 Hz control cadence.
    """
    # Strip _Float32 scalar overloads once; retaining them defeats this fast path.
    q, w, xx, velocity, a = map(float, (theta, omega, x, v, acceleration))
    k1, k2, k3, k4 = map(float, (k.k1, k.k2, k.k3, k.k4))
    (leff, vmax, vmax_b, catch_a, catch_r, catch_v, catch_da,
     drive, swing_drive, jerk, rail, margin, hysteresis, giveup) = map(float, (
        p.leff, p.vmax, p.vmax_b, p.catch_a, p.catch_r, p.catch_v, p.catch_da,
        p.amax_b, p.amax_s, p.jmax, p.rail, p.rail_margin,
        p.rail_hysteresis, p.giveup,
    ))
    if not all(map(isfinite, (
        q, w, xx, velocity, a, k1, k2, k3, k4, leff, vmax, vmax_b,
        catch_a, catch_r, catch_v, catch_da, drive, swing_drive, jerk,
        rail, margin, hysteresis, giveup,
    ))):
        return False
    if min(leff, vmax, vmax_b, drive, swing_drive, jerk) <= 0:
        return False

    speed = min(vmax, vmax_b)
    capture_speed = min(catch_v, speed)
    capture_position = min(.12, rail-2*margin)
    demand = -(k1*q+k2*w+k3*xx+k4*velocity)
    eps = _GUARD_EPS
    if (not isfinite(demand)
            or abs(q) >= catch_a-eps*(1+catch_a)
            or abs(w) >= catch_r-eps*(1+catch_r)
            or abs(xx) >= capture_position-eps*(1+abs(capture_position))
            or abs(velocity) >= capture_speed-eps*(1+capture_speed)
            or abs(demand) >= drive-eps*(1+drive)
            or abs(demand-a) > catch_da-eps*(1+catch_da)):
        return False
    if abs(q) > .20-eps and q*w > 0:
        return False

    gravity_over_length = 9.81/leff
    initial = gravity_over_length*q*q+w*w
    if not isfinite(initial):
        return False
    brake = max(drive, swing_drive)
    boundary = rail-margin
    position_limit = boundary-hysteresis
    dt = _DT
    jd = jerk*dt
    two_jerk = 2*jerk
    speed_epsilon = eps*(1+speed)

    for _ in range(36):
        request = -(k1*q+k2*w+k3*xx+k4*velocity)
        target = min(drive, max(-drive, request))
        remaining = a*a/two_jerk+abs(a)*dt

        # Signed settledVelocityAccel bounds, including inherited overspeed.
        bounds = []
        for cap in (speed, -speed):
            error = cap-velocity
            if a*error > 0:
                if abs(abs(error)-remaining) <= speed_epsilon:
                    return False
                settled = abs(error) <= remaining
            else:
                settled = False
            if settled:
                bound = 0.
            else:
                magnitude = min(brake, sqrt(jd*jd+two_jerk*abs(error))-jd)
                bound = magnitude if error >= 0 else -magnitude
            bounds.append(bound)
        target = min(bounds[0], max(bounds[1], target))
        next_a = a+min(jd, max(-jd, target-a))
        next_v = velocity+next_a*dt

        # Native canCapture immediately rejects a brake latch, so no latch
        # recovery or braking-state propagation is needed inside this screen.
        for direction in (-1, 1):
            outward_v = next_v*direction
            outward_a = next_a*direction
            room = boundary-direction*xx
            if room <= 0 and outward_v <= .005 and outward_a < 0:
                continue
            if outward_v > 0 or outward_a > 0:
                travel = max(0., outward_v)*dt
                distance = _stopping_distance(outward_v, outward_a, brake, jerk)
                if travel+distance >= room-eps*(1+abs(room)):
                    return False

        # Preserve the native anticipation of velocity gained while ramping
        # acceleration to zero. Ambiguous branch boundaries fail closed.
        if next_a > 0:
            settled_v = next_v+next_a*next_a/two_jerk
            if abs(settled_v-speed) <= speed_epsilon:
                return False
            if settled_v >= speed:
                target = min(target, 0.)
        elif next_a < 0:
            settled_v = next_v-next_a*next_a/two_jerk
            if abs(settled_v+speed) <= speed_epsilon:
                return False
            if settled_v <= -speed:
                target = max(target, 0.)
        a += min(jd, max(-jd, target-a))
        velocity += a*dt
        xx += velocity*dt
        alpha = (9.81*sin(q)-a*cos(q))/leff
        q_unwrapped = q+w*dt+.5*alpha*dt*dt
        w += alpha*dt
        if not isfinite(q_unwrapped) or not isfinite(w):
            return False
        q = atan2(sin(q_unwrapped), cos(q_unwrapped))
        if (abs(q) >= giveup-eps*(1+giveup)
                or abs(xx) > position_limit-eps*(1+abs(position_limit))):
            return False

    final = gravity_over_length*q*q+w*w
    limit = initial*1.05+.01
    return isfinite(final) and final <= limit-eps*(1+limit)
