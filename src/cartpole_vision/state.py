"""Timestamped measurements, calibration and causal velocity estimates."""
from collections import deque
import math
import numpy as np


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


class Derivative:
    """Local linear slope with an explicit effective time; no future frames."""
    def __init__(self, angular=False, window=.05, max_gap=.075, max_window=.12):
        self.angular, self.window, self.max_gap = angular, window, max_gap
        self.max_window = max_window
        self.samples = deque()
        self.last_raw = None
        self.span = None
        self.effective_t = None
        self.slope_se = None

    def clear(self):
        self.samples.clear()
        self.last_raw = None
        self.span = None
        self.effective_t = None
        self.slope_se = None

    def observe(self, t, value):
        if not math.isfinite(t) or not math.isfinite(value):
            self.clear()
            return None
        if self.samples and (t <= self.samples[-1][0] or t-self.samples[-1][0] > self.max_gap):
            self.clear()
        unwrapped = value
        if self.angular and self.samples:
            unwrapped = self.samples[-1][1]+wrap(value-self.last_raw)
        self.last_raw = value
        self.samples.append((t,unwrapped))
        # Keep four observations when delivery is slower than the nominal
        # window. A strict 80 ms window could never fit four frames at 30 FPS.
        while len(self.samples)>4 and t-self.samples[0][0]>self.window+1e-9:
            self.samples.popleft()
        while len(self.samples)>1 and t-self.samples[0][0]>self.max_window+1e-9:
            self.samples.popleft()
        self.span = t-self.samples[0][0]
        self.effective_t = self.slope_se = None
        if len(self.samples)<4 or self.span<.025:
            return None
        data=np.asarray(self.samples)
        # Local linear regression reduces differentiation noise. Its effective
        # time is near the middle of the window, not the latest frame. The
        # expression below is exact for constant acceleration with irregular dt.
        centered=data[:,0]-data[:,0].mean()
        variance=float(centered@centered)
        self.effective_t=float(data[:,0].mean()+np.sum(centered**3)/(2*variance))
        slope=float(centered@(data[:,1]-data[:,1].mean())/variance)
        residual=data[:,1]-data[:,1].mean()-slope*centered
        # Regression fit noise only: not a calibrated physical accuracy bound.
        self.slope_se=float(np.sqrt((residual@residual)/(len(data)-2)/variance))
        return slope



class StateEstimator:
    def __init__(self, stale_after=.15):
        self.stale_after=stale_after
        self.home_angle=None
        self.home_length=None
        self.cart_a=None
        self.cart_b=None
        self.cart_span_mm=None
        self.home_samples=deque(maxlen=25)
        self.cart_samples=deque(maxlen=25)
        self.angular=Derivative(angular=True)
        self.cart_velocity=Derivative()
        self.last_t=None
        self.last_cart_source=None
        self.last_state={}

    def reset(self):
        self.__init__(self.stale_after)

    def invalidate_measurement(self):
        # Calibration metadata must never accompany samples from the prior frame.
        self.last_state={'calibrated':self.home_angle is not None}
        self.last_t=None
        self.angular.clear();self.cart_velocity.clear()

    def update(self, pose, t, cart_pivot=None):
        if not math.isfinite(t):
            raise ValueError('timestamp must be finite')
        if self.last_t is not None and (t<=self.last_t or t-self.last_t>self.stale_after):
            self.angular.clear(); self.cart_velocity.clear()
            self.home_samples.clear(); self.cart_samples.clear()
        self.last_t=t
        state=dict(t=t,angle_valid=False,cart_valid=False,calibrated=self.home_angle is not None,
                   theta_rad=None,theta_down_rad=None,omega_rad_s=None,
                   image_angle_rad=None,pivot_px=None,bob_px=None,length_px=None,
                   confidence=0.,cart_x_m=None,cart_v_m_s=None,cart_cross_track_px=None,
                   cart_extrapolated=False,angular_window_s=None,cart_window_s=None,
                   omega_effective_t=None,cart_velocity_effective_t=None,
                   omega_delay_s=None,omega_samples=0,omega_fit_se_rad_s=None,
                   velocity_status='no arm measurement',
                   status='no arm measurement',cart_source=None)
        length_ok=False
        if pose is not None:
            state.update(image_angle_rad=pose.angle,pivot_px=list(pose.pivot),bob_px=list(pose.bob),
                         length_px=pose.length,confidence=pose.confidence)
            length_ok=self.home_length is None or abs(pose.length/self.home_length-1)<=.20
            if length_ok:
                self.home_samples.append((pose.angle,pose.length))
                state['status']='set hanging zero' if self.home_angle is None else 'measured'
                state['velocity_status']='needs hanging zero'
                if self.home_angle is not None:
                    down=wrap(pose.angle-self.home_angle)
                    theta=wrap(math.pi-down)
                    state.update(angle_valid=True,theta_rad=theta,theta_down_rad=down,
                                 omega_rad_s=self.angular.observe(t,theta),angular_window_s=self.angular.span,
                                 omega_effective_t=self.angular.effective_t)
                    state.update(omega_samples=len(self.angular.samples),
                                 omega_delay_s=None if self.angular.effective_t is None else t-self.angular.effective_t,
                                 omega_fit_se_rad_s=self.angular.slope_se,
                                 velocity_status='measured' if state['omega_rad_s'] is not None else 'warming up')
            else:
                self.home_samples.clear();self.angular.clear()
                state['status']='arm length inconsistent with calibration'
        else:
            self.home_samples.clear()
            # No output on a missing frame. Retain only recent real observations
            # so one miss does not force another full derivative warmup.
            if self.angular.samples and t-self.angular.samples[-1][0]>self.angular.max_gap:
                self.angular.clear()
        # An independently measured cart template can outlive an occluded arm.
        p=cart_pivot if cart_pivot is not None else (pose.pivot if pose is not None and length_ok else None)
        state['cart_source']='template' if cart_pivot is not None else ('arm pivot' if p is not None else None)
        if state['cart_source'] != self.last_cart_source:
            self.cart_velocity.clear()
            self.cart_samples.clear()
        self.last_cart_source=state['cart_source']
        state['cart_pivot_px']=None if p is None else list(p)
        if p is None:
            self.cart_samples.clear();self.cart_velocity.clear()
        else:
            p=np.asarray(p,float)
            self.cart_samples.append(p)
            if self.cart_a is not None and self.cart_b is not None:
                a,b=np.asarray(self.cart_a),np.asarray(self.cart_b)
                direction=b-a
                f=float((p-a)@direction/(direction@direction))
                cross=float(np.linalg.norm(p-a-f*direction))
                state['cart_cross_track_px']=cross
                if cross<=max(4.,np.linalg.norm(direction)*.02):
                    x=(f-.5)*self.cart_span_mm/1000
                    state.update(cart_valid=True,cart_x_m=x,cart_v_m_s=self.cart_velocity.observe(t,x),
                                 cart_window_s=self.cart_velocity.span,cart_extrapolated=not 0<=f<=1,
                                 cart_velocity_effective_t=self.cart_velocity.effective_t)
                else:
                    self.cart_velocity.clear()
        self.last_state=state
        return dict(state)

    def home(self):
        if len(self.home_samples)<self.home_samples.maxlen:
            raise ValueError('Need 25 consecutive measurements with the pendulum hanging still')
        a=np.array([p[0] for p in self.home_samples])
        mean=math.atan2(np.sin(a).mean(),np.cos(a).mean())
        if max(abs(wrap(x-mean)) for x in a)>math.radians(1.5):
            raise ValueError('Arm is moving; hold it hanging still')
        lengths=np.array([p[1] for p in self.home_samples])
        if np.ptp(lengths)>np.median(lengths)*.05:
            raise ValueError('Arm geometry is unstable')
        self.home_angle=mean;self.home_length=float(np.median(lengths))
        self.invalidate_measurement()

    def mark_cart(self, which, span_mm=None):
        if which not in ('a','b'):
            raise ValueError('Reference must be a or b')
        if len(self.cart_samples)<self.cart_samples.maxlen:
            raise ValueError('Need 25 consecutive measurements with the cart still')
        points=np.array(self.cart_samples);center=np.median(points,axis=0)
        if np.max(np.linalg.norm(points-center,axis=1))>2.5:
            raise ValueError('Cart is moving or pivot detection is unstable')
        if which=='a':
            self.cart_a=center.tolist();self.cart_b=None;self.cart_span_mm=None
        else:
            if self.cart_a is None:
                raise ValueError('Record reference A first')
            if span_mm is None or not math.isfinite(span_mm) or not 10<=span_mm<=2000:
                raise ValueError('Enter measured separation of 10–2000 mm')
            if np.linalg.norm(center-np.asarray(self.cart_a))<40:
                raise ValueError('References must be at least 40 pixels apart')
            self.cart_b=center.tolist();self.cart_span_mm=float(span_mm)
        self.cart_velocity.clear()
        self.cart_samples.clear()
        self.invalidate_measurement()

    def calibration(self):
        return dict(home_angle_rad=self.home_angle,home_length_px=self.home_length,
                    cart_a_px=self.cart_a,cart_b_px=self.cart_b,cart_span_mm=self.cart_span_mm)

    def snapshot(self, now):
        state=dict(self.last_state)
        age=None if self.last_t is None else max(0.,now-self.last_t)
        state['age_since_host_read_s']=age
        state['stale']=age is None or age>self.stale_after
        if state['stale']:
            state.update(angle_valid=False,cart_valid=False,theta_rad=None,theta_down_rad=None,
                         omega_rad_s=None,cart_x_m=None,cart_v_m_s=None,
                         omega_effective_t=None,cart_velocity_effective_t=None,
                         omega_delay_s=None,omega_fit_se_rad_s=None,velocity_status='camera stale',
                         status='camera stale',confidence=0.)
        state['home_samples']=len(self.home_samples)
        state['cart_samples']=len(self.cart_samples)
        state['calibration']=self.calibration()
        return state
