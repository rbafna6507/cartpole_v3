"""Bounded measured samples for charts; never fills in missing measurements."""
from collections import deque
import uuid


class MeasurementHistory:
    def __init__(self, seconds=15, capacity=4096):
        self.seconds = seconds
        self.samples = deque(maxlen=capacity)
        self.clear()

    def clear(self):
        self.samples.clear()
        self.epoch = uuid.uuid4().hex

    def append(self, state):
        if self.samples and (state['seq'] <= self.samples[-1]['seq'] or
                             state['t'] <= self.samples[-1]['t']):
            self.clear()
        self.samples.append({k: state.get(k) for k in (
            'seq', 't', 'theta_rad', 'omega_rad_s', 'cart_x_m',
            'angle_valid', 'cart_valid')})
        cutoff = state['t'] - self.seconds
        while self.samples and self.samples[0]['t'] < cutoff:
            self.samples.popleft()

    def read(self, after=None, epoch=None):
        last = self.samples[-1]['seq'] if self.samples else 0
        reset = epoch != self.epoch or after is None or after > last
        if self.samples and after is not None and after < self.samples[0]['seq'] - 1:
            reset = True
        rows = [dict(row) for row in self.samples if reset or row['seq'] > after]
        return dict(epoch=self.epoch, reset=reset, last_seq=last, samples=rows)
