#!/usr/bin/env python3
"""Rebuild plots and verify frozen prediction scores from public numeric CSVs.

This is a descriptive replay, not a refit or plant acceptance test.
Run: python system_id/reproduce.py --output /tmp/cartpole-id-review
Dependencies: numpy>=1.26, matplotlib>=3.8
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

def read(name):
    with (ROOT / 'data' / name).open() as f:
        rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'plots')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Content hash mismatch: {name}')
    fit = json.loads((ROOT / 'fits/passive-frozen.json').read_text())
    scores = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, record in zip(axes, fit['continuous_rollouts']):
        data = read(f"manual-{record['episode_id']}.csv")
        error = data['frozen_prediction_rad'] - data['raw_theta_rad']
        np.testing.assert_allclose(error, data['error_rad'], atol=1e-12, rtol=0)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        p95 = float(np.percentile(np.abs(error), 95))
        np.testing.assert_allclose([rmse, p95], [record['rmse_rad'], record['p95_rad']], rtol=1e-10, atol=1e-12)
        scores.append({'episode': record['episode_id'], 'samples': len(error), 'rmse_rad': rmse, 'p95_rad': p95})
        time = data['time_s'] - data['time_s'][0]
        ax.plot(time, data['raw_theta_rad'], label='Recorded raw encoder', lw=1.5)
        ax.plot(time, data['frozen_prediction_rad'], label='Frozen model prediction', lw=1.2)
        ax.set(title=f"{record['episode_id']} | RMS {np.rad2deg(rmse):.2f}°, p95 {np.rad2deg(p95):.2f}°", ylabel='Angle (rad)')
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Time since first scored sample (s)')
    fig.suptitle('Manual passive releases — candidate fails repeatability gates')
    fig.savefig(args.output / 'passive-repeatability.png', dpi=160)
    plt.close(fig)
    pulse_checks = []
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, ['pos10-short','neg10-short','pos20-short','neg20-short','pos30-validation','neg30-validation']):
        data = read(f'pulse-{name}-01.csv')
        if not np.all(np.diff(data['time_s']) >= 0):
            raise ValueError(f'Nonmonotonic time: {name}')
        if np.any(data['fault']):
            raise ValueError(f'Unexpected fault in completed pulse: {name}')
        pulse_checks.append({'run': name, 'rows': len(data['time_s']), 'duplicate_timestamps': int(np.sum(np.diff(data['time_s']) == 0))})
        ax.plot(data['time_s'], 1000 * data['counted_x_m'], color='#245e94')
        ax.set(title=name, ylabel='Counted position (mm)', xlabel='Relative MCU time (s)')
        ax.grid(alpha=.2)
    fig.suptitle('Signed cart pulses — issued-step position, not measured displacement')
    fig.savefig(args.output / 'cart-pulses.png', dpi=160)
    plt.close(fig)
    (args.output / 'verification.json').write_text(json.dumps({'hashes_verified': len(manifest), 'passive_scores_reproduced': scores, 'pulse_checks': pulse_checks, 'physical_model_accepted': False, 'scope': 'Stored predictions and numeric telemetry only; original fitter and simulator are not re-executed.'}, indent=2) + '\n')
    print(f'Verified {len(manifest)} source files, reproduced 3 passive scores and plotted 6 pulses.')

if __name__ == '__main__':
    main()
