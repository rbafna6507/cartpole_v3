"""Offline hanging-to-upright calibration with reproducible validation evidence.

The correction is added after the ideal half-turn. This is a reference conversion,
not an encoder linearization or a camera calibration. Validation bounds are
empirical observed error envelopes, not statistical confidence guarantees.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

SCHEMA_VERSION = 1
FIT_VERSION = 'paired-circular-offset-v1'
MODEL_KIND = 'hanging_to_upright_offset'


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return float(value)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def _keys(value: Any, required: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f'{name} requires exactly these fields: {sorted(required)}')


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _count_period(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 4 or value % 2:
        raise ValueError('counts_per_turn must be an even positive integer >= 4')
    return value


def signed_count_delta(value: float, reference: float, counts_per_turn: int = 4096) -> float:
    """Shortest signed difference in [-N/2, N/2), including at the wrap boundary."""
    period = _count_period(counts_per_turn)
    return (_number(value, 'value') - _number(reference, 'reference') + period / 2) % period - period / 2


def _round_count(value: float) -> int:
    return int(math.copysign(math.floor(abs(value) + .5), value))


@dataclass(frozen=True)
class CalibrationPolicy:
    """Minimum evidence and maximum tolerated errors; stricter policies are allowed.

    The default 0.5 degree target includes window spread and the declared external
    reference error. No independence assumption or sqrt(sample_count) shrinkage
    is made. Approaches are positive/negative *raw encoder count* directions.
    """
    min_window_samples: int = 32
    min_window_duration_s: float = .4
    max_window_span_counts: float = 4.
    max_window_drift_counts: float = 2.
    min_train_pairs: int = 6
    min_validation_pairs: int = 4
    min_train_per_direction: int = 3
    min_validation_per_direction: int = 2
    min_train_sessions: int = 2
    min_validation_sessions: int = 2
    max_error_deg: float = .5
    max_direction_gap_counts: float = 3.
    max_session_gap_counts: float = 3.
    max_offset_counts: int = 256

    def __post_init__(self) -> None:
        floors = {'min_window_samples': 32, 'min_train_pairs': 6, 'min_validation_pairs': 4,
                  'min_train_per_direction': 3, 'min_validation_per_direction': 2,
                  'min_train_sessions': 2, 'min_validation_sessions': 2}
        for key, floor in floors.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(f'{key} must be an integer >= {floor}')
        for key in ('min_window_duration_s', 'max_window_span_counts', 'max_window_drift_counts',
                    'max_error_deg', 'max_direction_gap_counts', 'max_session_gap_counts'):
            if _number(getattr(self, key), key) <= 0:
                raise ValueError(f'{key} must be positive')
        ceilings = {'max_window_span_counts': 4., 'max_window_drift_counts': 2.,
                    'max_error_deg': .5, 'max_direction_gap_counts': 3.,
                    'max_session_gap_counts': 3.}
        for key, ceiling in ceilings.items():
            if getattr(self, key) > ceiling:
                raise ValueError(f'{key} cannot exceed acceptance limit {ceiling}')
        if self.min_window_duration_s < .4:
            raise ValueError('min_window_duration_s must be >= 0.4')
        if isinstance(self.max_offset_counts, bool) or not isinstance(self.max_offset_counts, int) or not 1 <= self.max_offset_counts <= 256:
            raise ValueError('max_offset_counts must be an integer from 1 to 256')


def _window(value: Any, period: int, policy: CalibrationPolicy) -> dict[str, Any]:
    _keys(value, {'window_id', 'sampling', 'raw_counts', 'sample_period_s'}, 'window')
    _text(value['window_id'], 'window_id')
    _text(value['sampling'], 'sampling')
    sample_period = _number(value['sample_period_s'], 'sample_period_s')
    if sample_period <= 0:
        raise ValueError('sample_period_s must be positive')
    raw = value['raw_counts']
    if not isinstance(raw, list) or not raw:
        raise ValueError('raw_counts must contain native readings')
    for count in raw:
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count < period:
            raise ValueError('raw_counts must be integer native counts in [0, counts_per_turn)')
    issues = []
    if value['sampling'] != 'native':
        issues.append('resampled or interpolated window is not independent native evidence')
    if len(raw) < policy.min_window_samples:
        issues.append('insufficient native samples')
    duration = (len(raw) - 1) * sample_period
    if duration < policy.min_window_duration_s:
        issues.append('window duration too short')
    # A circular mean chooses a wrap-safe branch; the estimate itself is a median.
    angles = [count * math.tau / period for count in raw]
    anchor = math.atan2(sum(map(math.sin, angles)), sum(map(math.cos, angles))) * period / math.tau
    unwrapped = [anchor + signed_count_delta(count, anchor, period) for count in raw]
    center = statistics.median(unwrapped)
    radius = max(abs(value - center) for value in unwrapped)
    span = max(unwrapped) - min(unwrapped)
    quarter = max(1, len(unwrapped) // 4)
    drift = abs(statistics.median(unwrapped[-quarter:]) - statistics.median(unwrapped[:quarter]))
    if span > policy.max_window_span_counts:
        issues.append('unstable window span')
    if drift > policy.max_window_drift_counts:
        issues.append('unstable window drift')
    return {'center_counts': center % period, 'radius_counts': radius, 'span_counts': span,
            'drift_counts': drift, 'duration_s': duration, 'samples': len(raw), 'issues': issues}


def _prepare_dataset(dataset: Any) -> dict[str, Any]:
    _keys(dataset, {'schema_version', 'hardware_id', 'source_id', 'fixture_id', 'sensor_config_id',
                    'frame_id', 'counts_per_turn', 'provenance', 'pairs'}, 'dataset')
    if type(dataset['schema_version']) is not int or dataset['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported dataset schema_version')
    for key in ('hardware_id', 'source_id', 'fixture_id', 'sensor_config_id', 'frame_id'):
        _text(dataset[key], key)
    _count_period(dataset['counts_per_turn'])
    provenance = dataset['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('provenance must be an object')
    for key in ('acquisition_version', 'source'):
        _text(provenance.get(key), f'provenance.{key}')
    if not isinstance(dataset['pairs'], list):
        raise ValueError('pairs must be an array')
    pair_ids = set()
    window_ids = set()
    for pair in dataset['pairs']:
        _keys(pair, {'pair_id', 'session_id', 'split', 'direction', 'reference_uncertainty_counts',
                     'upright', 'hanging'}, 'pair')
        for key in ('pair_id', 'session_id'):
            _text(pair[key], key)
        if pair['pair_id'] in pair_ids:
            raise ValueError('duplicate pair_id')
        pair_ids.add(pair['pair_id'])
        if pair['split'] not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        if pair['direction'] not in ('positive', 'negative'):
            raise ValueError('direction must be positive or negative raw-count approach')
        _number(pair['reference_uncertainty_counts'], 'reference_uncertainty_counts', minimum=0)
        for pose in ('upright', 'hanging'):
            if not isinstance(pair[pose], dict):
                raise ValueError('pose must contain native window evidence')
            identifier = _text(pair[pose].get('window_id'), 'window_id')
            if identifier in window_ids:
                raise ValueError('reused window_id: every pair needs independent upright and hanging captures')
            window_ids.add(identifier)
    result = json.loads(_canonical(dataset))
    result['pairs'].sort(key=lambda item: item['pair_id'])
    return result


def fit_calibration(dataset: Mapping[str, Any], policy: CalibrationPolicy | None = None) -> CalibrationProfile:
    """Fit on training pairs, freeze the offset, then check untouched sessions.

    Native stability failures and anomalous pairs remain visible and reject the
    model. Missing data produces insufficient_data, never an ideal-half-turn
    fallback. Session/direction labels and reference uncertainty must be measured
    honestly by acquisition; this function cannot authenticate physical poses.
    """
    policy = policy or CalibrationPolicy()
    if not isinstance(policy, CalibrationPolicy):
        raise ValueError('policy must be CalibrationPolicy')
    evidence = _prepare_dataset(dataset)
    period = evidence['counts_per_turn']
    rows, excluded = [], []
    rejection, missing = [], []
    for pair in evidence['pairs']:
        upright = _window(pair['upright'], period, policy)
        hanging = _window(pair['hanging'], period, policy)
        issues = [f'{pose}: {issue}' for pose, window in [('upright', upright), ('hanging', hanging)] for issue in window['issues']]
        if issues:
            excluded.append({'pair_id': pair['pair_id'], 'reasons': issues})
            continue
        offset = signed_count_delta(upright['center_counts'], hanging['center_counts'] + period / 2, period)
        rows.append({'pair_id': pair['pair_id'], 'session_id': pair['session_id'],
                     'split': pair['split'], 'direction': pair['direction'], 'offset_counts': offset,
                     'reference_uncertainty_counts': pair['reference_uncertainty_counts'],
                     'measurement_bound_counts': upright['radius_counts'] + hanging['radius_counts'] + pair['reference_uncertainty_counts'] + 1.0,
                     'upright_window': upright, 'hanging_window': hanging})
    if excluded:
        rejection.append('one or more capture windows failed native/stability quality checks')
    groups = {split: [r for r in rows if r['split'] == split] for split in ('train', 'validation')}
    sessions = {split: {r['session_id'] for r in group} for split, group in groups.items()}
    if sessions['train'] & sessions['validation']:
        rejection.append('training and heldout validation session identities overlap')
    coverage = {}
    for split, group in groups.items():
        by_direction = {direction: sum(r['direction'] == direction for r in group) for direction in ('positive', 'negative')}
        coverage[split] = {'pairs': len(group), 'sessions': sorted(sessions[split]), 'directions': by_direction}
        if len(group) < getattr(policy, f'min_{split}_pairs'):
            missing.append(f'insufficient {split} pairs')
        if len(sessions[split]) < getattr(policy, f'min_{split}_sessions'):
            missing.append(f'insufficient independent {split} sessions')
        for direction, count in by_direction.items():
            if count < getattr(policy, f'min_{split}_per_direction'):
                missing.append(f'insufficient {split} {direction} direction coverage')
    offsets = [r['offset_counts'] for r in groups['train']]
    estimate = statistics.median(offsets) if offsets else None
    integer_offset = _round_count(estimate) if estimate is not None else None
    mad = statistics.median(abs(value - estimate) for value in offsets) if offsets else None
    outliers = []
    error_bounds = {'train': [], 'validation': []}
    target_counts = policy.max_error_deg * period / 360
    for row in rows:
        if integer_offset is not None:
            row['residual_counts'] = signed_count_delta(row['offset_counts'], integer_offset, period)
            row['prediction_bound_counts'] = abs(row['residual_counts']) + row['measurement_bound_counts']
            error_bounds[row['split']].append(row['prediction_bound_counts'])
            # Outlier reporting never removes a disagreeing pair from acceptance checks.
            if row['split'] == 'train' and abs(row['offset_counts'] - estimate) > max(target_counts, 4 * 1.4826 * mad):
                outliers.append(row['pair_id'])
    if integer_offset is not None and abs(integer_offset) > policy.max_offset_counts:
        rejection.append('fitted offset exceeds bounded correction range')
    for split, bounds in error_bounds.items():
        if bounds and max(bounds) > target_counts:
            rejection.append(f'{split} prediction envelope exceeds error target')
    direction_centers = {direction: statistics.median([r['offset_counts'] for r in rows if r['direction'] == direction])
                         for direction in ('positive', 'negative') if any(r['direction'] == direction for r in rows)}
    direction_gap = abs(signed_count_delta(direction_centers['positive'], direction_centers['negative'], period)) if len(direction_centers) == 2 else None
    if direction_gap is not None and direction_gap > policy.max_direction_gap_counts:
        rejection.append('direction-dependent offset / hysteresis exceeds limit')
    session_centers = {session: statistics.median([r['offset_counts'] for r in rows if r['session_id'] == session])
                       for session in sorted(sessions['train'] | sessions['validation'])}
    session_gap = max(session_centers.values()) - min(session_centers.values()) if session_centers else None
    if session_gap is not None and session_gap > policy.max_session_gap_counts:
        rejection.append('session/history repeatability exceeds limit')
    status = 'rejected' if rejection else 'insufficient_data' if missing else 'validated'
    validation = {'status': status, 'reasons': rejection + missing, 'coverage': coverage,
                  'excluded_pairs': excluded, 'outlier_pair_ids': outliers, 'pair_results': rows,
                  'direction_centers_counts': direction_centers, 'direction_gap_counts': direction_gap,
                  'session_centers_counts': session_centers, 'session_gap_counts': session_gap,
                  'training_mad_counts': mad, 'error_target_deg': policy.max_error_deg,
                  'error_target_counts': target_counts,
                  'prediction_bound_counts': {split: max(values) if values else None for split, values in error_bounds.items()},
                  'bound_interpretation': 'Observed residual plus full window radii, one count for two quantized pose readings, and declared external reference bound; not a confidence interval.'}
    document = {key: evidence[key] for key in ('hardware_id', 'source_id', 'fixture_id', 'sensor_config_id', 'frame_id', 'counts_per_turn')}
    document.update(schema_version=SCHEMA_VERSION, fit_version=FIT_VERSION, model_kind=MODEL_KIND,
                    offset_counts=integer_offset, estimate_counts=estimate, policy=asdict(policy),
                    evidence=evidence, validation=validation)
    document['calibration_id'] = 'h2u-' + hashlib.sha256(_canonical(document).encode()).hexdigest()
    document['approval'] = None
    return CalibrationProfile(_canonical(document))


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable, evidence-carrying model. Use from_dict/load for external data."""
    _document_json: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._document_json)

    @property
    def calibration_id(self) -> str:
        return self.to_dict()['calibration_id']

    @property
    def offset_counts(self) -> int | None:
        return self.to_dict()['offset_counts']

    @property
    def status(self) -> str:
        return self.to_dict()['validation']['status']

    @classmethod
    def from_dict(cls, document: Any) -> CalibrationProfile:
        if not isinstance(document, dict):
            raise ValueError('profile must be an object')
        try:
            policy = CalibrationPolicy(**document['policy'])
            recomputed = fit_calibration(document['evidence'], policy).to_dict()
            approval = document['approval']
        except (KeyError, TypeError) as exc:
            raise ValueError(f'malformed profile: {exc}') from exc
        candidate = dict(document); candidate['approval'] = None
        if _canonical(recomputed) != _canonical(candidate):
            raise ValueError('profile integrity failed: model and validation must reproduce from evidence')
        if approval is not None:
            _validate_approval(approval, recomputed['calibration_id'])
            if recomputed['validation']['status'] != 'validated':
                raise ValueError('only validated profiles can carry approval')
        recomputed['approval'] = approval
        return cls(_canonical(recomputed))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        return cls.from_dict(load_json(path))

    def approve(self, *, approved_by: str, approved_at: str) -> CalibrationProfile:
        # Recompute even for locally constructed dataclasses, never trust a status string.
        verified = self.from_dict(self.to_dict())
        if verified.status != 'validated':
            raise ValueError('only a validated profile can be approved')
        document = verified.to_dict()
        document['approval'] = {'approved_by': approved_by, 'approved_at': approved_at,
                                'calibration_id': self.calibration_id}
        return self.from_dict(document)

    def approved_offset_counts(self, *, hardware_id: str, source_id: str, fixture_id: str,
                               sensor_config_id: str, counts_per_turn: int = 4096,
                               frame_id: str | None = None) -> int:
        verified = self.from_dict(self.to_dict())
        document = verified.to_dict()
        if verified.status != 'validated' or document['approval'] is None:
            raise ValueError('calibration must be validated and explicitly approved')
        expected = {'hardware_id': hardware_id, 'source_id': source_id, 'fixture_id': fixture_id,
                    'sensor_config_id': sensor_config_id, 'counts_per_turn': counts_per_turn}
        if frame_id is not None:
            expected['frame_id'] = frame_id
        if any(document[key] != value for key, value in expected.items()):
            raise ValueError('calibration identity/configuration mismatch')
        return document['offset_counts']

    def upright_from_hanging(self, raw_counts: int, **identity: Any) -> int:
        offset = self.approved_offset_counts(**identity)
        period = self.to_dict()['counts_per_turn']
        if isinstance(raw_counts, bool) or not isinstance(raw_counts, int) or not 0 <= raw_counts < period:
            raise ValueError('raw_counts must be an integer native encoder reading')
        return (raw_counts + period // 2 + offset) % period


def _validate_approval(approval: Any, calibration_id: str) -> None:
    _keys(approval, {'approved_by', 'approved_at', 'calibration_id'}, 'approval')
    _text(approval['approved_by'], 'approved_by')
    date = _text(approval['approved_at'], 'approved_at')
    try:
        parsed = datetime.fromisoformat(date.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('approved_at must be an ISO timestamp with timezone') from exc
    if parsed.tzinfo is None or approval['calibration_id'] != calibration_id:
        raise ValueError('approval must include timezone and exact calibration_id')


def load_json(path: str | Path) -> Any:
    """Read strict JSON: duplicate keys and nonfinite values are rejected."""
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value: str) -> None:
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='fit and validate native paired windows; never approves')
    fit.add_argument('dataset'); fit.add_argument('--output', required=True); fit.add_argument('--policy')
    report = commands.add_parser('report', help='recompute and print validation report')
    report.add_argument('profile')
    check = commands.add_parser('check', help='require approved profile and exact application identity')
    check.add_argument('profile')
    for key in ('hardware-id', 'source-id', 'fixture-id', 'sensor-config-id'):
        check.add_argument('--' + key, required=True)
    check.add_argument('--counts-per-turn', type=int, default=4096)
    approve = commands.add_parser('approve', help='record explicit operator approval after successful validation')
    approve.add_argument('profile'); approve.add_argument('--output', required=True)
    approve.add_argument('--approved-by', required=True); approve.add_argument('--approved-at', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'fit':
            policy = CalibrationPolicy(**load_json(args.policy)) if args.policy else None
            profile = fit_calibration(load_json(args.dataset), policy)
        else:
            profile = CalibrationProfile.load(args.profile)
        if args.command == 'approve':
            profile = profile.approve(approved_by=args.approved_by, approved_at=args.approved_at)
        if args.command in ('fit', 'approve'):
            Path(args.output).write_text(json.dumps(profile.to_dict(), indent=2, allow_nan=False) + '\n')
        if args.command == 'check':
            offset = profile.approved_offset_counts(hardware_id=args.hardware_id, source_id=args.source_id,
                fixture_id=args.fixture_id, sensor_config_id=args.sensor_config_id, counts_per_turn=args.counts_per_turn)
            print(json.dumps({'calibration_id': profile.calibration_id, 'offset_counts': offset}))
        else:
            print(json.dumps(profile.to_dict()['validation'], indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'calibration: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
