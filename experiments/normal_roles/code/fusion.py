from __future__ import annotations
import csv, json, math, os, time, hashlib, shutil, gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Iterable, Mapping, Sequence
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import copy

EPS=1e-10
TOL=1e-12
REPLAY_TOL=1e-10
IDENTITY_TOL=1e-10
def rank_by_score(score: Sequence[float], candidate: Sequence[str]) -> list[str]:
    values = np.asarray(score, dtype=np.float64)
    order = sorted(range(len(candidate)), key=lambda index: (-float(values[index]), index))
    return [str(candidate[index]) for index in order]

def expected_metrics(score: Sequence[float], candidate: Sequence[str], root: str) -> dict[str, Any]:
    values = np.asarray(score, dtype=np.float64)
    names = [str(item) for item in candidate]
    if root not in names:
        return {'candidate_size': len(names), 'root_service': root, 'root_in_output': False, 'list_rank': None, 'expected_rank': None, 'expected_rr': 0.0, 'list_rank_mrr': 0.0, 'AC@1': 0.0, 'AC@3': 0.0, 'AC@5': 0.0, 'Avg@5': 0.0}
    root_value = float(values[names.index(root)])
    higher = int(np.sum(values > root_value + TOL))
    equal = int(np.sum(np.abs(values - root_value) <= TOL))
    expected_rank = float(higher + (equal + 1.0) / 2.0)
    expected_rr = float(np.mean([1.0 / (higher + offset) for offset in range(1, equal + 1)]))
    probabilities: dict[int, float] = {}
    for k_value in range(1, 6):
        if higher >= k_value:
            probabilities[k_value] = 0.0
        elif higher + equal <= k_value:
            probabilities[k_value] = 1.0
        else:
            probabilities[k_value] = float((k_value - higher) / equal)
    rank = rank_by_score(values, names)
    return {'candidate_size': len(names), 'root_service': root, 'root_in_output': True, 'list_rank': int(rank.index(root) + 1), 'expected_rank': expected_rank, 'expected_rr': expected_rr, 'list_rank_mrr': expected_rr, 'top1': probabilities[1], 'top3': probabilities[3], 'top5': probabilities[5], 'AC@1': probabilities[1], 'AC@2': probabilities[2], 'AC@3': probabilities[3], 'AC@4': probabilities[4], 'AC@5': probabilities[5], 'Avg@5': float(np.mean([probabilities[k_value] for k_value in range(1, 6)]))}

def ordinary_metrics(rank: Sequence[str], root: str) -> dict[str, Any]:
    names = [str(item) for item in rank]
    position = names.index(root) + 1 if root in names else None
    return {'candidate_size': len(names), 'root_service': root, 'root_in_output': position is not None, 'list_rank': position, 'expected_rank': float(position) if position is not None else None, 'expected_rr': float(1.0 / position) if position is not None else 0.0, 'list_rank_mrr': float(1.0 / position) if position is not None else 0.0, 'top1': float(position == 1) if position is not None else 0.0, 'top3': float(position is not None and position <= 3), 'top5': float(position is not None and position <= 5), 'AC@1': float(position == 1) if position is not None else 0.0, 'AC@2': float(position is not None and position <= 2), 'AC@3': float(position is not None and position <= 3), 'AC@4': float(position is not None and position <= 4), 'AC@5': float(position is not None and position <= 5), 'Avg@5': float(np.mean([float(position is not None and position <= k_value) for k_value in range(1, 6)]))}

def score_payload(score: Sequence[float], candidate: Sequence[str], operator: str, extra: Mapping[str, Any] | None=None) -> dict[str, Any]:
    values = np.asarray(score, dtype=np.float64).copy()
    if values.shape != (len(candidate),) or not np.all(np.isfinite(values)) or np.any(values < -REPLAY_TOL):
        raise ValueError('invalid score')
    values[np.abs(values) <= REPLAY_TOL] = 0.0
    if abs(float(np.sum(values)) - 1.0) > 5 * REPLAY_TOL:
        raise ValueError('score simplex violation')
    output: dict[str, Any] = {'score_vector': values, 'rank': rank_by_score(values, candidate), 'score_sum': float(np.sum(values)), 'score_min': float(np.min(values)), 'operator': operator}
    if extra:
        output.update(copy.deepcopy(dict(extra)))
    return output

def masked_values(x: Sequence[float], w: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.float64).copy()
    weights = np.asarray(w, dtype=np.float64).copy()
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError('source shape mismatch')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError('source nonfinite')
    if np.any(values < -REPLAY_TOL):
        raise ValueError('material negative source mass')
    values[np.abs(values) <= REPLAY_TOL] = 0.0
    support = weights > 0.0
    values[~support] = 0.0
    return (values, support)

def trace_probability(x: Sequence[float], w: Sequence[float]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values, support = masked_values(x, w)
    n = len(values)
    idx = np.flatnonzero(support)
    total = float(np.sum(values))
    if total > REPLAY_TOL:
        t = values / total
        informative = True
        reason = 'positive_trace_mass_normalized_on_support'
    else:
        t = np.zeros(n, dtype=np.float64)
        informative = False
        reason = 'empty_or_zero_trace_support'
    return (t, support, {'support_count': int(idx.size), 'support_fraction': float(idx.size / n), 'mass_total': total, 'support_empty': bool(idx.size == 0), 'mass_zero': bool(total <= REPLAY_TOL), 'informative': informative, 'reason': reason})

def arithmetic(p: np.ndarray, t: np.ndarray, support: np.ndarray, trace_info: Mapping[str, Any]) -> tuple[np.ndarray, float, str]:
    if not bool(trace_info['informative']):
        return (p.copy(), 0.0, 'trace_empty_or_zero_identity')
    m = float(np.sum(p[support]))
    if m <= REPLAY_TOL:
        return (p.copy(), m, 'metric_base_zero_on_trace_support_identity')
    result = p.copy()
    result[support] = (p[support] + m * t[support]) / 2.0
    return (result, m, None)
