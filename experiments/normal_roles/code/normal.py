from __future__ import annotations
import csv, json, math, os, time, hashlib, shutil, gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Iterable, Mapping, Sequence
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import roles, fusion

EPS=1e-10
TOL=1e-12
REPLAY_TOL=1e-10
IDENTITY_TOL=1e-10
def clean(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [clean(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

def sha_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def aggregate_role(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_counts: Counter[str] = Counter()
    child_counts: Counter[str] = Counter()
    inbound_edges = 0
    outbound_edges = 0
    inbound_spans = 0
    outbound_spans = 0
    both = 0
    neither = 0
    for item in observations:
        inbound_services = [str(value) for value in item.get('inbound_services', [])]
        outbound_services = [str(value) for value in item.get('outbound_services', [])]
        inbound_edges += len(inbound_services)
        outbound_edges += len(outbound_services)
        inbound_spans += int(bool(inbound_services))
        outbound_spans += int(bool(outbound_services))
        both += int(bool(inbound_services) and bool(outbound_services))
        neither += int(not inbound_services and (not outbound_services))
        parent_counts.update(inbound_services)
        child_counts.update(outbound_services)
    n = len(observations)
    return {'role_evidence_status': 'observable_raw_trace_parent_child' if n else 'observed_role_empty_after_sampling', 'role_class': 'sampled_complete_role_units', 'reason': 'one normal unique span plus all overlapping cross-service parent/child relations', 'normal_call_count': n, 'cross_service_inbound_edge_count': inbound_edges, 'cross_service_outbound_edge_count': outbound_edges, 'inbound_role_span_count': inbound_spans, 'outbound_role_span_count': outbound_spans, 'both_count': both, 'neither_count': neither, 'both_proportion': float(both / n) if n else 0.0, 'neither_proportion': float(neither / n) if n else 0.0, 'inbound_proportion': float(inbound_spans / n) if n else 0.0, 'outbound_proportion': float(outbound_spans / n) if n else 0.0, 'parent_service_counts': dict(sorted(parent_counts.items())), 'child_service_counts': dict(sorted(child_counts.items()))}

def sampled_order(observations: Sequence[Mapping[str, Any]], seed: int) -> np.ndarray:
    ordered = list(range(len(observations)))
    rng = np.random.default_rng(int(seed))
    return np.asarray(rng.permutation(ordered), dtype=np.int64)

def selected_for(observations: Sequence[Mapping[str, Any]], seed: int | None, retention: float, cache: dict[int, np.ndarray]) -> list[Mapping[str, Any]]:
    if retention == 1.0:
        return list(observations)
    assert seed is not None
    if seed not in cache:
        cache[seed] = sampled_order(observations, seed)
    count = int(math.floor(float(retention) * len(observations)))
    return [observations[int(index)] for index in cache[seed][:count]]

def role_selected_hash(observations: Sequence[Mapping[str, Any]]) -> str:
    return sha_payload([row['stable_id'] for row in observations])

def trace_from_roles(operations: Sequence[Mapping[str, Any]], op_observations: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]], retention: float, seed: int | None, candidate: Sequence[str], saved_kernels: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    index = {str(name): i for i, name in enumerate(candidate)}
    by_pair = {(str(row['service']), str(row['operation'])): row for row in operations}
    kernel_order = [(str(row['service']), str(row['operation'])) for row in saved_kernels]
    for pair in sorted(by_pair):
        if pair not in kernel_order:
            kernel_order.append(pair)
    x = np.zeros(len(candidate), dtype=np.float64)
    support = np.zeros(len(candidate), dtype=bool)
    q_total_candidate = 0.0
    role_rows: list[dict[str, Any]] = []
    baseline_kernel_max = 0.0
    for pair in kernel_order:
        op = by_pair.get(pair)
        if op is None:
            continue
        obs = list(op_observations.get(pair, []))
        order_cache: dict[int, np.ndarray] = {}
        chosen = selected_for(obs, seed, retention, order_cache)
        role = aggregate_role(chosen) if chosen else None
        kernel = roles._kernel_for_operation(pair[0], role, candidate)
        energy = float(op['q_l1'])
        if pair[0] in index and float(kernel.get('self_mass', 0.0)) > 0.0:
            support[index[pair[0]]] = True
            x[index[pair[0]]] += energy * float(kernel['self_mass'])
        for destination, weight in (kernel.get('destinations') or {}).items():
            if float(weight) <= 0.0:
                continue
            if str(destination) in index:
                support[index[str(destination)]] = True
                x[index[str(destination)]] += energy * float(weight)
        role_rows.append({'service': pair[0], 'operation': pair[1], 'q_l1': energy, 'n_so': role.get('normal_call_count') if role else None, 'u_so': role.get('outbound_role_span_count') if role else None, 'b_so': role.get('both_count') if role else None, 'h_so': role.get('child_count_total_h') if role else 0, 'kernel': kernel, 'raw_observation_count': len(obs), 'retained_observation_count': len(chosen), 'retained_observation_sha256': role_selected_hash(chosen)})
    q_total_candidate = float(np.sum(x))
    normalized = x / q_total_candidate if q_total_candidate > REPLAY_TOL else np.zeros_like(x)
    trace = {'x': normalized, 'W': support, 'q_total': q_total_candidate, 'support_count': int(np.sum(support)), 'mode': 'l1', 'routing': 'destination', 'informative': bool(q_total_candidate > REPLAY_TOL)}
    return (trace, {'operations': role_rows, 'baseline_kernel_max_abs_error': baseline_kernel_max})

def evaluate_full(source: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    candidate = [str(item) for item in source['candidate_order']]
    root = str(source['labels']['root_service'])
    p = np.asarray(source['arms']['NO_TRACE']['score_vector'], dtype=np.float64)
    raw_x = np.asarray(trace['x'], dtype=np.float64)
    support = np.asarray(trace['W'], dtype=bool)
    t, s, info = fusion.trace_probability(raw_x, support.astype(float))
    full, mass, fallback = fusion.arithmetic(p, t, s, info)
    payload = fusion.score_payload(full, candidate, 'normal_role_rebuilt_l1_destination_then_support_conditional_arithmetic', {'trace_q_total': float(trace['q_total']), 'trace_support_count': int(np.sum(s)), 'trace_informative': bool(info['informative']), 'trace_fallback': fallback, 'metric_support_mass_m': float(mass)})
    payload['score_vector'] = payload['score_vector'].tolist()
    payload['metrics'] = {'expected_tie_aware': fusion.expected_metrics(full, candidate, root), 'ordinary_name_rank': fusion.ordinary_metrics(payload['rank'], root)}
    return payload

def evaluate_identity(source: Mapping[str, Any], operations: Sequence[Mapping[str, Any]], kernel_order: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate = [str(item) for item in source['candidate_order']]
    root = str(source['labels']['root_service'])
    index = {name: i for i, name in enumerate(candidate)}
    by_pair = {(str(op['service']), str(op['operation'])): op for op in operations}
    order = [(str(op['service']), str(op['operation'])) for op in kernel_order]
    order.extend(pair for pair in sorted(by_pair) if pair not in order)
    mass = np.zeros(len(candidate), dtype=np.float64)
    support = np.zeros(len(candidate), dtype=float)
    for pair in order:
        if pair in by_pair and pair[0] in index:
            mass[index[pair[0]]] += float(by_pair[pair]['q_l1'])
            support[index[pair[0]]] = 1.0
    total = float(mass.sum())
    normalized = mass / total if total > REPLAY_TOL else np.zeros_like(mass)
    trace, mask, info = fusion.trace_probability(normalized, support)
    metric = np.asarray(source['arms']['NO_TRACE']['score_vector'], dtype=np.float64)
    score, _, _ = fusion.arithmetic(metric, trace, mask, info)
    payload = fusion.score_payload(score, candidate, 'identity_destination_control')
    payload['score_vector'] = payload['score_vector'].tolist()
    payload['metrics'] = {'expected_tie_aware': fusion.expected_metrics(score, candidate, root), 'ordinary_name_rank': fusion.ordinary_metrics(payload['rank'], root)}
    return payload
