from __future__ import annotations
import csv, json, math, os, time, hashlib, shutil, gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Iterable, Mapping, Sequence
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np

EPS=1e-10
TOL=1e-12
REPLAY_TOL=1e-10
IDENTITY_TOL=1e-10
def _finite_float(value: Any, default: float | None=None) -> float:
    if value is None:
        if default is None:
            raise ValueError('missing numeric value')
        return float(default)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'non-finite numeric value {value!r}')
    return result

def _count_field(role: Mapping[str, Any], name: str) -> int | None:
    if name not in role or role.get(name) is None:
        return None
    value = _finite_float(role.get(name))
    if value < -TOL or abs(value - round(value)) > 1e-08:
        raise ValueError(f'invalid normal role count {name}={value!r}')
    return max(0, int(round(value)))

def _child_counts(role: Mapping[str, Any]) -> tuple[dict[str, int] | None, int]:
    if 'child_service_counts' not in role or role.get('child_service_counts') is None:
        return (None, 0)
    raw = role.get('child_service_counts')
    if not isinstance(raw, Mapping):
        raise ValueError(f'child_service_counts is not a mapping: {raw!r}')
    counts: dict[str, int] = {}
    for service, value in raw.items():
        count = _finite_float(value)
        if count < -TOL or abs(count - round(count)) > 1e-08:
            raise ValueError(f'invalid child service count {service}={value!r}')
        counts[str(service)] = max(0, int(round(count)))
    return (counts, int(sum(counts.values())))

def _kernel_for_role(role: Mapping[str, Any] | None, candidate: Sequence[str]) -> dict[str, Any]:
    """Return a sparse row-stochastic destination kernel and role coverage."""
    if not isinstance(role, Mapping) or not role:
        return {'role_status': 'missing_role', 'role_evidence_status': None, 'normal_call_count': None, 'outbound_role_span_count': None, 'both_count': None, 'outbound_only_count': 0, 'b_outbound_only_fraction': 0.0, 'child_service_counts': {}, 'child_count_total_h': 0, 'self_mass': 1.0, 'destinations': {}, 'identity_kernel': True, 'domain_destination_count': 0}
    required = ('normal_call_count', 'outbound_role_span_count', 'both_count', 'child_service_counts')
    missing = [name for name in required if name not in role or role.get(name) is None]
    if missing:
        return {'role_status': 'missing_role_fields', 'role_evidence_status': str(role.get('role_evidence_status', '')) or None, 'normal_call_count': _count_field(role, 'normal_call_count'), 'outbound_role_span_count': _count_field(role, 'outbound_role_span_count'), 'both_count': _count_field(role, 'both_count'), 'outbound_only_count': 0, 'b_outbound_only_fraction': 0.0, 'child_service_counts': {}, 'child_count_total_h': 0, 'self_mass': 1.0, 'destinations': {}, 'identity_kernel': True, 'domain_destination_count': 0, 'missing_role_fields': missing}
    normal_count = _count_field(role, 'normal_call_count')
    outbound_count = _count_field(role, 'outbound_role_span_count')
    both_count = _count_field(role, 'both_count')
    assert normal_count is not None and outbound_count is not None and (both_count is not None)
    if both_count > outbound_count or outbound_count > normal_count:
        raise ValueError(f'normal role count out of bounds: normal_call_count={normal_count}, outbound_role_span_count={outbound_count}, both_count={both_count}')
    child_counts, h = _child_counts(role)
    if child_counts is None:
        child_counts = {}
    outbound_only = outbound_count - both_count
    b = float(outbound_only / normal_count) if normal_count > 0 else 0.0
    destinations: dict[str, float] = {}
    if normal_count > 0 and h > 0 and (outbound_only > 0):
        for service, count in child_counts.items():
            if service in candidate and count > 0:
                destinations[service] = float(b * count / h)
        if destinations:
            pass
    destination_sum = float(sum(destinations.values()))
    if destination_sum > 1.0 + IDENTITY_TOL:
        raise ValueError(f'destination kernel exceeds one: {destination_sum}')
    return {'role_status': 'observed_role_fields', 'role_evidence_status': str(role.get('role_evidence_status', '')) or None, 'normal_call_count': normal_count, 'outbound_role_span_count': outbound_count, 'both_count': both_count, 'outbound_only_count': outbound_only, 'b_outbound_only_fraction': b, 'child_service_counts': child_counts, 'child_count_total_h': h, 'self_mass': float(1.0 - destination_sum), 'destinations': destinations, 'identity_kernel': not bool(destinations), 'domain_destination_count': len(destinations)}

def _kernel_for_operation(source_service: str, role: Mapping[str, Any] | None, candidate: Sequence[str]) -> dict[str, Any]:
    kernel = _kernel_for_role(role, candidate)
    destinations = {service: float(mass) for service, mass in (kernel.get('destinations') or {}).items() if service != source_service and float(mass) > 0.0}
    destination_sum = float(sum(destinations.values()))
    kernel['destinations'] = destinations
    kernel['self_mass'] = float(1.0 - destination_sum)
    kernel['identity_kernel'] = not bool(destinations)
    kernel['domain_destination_count'] = len(destinations)
    kernel['kernel_row_sum'] = float(kernel['self_mass'] + destination_sum)
    return kernel
