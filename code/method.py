"""Implementation of the current TopoTrace-AD scoring method.

The functions in this module are a small dependency-free (apart from NumPy)
reading implementation of the method used for the supplied evidence package.
They do not load a model or a dataset.  A caller supplies the normal/incident
metric arrays, operation-bin summaries, and normal-observed role records.
The package README and prepared_observations.jsonl show the data contract.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EPS = 1.0e-10


def canonical_service(name: str) -> str:
    """Map a fine metric or role name to the service-level candidate name."""
    value = str(name).strip()
    value = value.split("_", 1)[0]
    if value.endswith("service"):
        value = value[:-7] + "service"
    if value == "frontendservice":
        return "frontend"
    if value.endswith("-db"):
        value = value[:-3]
    return value


def _finite_forward_fill(values: np.ndarray) -> np.ndarray:
    """Replace infinities with missing, forward-fill, then use zero."""
    arr = np.asarray(values, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    for j in range(arr.shape[1]):
        last = np.nan
        for i in range(arr.shape[0]):
            if np.isfinite(arr[i, j]):
                last = arr[i, j]
            elif np.isfinite(last):
                arr[i, j] = last
        arr[~np.isfinite(arr[:, j]), j] = 0.0
    return arr


def preprocess_metric_columns(
    normal: Sequence[Sequence[float]],
    incident: Sequence[Sequence[float]],
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply the fixed column/window preprocessing used by the method.

    Inputs are already the last normal and first incident windows.  The caller
    is responsible for selecting those windows and the fixed row sampling.
    """
    n = _finite_forward_fill(np.asarray(normal, dtype=np.float64))
    a = _finite_forward_fill(np.asarray(incident, dtype=np.float64))
    if n.ndim != 2 or a.ndim != 2 or n.shape[1] != a.shape[1]:
        raise ValueError("normal and incident metric arrays must be 2-D with equal columns")
    if len(names) != n.shape[1]:
        raise ValueError("metric names do not match metric columns")
    keep: list[int] = []
    clean_names: list[str] = []
    rename: dict[int, str] = {}
    for j, raw in enumerate(names):
        text = str(raw)
        if text.endswith("_latency-50"):
            continue
        if text.endswith("_latency-90"):
            text = text[:-len("-90")]
        keep.append(j)
        clean_names.append(text)
        rename[j] = text
    n = n[:, keep]
    a = a[:, keep]
    for j, text in enumerate(clean_names):
        if text.endswith("_mem"):
            n[:, j] /= 1.0e6
            a[:, j] /= 1.0e6
    # Constant within either partition is excluded, matching the fixed
    # observable-column boundary.  Empty inputs remain explicit errors.
    if n.shape[0] == 0 or a.shape[0] == 0:
        raise ValueError("empty metric window")
    variable = (np.ptp(n, axis=0) > 0.0) & (np.ptp(a, axis=0) > 0.0)
    return n[:, variable], a[:, variable], [x for x, ok in zip(clean_names, variable) if ok]


def normal_calibrated_metric_field(
    normal: Sequence[Sequence[float]],
    incident: Sequence[Sequence[float]],
    fine_names: Sequence[str],
    candidates: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Create the nonnegative normalized metric field and support mask."""
    n, a, names = preprocess_metric_columns(normal, incident, fine_names)
    if not names:
        return np.full(len(candidates), 1.0 / len(candidates)), np.zeros(len(candidates)), {}
    mean = np.mean(n, axis=0)
    scale = np.std(n, axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
    values = np.max(np.abs((a - mean) / scale), axis=0)
    values = np.where(np.isfinite(values) & (values >= 0.0), values, 0.0)
    total = float(np.sum(values))
    fine_mass = values / total if total > EPS and np.isfinite(total) else np.zeros_like(values)
    service_mass: dict[str, float] = defaultdict(float)
    for name, mass in zip(names, fine_mass):
        service_mass[canonical_service(name)] += float(mass)
    index = {str(s): i for i, s in enumerate(candidates)}
    x = np.zeros(len(candidates), dtype=np.float64)
    w = np.zeros(len(candidates), dtype=np.float64)
    for service, mass in service_mass.items():
        if service in index:
            x[index[service]] = mass
            w[index[service]] = 1.0
    support = x > EPS
    if not np.any(support):
        support = w > 0.0
    mass = float(np.sum(np.where(support, x, 0.0)))
    if mass > EPS:
        p = np.where(support, x / mass, 0.0)
    elif np.any(support):
        p = support.astype(np.float64) / float(np.sum(support))
    else:
        p = np.full(len(candidates), 1.0 / max(len(candidates), 1), dtype=np.float64)
    return p, w, dict(service_mass)


def fixed_median(values: Sequence[float]) -> float:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    return float(np.median(arr)) if arr.size else 0.0


def operation_persistence(
    normal_bins: Sequence[float],
    incident_bins: Sequence[float],
    operation_count: int,
    total_observed_bins: int,
) -> dict[str, float]:
    """Return robust and mean signed locations plus their squared energies."""
    normal = np.asarray(normal_bins, dtype=np.float64)
    incident = np.asarray(incident_bins, dtype=np.float64)
    normal = normal[np.isfinite(normal)]
    incident = incident[np.isfinite(incident)]
    mu = float(np.mean(normal)) if normal.size else 0.0
    sigma = float(np.std(normal, ddof=0)) if normal.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    z = (incident - mu) / sigma
    z = z[np.isfinite(z)]
    coverage = float(len(z) / total_observed_bins) if total_observed_bins > 0 else 0.0
    t_median = coverage * fixed_median(z)
    t_mean = coverage * float(np.mean(z)) if z.size else 0.0
    denom = float(operation_count) if operation_count > 0 else 1.0
    return {
        "coverage": coverage,
        "t_median": float(t_median),
        "t_mean": float(t_mean),
        "q_median": float(t_median * t_median / denom),
        "q_mean": float(t_mean * t_mean / denom),
    }


def destination_kernel(
    source: str,
    candidates: Sequence[str],
    child_counts: Mapping[str, float] | None,
    n_calls: int,
    outbound_span_count: int,
    both_role_count: int,
    *,
    identity: bool = False,
) -> np.ndarray:
    """Construct one complete normal-observed responsibility row."""
    services = list(map(str, candidates))
    result = np.zeros(len(services), dtype=np.float64)
    if source not in services:
        return result
    source_idx = services.index(source)
    result[source_idx] = 1.0
    if identity:
        return result
    counts = {str(k): float(v) for k, v in (child_counts or {}).items()}
    if any((not np.isfinite(v) or v < 0.0 or v != int(v)) for v in counts.values()):
        raise ValueError("invalid child-service counts")
    if n_calls < 0:
        raise ValueError("negative normal call count")
    if n_calls == 0:
        return result
    if not (0 <= both_role_count <= outbound_span_count <= n_calls):
        raise ValueError("invalid role counts: expected 0 <= both <= outbound <= calls")
    H = float(sum(counts.values()))
    outbound_share = float(outbound_span_count - both_role_count) / float(n_calls)
    if H <= 0.0 or outbound_share <= 0.0:
        return result
    transported = 0.0
    for dest, count in counts.items():
        if dest == source or dest not in services or count <= 0.0:
            continue
        mass = outbound_share * count / H
        result[services.index(dest)] += mass
        transported += mass
    result[source_idx] = 1.0 - transported
    if np.any(result < -EPS) or not np.isclose(float(np.sum(result)), 1.0, atol=1e-9):
        raise ValueError("destination row is not a nonnegative simplex vector")
    result[result < 0.0] = 0.0
    return result


def transport_energy(
    operations: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    energy_key: str = "q_median",
    identity: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Transport cached operation energies through shared responsibility rows."""
    x = np.zeros(len(candidates), dtype=np.float64)
    support = np.zeros(len(candidates), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for op in operations:
        source = str(op["source_service"])
        k = destination_kernel(
            source,
            candidates,
            op.get("child_counts"),
            int(op.get("n_calls", 0)),
            int(op.get("outbound_span_count", 0)),
            int(op.get("both_role_count", 0)),
            identity=identity,
        )
        energy = float(op.get(energy_key, 0.0))
        if not np.isfinite(energy) or energy < 0.0:
            raise ValueError("operation energy must be finite and nonnegative")
        x += energy * k
        support[k > 0.0] = 1.0
        rows.append({"source_service": source, "operation": str(op.get("operation", "")), "energy": energy, "kernel": k.tolist()})
    total = float(np.sum(x))
    t = x / total if total > EPS else np.zeros_like(x)
    return t, support, rows


def support_conditional_consensus(p: Sequence[float], t: Sequence[float], support: Sequence[float]) -> np.ndarray:
    """Mix only metric mass already in the structural trace support."""
    metric = np.asarray(p, dtype=np.float64).copy()
    trace = np.asarray(t, dtype=np.float64)
    mask = np.asarray(support, dtype=np.float64) > EPS
    if trace.size != metric.size or not np.any(mask):
        return metric / float(np.sum(metric))
    mass = float(np.sum(metric[mask]))
    if mass <= EPS or float(np.sum(trace)) <= EPS:
        return metric / float(np.sum(metric))
    out = metric.copy()
    out[mask] = 0.5 * (metric[mask] + mass * trace[mask])
    out[~mask] = metric[~mask]
    out = np.maximum(out, 0.0)
    return out / float(np.sum(out))


def score_incident(
    metric_normal: Sequence[Sequence[float]],
    metric_incident: Sequence[Sequence[float]],
    fine_names: Sequence[str],
    operations: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
) -> dict[str, Any]:
    """Compute all fixed arms from one shared incident representation."""
    p, w, service_mass = normal_calibrated_metric_field(metric_normal, metric_incident, fine_names, candidates)
    operation_rows: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    bins_by_source: dict[str, set[int]] = defaultdict(set)
    operation_counts: dict[str, int] = defaultdict(int)
    for op in operations:
        if "incident_bin_indices" not in op:
            raise ValueError("incident_bin_indices are required for service union coverage")
        bins_by_source[str(op["source_service"])].update(op["incident_bin_indices"])
        operation_counts[str(op["source_service"])] += 1
    for op in operations:
        stats = operation_persistence(
            op.get("normal_bins", []),
            op.get("incident_bins", []),
            operation_counts[str(op["source_service"])],
            len(bins_by_source[str(op["source_service"])]),
        )
        prepared.append({**dict(op), **stats})
    t_median, support, rows = transport_energy(prepared, candidates, energy_key="q_median")
    t_mean, _, _ = transport_energy(prepared, candidates, energy_key="q_mean")
    q_median = float(sum(float(op["q_median"]) for op in prepared))
    q_mean = float(sum(float(op["q_mean"]) for op in prepared))
    full = support_conditional_consensus(p, t_median, support) if q_median > EPS else p.copy()
    mean_full = support_conditional_consensus(p, t_mean, support) if q_mean > EPS else p.copy()
    identity_trace, identity_support, _ = transport_energy(prepared, candidates, energy_key="q_median", identity=True)
    identity = support_conditional_consensus(p, identity_trace, identity_support) if q_median > EPS else p.copy()
    identity_mean_trace, identity_mean_support, _ = transport_energy(prepared, candidates, energy_key="q_mean", identity=True)
    identity_mean = support_conditional_consensus(p, identity_mean_trace, identity_mean_support) if q_mean > EPS else p.copy()
    uniform = (w > 0.0).astype(np.float64)
    uniform = uniform / float(np.sum(uniform)) if np.any(uniform) else np.full(len(candidates), 1.0 / len(candidates))
    uniform_full = support_conditional_consensus(uniform, t_median, support) if q_median > EPS else uniform.copy()
    return {
        "metric_base": p,
        "metric_support": w,
        "metric_service_mass": service_mass,
        "operation_stats": prepared,
        "responsibility_rows": rows,
        "trace_median": t_median,
        "trace_mean": t_mean,
        "arms": {
            "L1_FULL": full,
            "MEAN_FULL": mean_full,
            "NO_TRACE": p,
            "NO_METRIC_VALUES": uniform_full,
            "IDENTITY_DEST": identity,
            "MEAN_IDENTITY_DEST": identity_mean,
        },
    }


def rank_scores(score: Sequence[float], candidates: Sequence[str]) -> list[str]:
    """Stable descending ranking; candidate order resolves exact ties."""
    values = np.asarray(score, dtype=np.float64)
    return [str(candidates[i]) for i in sorted(range(len(values)), key=lambda i: (-float(values[i]), i))]


def metric_probability(values: Sequence[float], weights: Sequence[float]) -> np.ndarray:
    """Normalize the metric field with the original supported-zero convention."""
    x = np.asarray(values, dtype=float).copy()
    w = np.asarray(weights, dtype=float) > 0.0
    if x.shape != w.shape or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("invalid metric field")
    if np.any(x < -EPS):
        raise ValueError("negative metric mass")
    x[np.abs(x) <= EPS] = 0.0
    x[~w] = 0.0
    if x.sum() > EPS:
        return x / x.sum()
    return w.astype(float) / w.sum() if w.any() else np.full(len(x), 1.0 / len(x))


def score_prepared(case: Mapping[str, Any]) -> dict[str, Any]:
    """Score normal-calibrated observations, without labels or stored scores.

    Each operation supplies incident bin indices/means, normal mean/scale,
    and a normal role object. None of these fields is an output score.
    """
    candidates = list(case["candidates"])
    operations = list(case["operations"])
    p = metric_probability(case["metric_x"], case["metric_W"])
    uniform = metric_probability(np.ones(len(p)), case["metric_W"])
    unions: dict[str, set[int]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for op in operations:
        unions[op["source_service"]].update(op["incident_bin_indices"])
        counts[op["source_service"]] += 1
    prepared = []
    for op in operations:
        service = op["source_service"]
        values = np.asarray(op["incident_bins"], dtype=float)
        if len(values) != len(op["incident_bin_indices"]) or not np.all(np.isfinite(values)):
            raise ValueError("invalid operation bins")
        scale = float(op["normal_scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("prepared normal scale must be positive")
        z = (values - float(op["normal_mean"])) / scale
        coverage = len(values) / len(unions[service]) if unions[service] else 0.0
        t_med = coverage * float(np.median(z)) if len(z) else 0.0
        t_mean = coverage * float(np.mean(z)) if len(z) else 0.0
        if not op["has_kernel_record"]:
            continue
        prepared.append({**op, "q_median": t_med**2 / counts[service],
                         "q_mean": t_mean**2 / counts[service]})
    output = {"NO_TRACE": p}
    for name, energy, identity in [("L1_FULL", "q_median", False),
                                   ("MEAN_FULL", "q_mean", False),
                                   ("IDENTITY_DEST", "q_median", True),
                                   ("MEAN_IDENTITY_DEST", "q_mean", True)]:
        t, support, _ = transport_energy(prepared, candidates, energy_key=energy, identity=identity)
        # Match the original post-aggregation zero-field convention.
        t[np.abs(t) <= EPS] = 0.0
        if t.sum() > EPS:
            t /= t.sum()
        output[name] = support_conditional_consensus(p, t, support)
        if name == "L1_FULL":
            output["NO_METRIC_VALUES"] = support_conditional_consensus(uniform, t, support)
    return {"arms": output, "operation_stats": prepared}


__all__ = [
    "EPS",
    "canonical_service",
    "preprocess_metric_columns",
    "normal_calibrated_metric_field",
    "operation_persistence",
    "destination_kernel",
    "transport_energy",
    "support_conditional_consensus",
    "score_incident",
    "rank_scores",
]
