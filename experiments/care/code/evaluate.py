"""Evaluation utilities for the supplied service-level ranking records."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def rank_metrics(ranking: Sequence[str], root: str, candidate_count: int | None = None) -> dict[str, float | int | None]:
    """Compute conventional metrics for one ordered candidate list."""
    name = str(root)
    try:
        position = list(ranking).index(name) + 1
    except ValueError:
        position = None
    n = int(candidate_count or len(ranking))
    if position is None:
        rr = 0.0
        return {
            "candidate_size": n,
            "root_rank": None,
            "MRR": rr,
            "AC@1": 0.0,
            "AC@3": 0.0,
            "Avg@5": 0.0,
            "AC@5": 0.0,
        }
    return {
        "candidate_size": n,
        "root_rank": position,
        "MRR": 1.0 / position,
        "AC@1": float(position <= 1),
        "AC@3": float(position <= 3),
        "Avg@5": max(0.0, (5.0 - position + 1.0) / 5.0),
        "AC@5": float(position <= 5),
    }


def aggregate(rows: Iterable[Mapping[str, object]], keys: Sequence[str]) -> list[dict[str, object]]:
    """Average numeric ranking fields by the requested independent-unit keys."""
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    metrics = ("AC@1", "AC@3", "Avg@5", "MRR", "AC@5")
    output: list[dict[str, object]] = []
    for key, group in groups.items():
        item = {k: v for k, v in zip(keys, key)}
        item["n"] = len(group)
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            item[metric] = sum(values) / len(values) if values else None
        output.append(item)
    return sorted(output, key=lambda x: tuple(str(x.get(k, "")) for k in keys))


def score_sum(score: Sequence[float]) -> float:
    return float(sum(float(x) for x in score))


def expected_metrics(scores: Sequence[float], candidates: Sequence[str], root: str,
                     tolerance: float = 1e-12) -> dict[str, float]:
    """Uniform tie-breaking expectation; average reciprocals, not reciprocal average rank."""
    if len(scores) != len(candidates) or len(set(candidates)) != len(candidates):
        raise ValueError("Scores require one value per unique candidate")
    if root not in candidates:
        return {k: 0.0 for k in ("AC@1", "AC@3", "AC@5", "Avg@5", "list_rank_mrr")}
    value = float(scores[list(candidates).index(root)])
    higher = sum(float(v) > value + tolerance for v in scores)
    equal = sum(abs(float(v) - value) <= tolerance for v in scores)
    probabilities = {k: min(1.0, max(0.0, (k-higher)/equal)) for k in range(1, 6)}
    return {"AC@1": probabilities[1], "AC@3": probabilities[3], "AC@5": probabilities[5],
            "Avg@5": sum(probabilities.values())/5,
            "list_rank_mrr": sum(1/(higher+j) for j in range(1, equal+1))/equal}


__all__ = ["rank_metrics", "expected_metrics", "aggregate", "score_sum"]
