"""Correct coverage stratification without rerunning the main curve.

The upstream condition rows already contain the frozen masks, scores, and
evaluation labels.  This script only reclassifies each row from that row's
own support-membership flags, averages seeds within incident, and then
aggregates incidents.  It never uses the first seed as a proxy for another
seed and never rebuilds P/W/K or any score.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PACKAGE=Path(__file__).resolve().parents[1]
UPSTREAM_ROOT=PACKAGE
INPUT_PATH=PACKAGE/'reproduced/coverage_condition_results.jsonl'
OUT_ROOT=PACKAGE/'reproduced/stratified'
RESULT_ROOT=OUT_ROOT
EXPECTED_ROWS = 2970
EXPECTED_EVENTS = 270
SEEDS = (17, 29, 43)
RETENTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("SUPPORT_CONDITIONAL", "GLOBAL_ARITHMETIC", "METRIC_ONLY")
METRICS = ("MRR", "AC@1", "AC@3", "Avg@5", "AC@5")
EFFECTS = (
    ("SUPPORT_CONDITIONAL", "GLOBAL_ARITHMETIC", "support_minus_global"),
    ("SUPPORT_CONDITIONAL", "METRIC_ONLY", "support_minus_metric_only"),
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260910


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def as_float(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric value: {value}")
    return value


def retention_key(value: Any) -> float:
    return round(float(value), 8)


def metric_from(row: Mapping[str, Any], arm: str, metric: str) -> float:
    values = row["arms"][arm]["metrics"]["expected_tie_aware"]
    key = "list_rank_mrr" if metric == "MRR" else metric
    return as_float(values[key])


def rank_from(row: Mapping[str, Any], arm: str) -> float:
    return as_float(row["arms"][arm]["metrics"]["ordinary_name_rank"]["list_rank"])


def membership_layers(row: Mapping[str, Any]) -> list[tuple[str, bool]]:
    # The two layers intentionally overlap, matching the historical table:
    # each layer answers a separate support-membership question.
    return [
        (
            "original_support_in"
            if bool(row["root_in_original_support"])
            else "original_support_out",
            bool(row["root_in_original_support"]),
        ),
        (
            "retained_support_in"
            if bool(row["root_in_retained_support"])
            else "retained_support_out",
            bool(row["root_in_retained_support"]),
        ),
    ]


def condition_projection(row: Mapping[str, Any], condition_index: int, layer: str, membership: bool) -> dict[str, Any]:
    labels = row.get("labels") or {}
    out: dict[str, Any] = {
        "condition_index": condition_index,
        "case_id": row["case_id"],
        "system": row["system"],
        "panel": row["panel"],
        "window_length_minutes": row["window_length_minutes"],
        "retention": retention_key(row["retention"]),
        "seed": int(row["seed"]),
        "layer": layer,
        "layer_membership": bool(membership),
        "root_in_original_support": bool(row["root_in_original_support"]),
        "root_in_retained_support": bool(row["root_in_retained_support"]),
        "root_service": labels.get("root_service"),
        "fault_type": labels.get("fault_type"),
        "scenario_id": labels.get("scenario_id"),
        "replicate": labels.get("replicate"),
        "labels_loaded_in_source_row": bool(row.get("labels_loaded", False)),
        "mask_construction_label_free": True,
    }
    for arm in ARMS:
        for metric in METRICS:
            out[f"{arm}__{metric}"] = metric_from(row, arm, metric)
        out[f"{arm}__rank"] = rank_from(row, arm)
    out["support_minus_global__rank"] = out["SUPPORT_CONDITIONAL__rank"] - out["GLOBAL_ARITHMETIC__rank"]
    out["support_minus_metric_only__rank"] = out["SUPPORT_CONDITIONAL__rank"] - out["METRIC_ONLY__rank"]
    return out


def mean(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=float))) if values else None


def summarize_event_layer(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    seed_values = sorted({int(row["seed"]) for row in rows})
    retention = retention_key(first["retention"])
    expected_seed_count = 1 if retention in (0.0, 1.0) else 3
    out: dict[str, Any] = {
        "case_id": first["case_id"],
        "system": first["system"],
        "panel": first["panel"],
        "window_length_minutes": first["window_length_minutes"],
        "retention": retention,
        "layer": first["layer"],
        "root_service": first.get("root_service"),
        "fault_type": first.get("fault_type"),
        "scenario_id": first.get("scenario_id"),
        "replicate": first.get("replicate"),
        "condition_count": len(rows),
        "seed_count": len(seed_values),
        "seed_values": seed_values,
        "expected_seed_count": expected_seed_count,
        "seed_proportion": len(seed_values) / expected_seed_count,
        "mask_construction_label_free": all(bool(row["mask_construction_label_free"]) for row in rows),
    }
    for arm in ARMS:
        for metric in METRICS:
            out[f"{arm}__{metric}"] = mean([float(row[f"{arm}__{metric}"]) for row in rows])
        out[f"{arm}__rank"] = mean([float(row[f"{arm}__rank"]) for row in rows])
    for metric in METRICS:
        out[f"support_minus_global__{metric}"] = out[f"SUPPORT_CONDITIONAL__{metric}"] - out[f"GLOBAL_ARITHMETIC__{metric}"]
        out[f"support_minus_metric_only__{metric}"] = out[f"SUPPORT_CONDITIONAL__{metric}"] - out[f"METRIC_ONLY__{metric}"]
    out["support_minus_global__rank"] = out["SUPPORT_CONDITIONAL__rank"] - out["GLOBAL_ARITHMETIC__rank"]
    out["support_minus_metric_only__rank"] = out["SUPPORT_CONDITIONAL__rank"] - out["METRIC_ONLY__rank"]
    for effect_key in ("support_minus_global", "support_minus_metric_only"):
        vals = [float(row[f"{effect_key}__rank"]) for row in rows]
        out[f"{effect_key}__rank_wins"] = int(sum(value < 0.0 for value in vals))
        out[f"{effect_key}__rank_ties"] = int(sum(value == 0.0 for value in vals))
        out[f"{effect_key}__rank_losses"] = int(sum(value > 0.0 for value in vals))
    return out


def aggregate_summary(event_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[tuple[str, list[Mapping[str, Any]]]] = [("ALL", event_rows)]
    for system in sorted({str(row["system"]) for row in event_rows}):
        scopes.append((system, [row for row in event_rows if str(row["system"]) == system]))
    rows: list[dict[str, Any]] = []
    for scope, scope_rows in scopes:
        for retention in RETENTIONS:
            for layer in sorted({str(row["layer"]) for row in event_rows}):
                subset = [
                    row
                    for row in scope_rows
                    if retention_key(row["retention"]) == retention and row["layer"] == layer
                ]
                if not subset:
                    continue
                expected_seed_count = 1 if retention in (0.0, 1.0) else 3
                out: dict[str, Any] = {
                    "scope": scope,
                    "retention": retention,
                    "layer": layer,
                    "n_event_layer_units": len(subset),
                    "n_events": len({row["case_id"] for row in subset}),
                    "n_conditions": int(sum(int(row["condition_count"]) for row in subset)),
                    "seed_count_total": int(sum(int(row["seed_count"]) for row in subset)),
                    "expected_seed_count_per_event": expected_seed_count,
                    "mean_seed_proportion": mean([float(row["seed_proportion"]) for row in subset]),
                    "min_seed_proportion": min(float(row["seed_proportion"]) for row in subset),
                    "max_seed_proportion": max(float(row["seed_proportion"]) for row in subset),
                }
                for arm in ARMS:
                    for metric in METRICS:
                        out[f"{arm}__{metric}"] = mean([float(row[f"{arm}__{metric}"]) for row in subset])
                    out[f"{arm}__rank"] = mean([float(row[f"{arm}__rank"]) for row in subset])
                for effect_key in ("support_minus_global", "support_minus_metric_only"):
                    for metric in METRICS:
                        out[f"{effect_key}__{metric}"] = mean([float(row[f"{effect_key}__{metric}"]) for row in subset])
                    out[f"{effect_key}__rank"] = mean([float(row[f"{effect_key}__rank"]) for row in subset])
                    out[f"{effect_key}__rank_wins"] = int(sum(int(row[f"{effect_key}__rank_wins"]) for row in subset))
                    out[f"{effect_key}__rank_ties"] = int(sum(int(row[f"{effect_key}__rank_ties"]) for row in subset))
                    out[f"{effect_key}__rank_losses"] = int(sum(int(row[f"{effect_key}__rank_losses"]) for row in subset))
                rows.append(out)
    return rows


def cluster_bootstrap(event_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scopes: list[tuple[str, list[Mapping[str, Any]]]] = [("ALL", event_rows)]
    for system in sorted({str(row["system"]) for row in event_rows}):
        scopes.append((system, [row for row in event_rows if str(row["system"]) == system]))
    output: list[dict[str, Any]] = []
    for scope, scope_rows in scopes:
        for retention in RETENTIONS:
            for layer in sorted({str(row["layer"]) for row in event_rows}):
                subset = [
                    row
                    for row in scope_rows
                    if retention_key(row["retention"]) == retention and row["layer"] == layer
                ]
                if not subset:
                    continue
                clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in subset:
                    cluster = f"{row.get('root_service')}::{row.get('fault_type')}"
                    clusters[cluster].append(row)
                cluster_names = sorted(clusters)
                for left, right, effect_name in EFFECTS:
                    for metric in METRICS:
                        effect_key = f"{effect_name}__{metric}"
                        cluster_values = np.asarray(
                            [
                                np.mean(
                                    [
                                        float(row[f"{left}__{metric}"]) - float(row[f"{right}__{metric}"])
                                        for row in clusters[name]
                                    ]
                                )
                                for name in cluster_names
                            ],
                            dtype=float,
                        )
                        estimate = float(np.mean(cluster_values))
                        if len(cluster_values) == 1:
                            boot = np.full(BOOTSTRAP_REPLICATES, estimate, dtype=float)
                        else:
                            draws = rng.integers(0, len(cluster_values), size=(BOOTSTRAP_REPLICATES, len(cluster_values)))
                            boot = cluster_values[draws].mean(axis=1)
                        output.append(
                            {
                                "scope": scope,
                                "retention": retention,
                                "layer": layer,
                                "effect": effect_name,
                                "metric": metric,
                                "estimate": estimate,
                                "ci95_low": float(np.percentile(boot, 2.5)),
                                "ci95_high": float(np.percentile(boot, 97.5)),
                                "n_event_layer_units": len(subset),
                                "n_clusters": len(cluster_values),
                                "cluster_unit": "root_service::fault_type",
                                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                                "bootstrap_seed": BOOTSTRAP_SEED,
                                "seed_rows_nested_within_event": True,
                            }
                        )
    return output


def rank_effect_rows(event_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in event_rows:
        out = {
            "case_id": row["case_id"],
            "system": row["system"],
            "retention": row["retention"],
            "layer": row["layer"],
            "root_service": row.get("root_service"),
            "fault_type": row.get("fault_type"),
            "seed_count": row["seed_count"],
            "seed_proportion": row["seed_proportion"],
        }
        for effect in ("support_minus_global", "support_minus_metric_only"):
            out[f"{effect}_rank_delta"] = row[f"{effect}__rank"]
            out[f"{effect}_rank_wins"] = row[f"{effect}__rank_wins"]
            out[f"{effect}_rank_ties"] = row[f"{effect}__rank_ties"]
            out[f"{effect}_rank_losses"] = row[f"{effect}__rank_losses"]
        rows.append(out)
    return rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    source_rows = [json.loads(line) for line in INPUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(source_rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} upstream condition rows, got {len(source_rows)}")
    if len({row["case_id"] for row in source_rows}) != EXPECTED_EVENTS:
        raise RuntimeError("upstream incident count is not 270")
    if any(row.get("method") != "TopoTrace-AD coverage mismatch" for row in source_rows):
        raise RuntimeError("unexpected upstream method identity")

    condition_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        for layer, membership in membership_layers(row):
            condition_rows.append(condition_projection(row, index, layer, membership))

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        grouped[
            (
                row["system"],
                row["panel"],
                row["case_id"],
                retention_key(row["retention"]),
                row["layer"],
            )
        ].append(row)
    event_rows = [summarize_event_layer(rows) for _, rows in sorted(grouped.items(), key=lambda item: str(item[0]))]
    summary_rows = aggregate_summary(event_rows)
    bootstrap_rows = cluster_bootstrap(event_rows)

    write_jsonl(RESULT_ROOT / "coverage_strata_condition_results.jsonl", condition_rows)
    write_jsonl(RESULT_ROOT / "coverage_strata_incident_averages.jsonl", event_rows)
    write_csv(RESULT_ROOT / "coverage_strata_summary.csv", summary_rows)
    write_csv(RESULT_ROOT / "coverage_strata_event_seed_audit.csv", [
        {
            "case_id": row["case_id"],
            "system": row["system"],
            "retention": row["retention"],
            "layer": row["layer"],
            "condition_count": row["condition_count"],
            "seed_count": row["seed_count"],
            "seed_values": ",".join(str(value) for value in row["seed_values"]),
            "expected_seed_count": row["expected_seed_count"],
            "seed_proportion": row["seed_proportion"],
        }
        for row in event_rows
    ])
    write_csv(RESULT_ROOT / "coverage_strata_rank_effects.csv", rank_effect_rows(event_rows))
    write_json(RESULT_ROOT / "coverage_strata_paired_bootstrap.json", {
        "method": "root-service x fault-type cluster bootstrap over event-layer units",
        "rows": bootstrap_rows,
    })

    seed_counts = Counter((retention_key(row["retention"]), int(row["seed"])) for row in source_rows)
    properties = {
        "status": "COMPLETED",
        "method": "TOPOTRACE-WEB-COVERAGE-STRATIFICATION-CORRECTED",
        "upstream_method": "TopoTrace-AD coverage mismatch",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "input_condition_rows": len(source_rows),
        "expected_input_condition_rows": EXPECTED_ROWS,
        "output_condition_rows": EXPECTED_ROWS,
        "output_expanded_layer_rows": len(condition_rows),
        "output_event_layer_rows": len(event_rows),
        "input_event_count": len({row["case_id"] for row in source_rows}),
        "event_layer_definition": "event x retention x layer after averaging all condition rows/seeds for that incident",
        "layers": ["original_support_in/out", "retained_support_in/out"],
        "classification_source": "each condition row's own root_in_original_support and root_in_retained_support",
        "first_seed_classification_used": False,
        "labels_read_for_evaluation_context_only": True,
        "mask_construction_label_free": True,
        "main_curve_rerun": False,
        "upstream_main_curve_preserved": True,
        "seed_values": list(SEEDS),
        "seed_counts_by_retention_and_seed": {f"{retention:g}__{seed}": count for (retention, seed), count in sorted(seed_counts.items())},
        "expected_seed_count": {"0.0": 1, "0.25": 3, "0.5": 3, "0.75": 3, "1.0": 1},
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "root_service::fault_type",
            "seed_rows_nested_within_event": True,
        },
    }
    write_json(RESULT_ROOT / "coverage_strata_properties.json", properties)

    preserved_files: list[dict[str, Any]] = []
    for name in ("coverage_curve.csv", "coverage_properties.json", "coverage_prelabel_seal.json"):
        source = UPSTREAM_ROOT / "results" / name
        destination = RESULT_ROOT / f"upstream_{name}"
        if source.exists():
            shutil.copyfile(source, destination)
            preserved_files.append(
                {
                    "source": str(source),
                    "destination": str(destination),
                    "bytes": int(destination.stat().st_size),
                    "sha256": sha256(destination),
                }
            )
    write_json(
        RESULT_ROOT / "upstream_main_curve_provenance.json",
        {
            "main_curve_rerun": False,
            "preserved_upstream_files": preserved_files,
            "note": "Copied as provenance only; corrected stratification is computed from the frozen condition rows.",
        },
    )

    report = [
        "# Corrected coverage stratification",
        "",
        "This artifact re-aggregates the frozen 2,970 condition rows. It does not rerun the main coverage curve and does not rebuild masks, scores, or P/W/K.",
        "",
        f"- Input: `{len(source_rows)}` condition rows, `{properties['input_event_count']}` incidents, SHA-256 `{properties['input_sha256']}`.",
        f"- Output: `{len(condition_rows)}` explicit layer rows and `{len(event_rows)}` incident × retention × layer averages.",
        "- Classification uses each condition row's own `root_in_original_support` and `root_in_retained_support`; first-seed classification is not used.",
        "- Labels are retained only as post-construction evaluation context; they do not construct the frozen masks.",
        "- Endpoint retentions have one seed; partial retentions have three seeds. Seed proportions are preserved in the event audit.",
        "- Bootstrap unit is `root_service::fault_type`, with seeds nested within incident; 10,000 replicates and seed 20260910.",
        "",
        "The original main curve remains an upstream valid result and is copied only by provenance. The corrected layer tables are the new stratified evidence.",
        "",
        "## Outputs",
        "",
        "- `results/coverage_strata_condition_results.jsonl`",
        "- `results/coverage_strata_incident_averages.jsonl`",
        "- `results/coverage_strata_summary.csv`",
        "- `results/coverage_strata_event_seed_audit.csv`",
        "- `results/coverage_strata_rank_effects.csv`",
        "- `results/coverage_strata_paired_bootstrap.json`",
        "- `results/coverage_strata_properties.json`",
    ]
    (OUT_ROOT / "scientific_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETED", "condition_rows": len(condition_rows), "event_layer_rows": len(event_rows), "summary_rows": len(summary_rows), "bootstrap_rows": len(bootstrap_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
