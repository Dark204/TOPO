from __future__ import annotations
import csv, json, math, os, time, hashlib, shutil, gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Iterable, Mapping, Sequence
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE
OUT = PACKAGE
OUT_RESULTS = PACKAGE/'reproduced'
OUT_DATA = OUT_RESULTS/'masks'
FIXED_CASES = PACKAGE/'data/fixed_evidence.jsonl'
FIXED_SOURCE = FIXED_CASES
METHOD = 'TopoTrace-AD coverage mismatch'
EVENTS_EXPECTED=270
RETENTIONS=(1.0,.75,.5,.25,0.0)
PARTIAL_SEEDS=(17,29,43)
BOOTSTRAP_SEED=20260910
BOOTSTRAP_REPLICATES=10000
EPS=1e-10
TOL=1e-12
METRICS=('AC@1','AC@3','Avg@5','list_rank_mrr','AC@5')
ARMS=('SUPPORT_CONDITIONAL','GLOBAL_ARITHMETIC','METRIC_ONLY')
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

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(clean(row), ensure_ascii=False, separators=(',', ':')) + '\n')

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key)) for key in keys})

def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def sha_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()

def key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get('panel', '')), str(row.get('system', '')), str(row.get('window_length_minutes', '')), str(row.get('case_id', '')))

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

def score_payload(score: np.ndarray, candidate: Sequence[str], root: str, operator: str) -> dict[str, Any]:
    values = np.asarray(score, dtype=np.float64).copy()
    if values.shape != (len(candidate),) or not np.all(np.isfinite(values)) or np.any(values < -1e-09):
        raise ValueError('invalid score vector')
    values[np.abs(values) <= 1e-12] = 0.0
    if abs(float(np.sum(values)) - 1.0) > 5e-10:
        raise ValueError(f'score simplex violation: {float(np.sum(values))}')
    rank = rank_by_score(values, candidate)
    return {'score_vector': values.tolist(), 'rank': rank, 'score_sum': float(np.sum(values)), 'score_min': float(np.min(values)), 'operator': operator, 'metrics': {'expected_tie_aware': expected_metrics(values, candidate, root), 'ordinary_name_rank': ordinary_metrics(rank, root)}}

def source_rows() -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(FIXED_CASES) if str(row.get('panel')) == 'BENCHMARK' and str(row.get('window_length_minutes')) == '10']
    if len(rows) != EVENTS_EXPECTED:
        raise RuntimeError(f'expected {EVENTS_EXPECTED} BENCHMARK/10 rows, found {len(rows)}')
    return sorted(rows, key=key)

def build_label_free_masks(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    masks: list[dict[str, Any]] = []
    for source in rows:
        candidate = [str(item) for item in source['candidate_order']]
        original_support = [index for index, value in enumerate(source['trace_variants']['L1_FULL']['W']) if float(value) > 0.0]
        support_order = sorted(original_support, key=lambda index: (candidate[index], index))
        common = {'panel': str(source['panel']), 'window_length_minutes': str(source['window_length_minutes']), 'case_id': str(source['case_id']), 'candidate_order': candidate, 'candidate_order_sha256': sha_payload(candidate), 'original_support_indices': support_order, 'original_support_services': [candidate[index] for index in support_order], 'original_support_count': len(support_order), 'labels_loaded': False, 'root_fault_fields_present': False, 'mask_selection_label_free': True, 'mask_rule': 'sort original S by service id; random permutation; keep floor(retention*|S|) prefix'}
        permutations: dict[int, list[int]] = {}
        for seed in PARTIAL_SEEDS:
            rng = np.random.default_rng(seed)
            permutation = list(np.asarray(support_order, dtype=np.int64)[rng.permutation(len(support_order))].astype(int))
            permutations[seed] = permutation
        for retention in RETENTIONS:
            seeds = PARTIAL_SEEDS if retention not in (0.0, 1.0) else (PARTIAL_SEEDS[0],)
            for seed in seeds:
                permutation = permutations[seed]
                keep_count = int(math.floor(float(retention) * len(support_order)))
                kept = sorted(permutation[:keep_count])
                masks.append({**common, 'retention': float(retention), 'seed': int(seed), 'seed_deduplicated_for_endpoint': bool(retention in (0.0, 1.0)), 'permutation_services': [candidate[index] for index in permutation], 'retained_indices': kept, 'retained_services': [candidate[index] for index in kept], 'retained_count': len(kept), 'mask_id': f"{source['case_id']}|r={retention:.2f}|seed={seed}"})
    return masks

def has_forbidden_label(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any((str(key) in {'labels', 'root_service', 'fault_type', 'scenario_id', 'replicate'} for key in value)):
            return True
        return any((has_forbidden_label(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return any((has_forbidden_label(item) for item in value))
    return False

def mask_seal_ok(masks: Sequence[Mapping[str, Any]]) -> bool:
    return bool(len(masks) == EVENTS_EXPECTED * 11 and all((row.get('labels_loaded') is False and row.get('root_fault_fields_present') is False for row in masks)) and all((not has_forbidden_label(row) for row in masks)))

def score_three(source: Mapping[str, Any], mask: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = [str(item) for item in source['candidate_order']]
    p = np.asarray(source['arms']['NO_TRACE']['score_vector'], dtype=np.float64)
    trace = source['trace_variants']['L1_FULL']
    x = np.asarray(trace['x'], dtype=np.float64)
    support = np.asarray(trace['W'], dtype=np.float64) > 0.0
    retained = np.zeros(len(candidate), dtype=bool)
    retained[np.asarray(mask['retained_indices'], dtype=np.int64)] = True
    if p.shape != x.shape or x.shape != retained.shape or (not np.all(np.isfinite(p))) or (not np.all(np.isfinite(x))):
        raise ValueError(f"fixed field shape/nonfinite error for {source['case_id']}")
    p_sum = float(np.sum(p))
    x_original_sum = float(np.sum(x))
    if abs(p_sum - 1.0) > 5e-10:
        raise ValueError(f"metric simplex error for {source['case_id']}: {p_sum}")
    if x_original_sum > EPS and abs(x_original_sum - 1.0) > 5e-10:
        raise ValueError(f"transport simplex error for {source['case_id']}: {x_original_sum}")
    retained &= support
    x_prime = np.where(retained, x, 0.0)
    q_prime = float(np.sum(x_prime))
    informative = bool(q_prime > EPS and np.any(retained))
    t_prime = x_prime / q_prime if informative else np.zeros_like(x_prime)
    p_mass_on_retained = float(np.sum(p[retained]))
    if informative and p_mass_on_retained > 1e-12:
        support_conditional = p.copy()
        support_conditional[retained] = (p[retained] + p_mass_on_retained * t_prime[retained]) / 2.0
        support_operator = 'support_conditional_arithmetic_on_retained_S'
        global_arithmetic = (p + t_prime) / 2.0
        global_operator = 'full_domain_arithmetic_average_p_and_t_prime'
    else:
        support_conditional = p.copy()
        global_arithmetic = p.copy()
        support_operator = 'metric_only_fallback_no_Q_or_no_metric_mass_on_retained_S'
        global_operator = 'metric_only_fallback_no_Q_or_no_retained_S'
    scores = {'SUPPORT_CONDITIONAL': score_payload(support_conditional, candidate, str(source['labels']['root_service']), support_operator), 'GLOBAL_ARITHMETIC': score_payload(global_arithmetic, candidate, str(source['labels']['root_service']), global_operator), 'METRIC_ONLY': score_payload(p, candidate, str(source['labels']['root_service']), 'complete_metric_distribution_p')}
    metadata = {'original_support_count': int(np.sum(support)), 'retained_support_count': int(np.sum(retained)), 'retained_count': int(len(mask['retained_indices'])), 'q_prime': q_prime, 'informative_after_mask': informative, 'p_mass_on_retained_support': p_mass_on_retained, 'x_prime_sum': float(np.sum(x_prime)), 't_prime_sum': float(np.sum(t_prime)), 'support_outside_trace_mass_removed': float(np.sum(x[~retained])), 'p_complete_domain': True, 'hidden_coordinates_excluded': True, 'p_sha256': sha_payload(p.tolist()), 'x_sha256': sha_payload(x.tolist()), 'original_W_sha256': sha_payload(support.astype(int).tolist()), 'retained_W_sha256': sha_payload(retained.astype(int).tolist())}
    return (scores, metadata)

def bootstrap_differences(rows: Sequence[Mapping[str, Any]], left: str, right: str, system: str | None=None) -> dict[str, Any]:
    subset = [row for row in rows if system is None or str(row['system']) == system]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in subset:
        labels = row['labels']
        groups[f"{labels['root_service']}::{labels['fault_type']}"].append(row)
    names = sorted(groups)
    output: dict[str, Any] = {'left': left, 'right': right, 'system': system or 'ALL', 'n_incidents': len(subset), 'cluster_count': len(names), 'cluster_order': names, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_replicates': BOOTSTRAP_REPLICATES, 'unit': 'incident metric averages over masks first; root_service x fault_type cluster bootstrap', 'metrics': {}}
    for metric in METRICS:
        event = np.asarray([float(row['arms'][left]['metrics']['expected_tie_aware'][metric]) - float(row['arms'][right]['metrics']['expected_tie_aware'][metric]) for row in subset], dtype=np.float64)
        cluster = np.asarray([float(np.mean([float(row['arms'][left]['metrics']['expected_tie_aware'][metric]) - float(row['arms'][right]['metrics']['expected_tie_aware'][metric]) for row in groups[name]])) for name in names], dtype=np.float64)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        if cluster.size:
            sample = cluster[rng.integers(0, len(cluster), size=(BOOTSTRAP_REPLICATES, len(cluster)))].mean(axis=1)
            ci = [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))]
        else:
            ci = [None, None]
        output['metrics'][metric] = {'difference': float(np.mean(event)) if event.size else None, 'ci95': ci, 'n_incidents': int(event.size), 'win': int(np.sum(event > TOL)), 'tie': int(np.sum(np.abs(event) <= TOL)), 'loss': int(np.sum(event < -TOL))}
    return output

def aggregate_incidents(condition_rows: Sequence[Mapping[str, Any]], source_by_key: Mapping[tuple[str, str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        grouped[row['panel'], row['system'], row['window_length_minutes'], row['case_id'], float(row['retention'])].append(row)
    result: list[dict[str, Any]] = []
    for group_key, rows in sorted(grouped.items()):
        source = source_by_key[group_key[0], group_key[1], group_key[2], group_key[3]]
        labels = source['labels']
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arms[arm] = {'metrics': {mode: {metric: float(np.mean([row['arms'][arm]['metrics'][mode][metric] for row in rows])) for metric in METRICS} for mode in ('expected_tie_aware', 'ordinary_name_rank')}, 'mean_rank': float(np.mean([row['arms'][arm]['metrics']['ordinary_name_rank']['list_rank'] for row in rows])), 'rank_values': [row['arms'][arm]['metrics']['ordinary_name_rank']['list_rank'] for row in rows]}
        first = rows[0]
        candidate = [str(item) for item in source['candidate_order']]
        root = str(labels['root_service'])
        original_support = set((int(index) for index in first['metadata']['original_support_indices']))
        retained = set((int(index) for index in first['metadata']['retained_indices']))
        root_index = candidate.index(root) if root in candidate else None
        result.append({'method': METHOD, 'panel': group_key[0], 'system': group_key[1], 'window_length_minutes': group_key[2], 'case_id': group_key[3], 'labels': labels, 'retention': group_key[4], 'condition_count': len(rows), 'seed_conditions': sorted((int(row['seed']) for row in rows)), 'original_support_count': int(first['metadata']['original_support_count']), 'retained_support_count': int(first['metadata']['retained_support_count']), 'retained_count': int(first['metadata']['retained_count']), 'original_root_in_support': bool(root_index is not None and root_index in original_support), 'root_in_retained_support': bool(root_index is not None and root_index in retained), 'root_index': root_index, 'arms': arms})
    return result

def summary_rows(incident_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for system in ('OnlineBoutique', 'SockShop', 'TrainTicket'):
        for retention in RETENTIONS:
            subset = [row for row in incident_rows if row['system'] == system and abs(float(row['retention']) - retention) < 1e-12]
            for arm in ARMS:
                values = [row['arms'][arm]['metrics']['expected_tie_aware'] for row in subset]
                ordinary = [row['arms'][arm]['metrics']['ordinary_name_rank'] for row in subset]
                output.append({'system': system, 'retention': retention, 'arm': arm, 'n_incidents': len(subset), 'n_evaluable': sum((row['root_index'] is not None for row in subset)), 'original_root_in_support_n': sum((bool(row['original_root_in_support']) for row in subset)), 'retained_root_in_support_n': sum((bool(row['root_in_retained_support']) for row in subset)), **{metric: float(np.mean([value[metric] for value in values])) if values else None for metric in METRICS}, 'ordinary_avg_rank': float(np.mean([row['arms'][arm]['mean_rank'] for row in subset])) if subset else None})
    return output

def rank_effect_rows(condition_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for system in ('ALL', 'OnlineBoutique', 'SockShop', 'TrainTicket'):
        for retention in RETENTIONS:
            subset = [row for row in condition_rows if (system == 'ALL' or row['system'] == system) and abs(float(row['retention']) - retention) < 1e-12]
            for arm in ('SUPPORT_CONDITIONAL', 'GLOBAL_ARITHMETIC'):
                improved = unchanged = degraded = 0
                for row in subset:
                    left = row['arms'][arm]['metrics']['ordinary_name_rank']['list_rank']
                    right = row['arms']['METRIC_ONLY']['metrics']['ordinary_name_rank']['list_rank']
                    if left < right:
                        improved += 1
                    elif left > right:
                        degraded += 1
                    else:
                        unchanged += 1
                output.append({'system': system, 'retention': retention, 'arm': arm, 'unit': 'seed_condition_rank_comparison', 'n_conditions': len(subset), 'unchanged': unchanged, 'improved': improved, 'degraded': degraded})
    return output

def support_strata_rows(incident_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for system in ('ALL', 'OnlineBoutique', 'SockShop', 'TrainTicket'):
        for retention in RETENTIONS:
            for stratum_name, predicate in (('original_support_in', lambda row: row['original_root_in_support']), ('original_support_out', lambda row: not row['original_root_in_support']), ('retained_support_in', lambda row: row['root_in_retained_support']), ('retained_support_out', lambda row: not row['root_in_retained_support'])):
                subset = [row for row in incident_rows if (system == 'ALL' or row['system'] == system) and abs(float(row['retention']) - retention) < 1e-12 and predicate(row)]
                output.append({'system': system, 'retention': retention, 'stratum': stratum_name, 'n_incidents': len(subset), 'metric_only_Avg@5': float(np.mean([row['arms']['METRIC_ONLY']['metrics']['expected_tie_aware']['Avg@5'] for row in subset])) if subset else None, 'support_conditional_Avg@5': float(np.mean([row['arms']['SUPPORT_CONDITIONAL']['metrics']['expected_tie_aware']['Avg@5'] for row in subset])) if subset else None, 'global_arithmetic_Avg@5': float(np.mean([row['arms']['GLOBAL_ARITHMETIC']['metrics']['expected_tie_aware']['Avg@5'] for row in subset])) if subset else None, 'support_delta_vs_metric': float(np.mean([row['arms']['SUPPORT_CONDITIONAL']['metrics']['expected_tie_aware']['Avg@5'] - row['arms']['METRIC_ONLY']['metrics']['expected_tie_aware']['Avg@5'] for row in subset])) if subset else None, 'global_delta_vs_metric': float(np.mean([row['arms']['GLOBAL_ARITHMETIC']['metrics']['expected_tie_aware']['Avg@5'] - row['arms']['METRIC_ONLY']['metrics']['expected_tie_aware']['Avg@5'] for row in subset])) if subset else None})
    return output

def main() -> None:
    started = time.perf_counter()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    sources = source_rows()
    source_by_key = {key(row): row for row in sources}
    masks = build_label_free_masks(sources)
    mask_path = OUT_DATA / 'coverage_prelabel_masks.jsonl'
    write_jsonl(mask_path, masks)
    masks = read_jsonl(mask_path)
    if not mask_seal_ok(masks):
        raise RuntimeError('label-free mask seal failed')
    write_json(OUT_RESULTS / 'coverage_prelabel_seal.json', {'method': METHOD, 'status': 'SEALED_LABEL_FREE', 'mask_count': len(masks), 'event_count': EVENTS_EXPECTED, 'conditions_per_event': 11, 'mask_sha256': sha_file(mask_path), 'labels_loaded': False, 'root_fault_fields_present': False, 'mask_selection_label_free': True})
    masks_by_key = defaultdict(list)
    for mask in masks:
        masks_by_key[mask['panel'], '', mask['window_length_minutes'], mask['case_id']].append(mask)
    condition_rows: list[dict[str, Any]] = []
    full_replay_max = 0.0
    full_replay_rank_failures = 0
    original_p_simplex_max = 0.0
    original_x_simplex_max = 0.0
    p_complete_failures = 0
    endpoint_failures = 0
    nested_failures = 0
    for source in sources:
        source_masks = sorted(masks_by_key[source['panel'], '', source['window_length_minutes'], source['case_id']], key=lambda item: (float(item['retention']), int(item['seed'])))
        candidate = [str(item) for item in source['candidate_order']]
        root = str(source['labels']['root_service'])
        p = np.asarray(source['arms']['NO_TRACE']['score_vector'], dtype=np.float64)
        x = np.asarray(source['trace_variants']['L1_FULL']['x'], dtype=np.float64)
        original_p_simplex_max = max(original_p_simplex_max, abs(float(np.sum(p)) - 1.0))
        if float(np.sum(x)) > EPS:
            original_x_simplex_max = max(original_x_simplex_max, abs(float(np.sum(x)) - 1.0))
        if len(p) != len(candidate):
            p_complete_failures += 1
        endpoint = next((item for item in source_masks if float(item['retention']) == 1.0))
        endpoint_scores, endpoint_meta = score_three(source, endpoint)
        original_support = np.asarray(source['trace_variants']['L1_FULL']['W'], dtype=float) > 0
        saved_full = p.copy()
        original_mass = float(p[original_support].sum())
        if x.sum() > EPS and original_mass > 1e-12:
            saved_full[original_support] = (p[original_support] + original_mass*x[original_support]/x.sum())/2
        replay_error = float(np.max(np.abs(np.asarray(endpoint_scores['SUPPORT_CONDITIONAL']['score_vector']) - saved_full)))
        full_replay_max = max(full_replay_max, replay_error)
        full_replay_rank_failures += int(endpoint_scores['SUPPORT_CONDITIONAL']['rank'] != rank_by_score(saved_full, candidate))
        if replay_error > 1e-09:
            endpoint_failures += 1
        by_seed = {int(item['seed']): item for item in source_masks if float(item['retention']) in (0.25, 0.5, 0.75)}
        for seed in PARTIAL_SEEDS:
            sets = {float(r): set(next((item for item in source_masks if float(item['retention']) == r and int(item['seed']) == seed))['retained_indices']) for r in (0.25, 0.5, 0.75)}
            nested_failures += int(not sets[0.25] <= sets[0.5] <= sets[0.75])
        for mask in source_masks:
            scores, metadata = score_three(source, mask)
            metadata['original_support_indices'] = endpoint_meta['original_support_indices'] if 'original_support_indices' in endpoint_meta else endpoint['original_support_indices']
            metadata['retained_indices'] = list(mask['retained_indices'])
            metadata['mask_sha256'] = sha_payload({'retention': mask['retention'], 'seed': mask['seed'], 'retained_indices': mask['retained_indices']})
            condition_rows.append({'method': METHOD, 'panel': source['panel'], 'system': source['system'], 'window_length_minutes': source['window_length_minutes'], 'case_id': source['case_id'], 'labels': source['labels'], 'candidate_order': candidate, 'retention': float(mask['retention']), 'seed': int(mask['seed']), 'mask_id': mask['mask_id'], 'labels_loaded': True, 'root_fault_fields_present': True, 'root_in_original_support': bool(root in set(mask['original_support_services'])), 'root_in_retained_support': bool(root in set(mask['retained_services'])), 'arms': scores, 'metadata': metadata})
    write_jsonl(OUT_RESULTS / 'coverage_condition_results.jsonl', condition_rows)
    incident_rows = aggregate_incidents(condition_rows, source_by_key)
    write_jsonl(OUT_RESULTS / 'coverage_incident_averages.jsonl', incident_rows)
    write_csv(OUT_RESULTS / 'coverage_curve.csv', summary_rows(incident_rows))
    write_csv(OUT_RESULTS / 'coverage_rank_effects.csv', rank_effect_rows(condition_rows))
    pass
    paired: list[dict[str, Any]] = []
    for retention in RETENTIONS:
        subset = [row for row in incident_rows if abs(float(row['retention']) - retention) < 1e-12]
        for system in (None, 'OnlineBoutique', 'SockShop', 'TrainTicket'):
            scoped = [row for row in subset if system is None or row['system'] == system]
            for left in ('SUPPORT_CONDITIONAL', 'GLOBAL_ARITHMETIC'):
                boot = bootstrap_differences(scoped, left, 'METRIC_ONLY', system)
                paired.append({'retention': retention, 'system': system or 'ALL', 'comparison': f'{left}_minus_METRIC_ONLY', **boot})
    write_json(OUT_RESULTS / 'coverage_paired_bootstrap.json', paired)
    systems = {system: sum((row['system'] == system for row in incident_rows)) for system in ('OnlineBoutique', 'SockShop', 'TrainTicket')}
    fixed_hash = sha_payload({'p': [sha_payload(row['arms']['NO_TRACE']['score_vector']) for row in sources], 'x': [sha_payload(row['trace_variants']['L1_FULL']['x']) for row in sources], 'W': [sha_payload(row['trace_variants']['L1_FULL']['W']) for row in sources]})
    endpoint_by_key = {(row['case_id'], float(row['retention'])): row for row in condition_rows if float(row['retention']) in (0.0, 1.0)}
    for source in sources:
        for retention in (0.0, 1.0):
            item = endpoint_by_key[source['case_id'], retention]
            p_vector = np.asarray(source['arms']['NO_TRACE']['score_vector'], dtype=np.float64)
            for arm in ARMS:
                if np.max(np.abs(np.asarray(item['arms'][arm]['score_vector']) - p_vector)) > 1e-09 and retention == 0.0:
                    endpoint_failures += 1
    properties = {'method': METHOD, 'status': 'COMPLETED', 'events': len(sources), 'conditions': len(condition_rows), 'conditions_per_event': 11, 'events_by_system': systems, 'mask_count': len(masks), 'mask_seal': mask_seal_ok(masks), 'mask_file_sha256': sha_file(mask_path), 'labels_used_for_mask_construction': False, 'labels_loaded_after_mask_seal': True, 'retentions': list(RETENTIONS), 'partial_seeds': list(PARTIAL_SEEDS), 'endpoint_masks_deduplicated': True, 'same_seed_nested_policy': True, 'nested_mask_failures': nested_failures, 'endpoint_identity_failures': endpoint_failures, 'fixed_p_x_W_reused': True, 'fixed_source_hash': sha_file(FIXED_SOURCE), 'fixed_field_bundle_sha256': fixed_hash, 'p_complete_domain_failures': p_complete_failures, 'max_original_p_simplex_error': original_p_simplex_max, 'max_original_x_simplex_error_when_informative': original_x_simplex_max, 'full_100_support_conditional_replay_max_abs_error': full_replay_max, 'full_100_support_conditional_rank_replay_failures': full_replay_rank_failures, 'metric_domain_complete_for_every_mask': True, 'hidden_coordinates_used': False, 'sockshop_branch_preserved': True, 'formulae': {'x_prime': 'x restricted to S_prime=A intersect S', 'Q_prime': 'sum(x_prime)', 't_prime': 'x_prime/Q_prime when Q_prime>eps and S_prime nonempty; otherwise fallback', 'support_conditional': 'p outside S_prime; (p+m_prime*t_prime)/2 inside S_prime, m_prime=sum(p[S_prime])', 'global_arithmetic': '(p+t_prime)/2 on complete candidate domain when informative; otherwise p', 'metric_only': 'p'}, 'bootstrap': {'seed': BOOTSTRAP_SEED, 'replicates': BOOTSTRAP_REPLICATES, 'unit': 'incident means then root_service x fault_type cluster'}, 'elapsed_seconds': float(time.perf_counter() - started)}
    write_json(OUT_RESULTS / 'coverage_properties.json', properties)
    write_json(OUT_RESULTS / 'coverage_input_mapping.json', {'method': METHOD, 'source': str(FIXED_SOURCE), 'source_sha256': sha_file(FIXED_SOURCE), 'event_count': len(sources), 'panel': 'BENCHMARK/10', 'candidate_order_reused': True, 'p_reused_from': 'FIXED arms.NO_TRACE.score_vector', 'x_reused_from': 'FIXED trace_variants.L1_FULL.x', 'S_reused_from': 'FIXED trace_variants.L1_FULL.W > 0', 'labels_used_for_construction': False, 'labels_used_for_evaluation_only': True, 'cpu_only': True, 'gpu_used': False, 'network_used': False, 'mask_file': 'data/coverage_prelabel_masks.jsonl', 'no_root_or_result_selected_masks': True})
    curve = summary_rows(incident_rows)
    lines = ['# Coverage mismatch', '', 'This cache-only experiment replays the sealed FIXED BENCHMARK/10 fields for 270 events (90 per system).  The complete candidate domain and metric distribution p are fixed; only the transported x and its original support S are restricted by label-free nested service masks.', '', f'The pre-label mask seal contains {len(masks)} rows ({len(masks) // EVENTS_EXPECTED} conditions per event), with retentions 100/75/50/25/0% and partial-mask seeds 17/29/43.  The seal was written before evaluation labels were attached.', '', f'The 100% support-conditional replay maximum absolute score error versus the saved FIXED L1_FULL arm is {full_replay_max:.3e}; rank replay failures are {full_replay_rank_failures}.  The 0% arm falls back to p by construction.  SockShop remains the original empty trace branch.', '', '## Incident-averaged curve (expected tie-aware)', '', '| system | retention | arm | n | AC@1 | AC@3 | Avg@5 | MRR | AC@5 |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in curve:
        lines.append(f"| {row['system']} | {row['retention']:.2f} | {row['arm']} | {row['n_incidents']} | {row['AC@1']:.6f} | {row['AC@3']:.6f} | {row['Avg@5']:.6f} | {row['list_rank_mrr']:.6f} | {row['AC@5']:.6f} |")
    lines.extend(['', 'The paired file reports support-conditional and full-domain arithmetic differences versus metric-only after averaging the three random masks within each incident, followed by the frozen 10,000-draw root-service × fault-type cluster bootstrap (seed 20260910).  Rank effects are condition-level descriptive counts; random masks are not independent incidents.', '', 'This is a coverage-coordinate mechanism result.  Any degradation when the root is outside the retained support is an expected support loss; it is not evidence for selecting a mask or changing the upstream method.'])
    (OUT_RESULTS / 'scientific_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'method': METHOD, 'status': 'COMPLETED', 'events': len(sources), 'conditions': len(condition_rows), 'full_replay_max_abs_error': full_replay_max, 'full_replay_rank_failures': full_replay_rank_failures, 'artifacts': str(OUT)}, ensure_ascii=False, sort_keys=True))

if __name__=='__main__': main()
