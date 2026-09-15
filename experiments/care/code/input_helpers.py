from __future__ import annotations
from typing import Any,Mapping
from pathlib import Path
import numpy as np
import pandas as pd
FEATURE_NAMES=['latency', 'cpu_use', 'mem_use_percent', 'mem_use_amount', 'file_write_rate', 'file_read_rate', 'net_send_rate', 'net_receive_rate', 'http_status']
def normalize_service(system: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return ''
    name = str(value)
    if system.lower() == 'onlineboutique' and name == 'frontendservice':
        return 'frontend'
    return name

def _trace_columns() -> list[str]:
    return ['traceID', 'spanID', 'serviceName', 'parentSpanID', 'startTimeMillis', 'duration', 'statusCode']

def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)

def _sorted_profile(frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    out = frame.copy()
    out['time'] = pd.to_numeric(out['time'], errors='coerce')
    out = out.dropna(subset=['time']).sort_values('time', kind='mergesort')
    out = out.drop_duplicates(subset=['time'], keep='last')
    return (out['time'].to_numpy(dtype=float), out.reset_index(drop=True))

def attach_profile(edges: pd.DataFrame, profile_series: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]]) -> pd.DataFrame:
    out = edges.copy()
    for feature in FEATURE_NAMES:
        if feature not in out.columns:
            out[feature] = np.nan
        out[feature] = pd.to_numeric(out[feature], errors='coerce').astype(float)
    for target, positions in out.groupby('target', sort=False).groups.items():
        target = str(target)
        if target not in profile_series:
            continue
        row_positions = np.asarray(list(positions), dtype=np.int64)
        target_times = out.loc[row_positions, 'time_sec'].to_numpy(dtype=float)
        for feature, (times, values) in profile_series[target].items():
            idx = np.searchsorted(times, target_times, side='right') - 1
            valid = (idx >= 0) & (idx < len(times))
            if not np.any(valid):
                continue
            assigned = np.full(len(row_positions), np.nan, dtype=float)
            assigned[valid] = values[idx[valid]]
            out.loc[row_positions, feature] = assigned
    return out
