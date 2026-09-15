# CARE on the shared benchmark

This experiment runs CARE on the Online Boutique and Train Ticket events used by TopoTrace-AD. It uses the fixed author commit recorded in `parameters.json`, not a new implementation of CARE's algorithm.

```bash
python experiments/care/code/fetch_upstream.py
python experiments/care/code/run_care.py --workers 2
```

Install `requirements.txt` from this directory first. Inputs are under `data/`, supplied in the `Experiment_Inputs.zip` release asset. `--limit 2` performs an installation check; omit it for the full 180-event run.

The author feature selection, anomaly detection, severity assignment, graph analysis, and weighted spectrum routines are called directly. The headless adapter removes four GUI display statements in memory. It does not replace the scoring algorithm. The downloaded author files remain unchanged and are checked against their hashes before execution.

Normal and incident windows are 300 seconds on either side of the injection time. Latency retains Jaeger microseconds; memory and file amounts use bytes; CPU keeps the benchmark scale. Memory fraction is usage divided by limit. Supplied file/network time series are not differenced again. Missing dimensions remain missing. Root and same-service invocations are retained, and parent identities use both trace and span IDs. The input mapping records these choices.

To prepare one event from source telemetry:

```bash
python experiments/care/code/prepare_from_raw.py --system OnlineBoutique --trace PATH_TO_TRACES_CSV --profile PATH_TO_METRICS_CSV --inject-time UNIX_SECONDS --output INPUT_DIRECTORY
```

Train Ticket traces may be Parquet files. The batch runner uses `data/input_manifest.json`. Labels from `evaluation_labels.json` are opened only after scoring has finished.

Outputs include `care_case_results.jsonl` and `care_main_results.csv` under `reproduced/`. The denominator includes all 90 events per applicable system. An event for which the author method produces no ranking receives zero retrieval credit rather than a fabricated ranking. Sock Shop is not evaluated because its benchmark representation lacks request-level traces.

Author repository and commit: see [SOURCES.md](SOURCES.md). Author files are excluded from the Git upload and obtained through the hash-checked downloader.
