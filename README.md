# TopoTrace-AD

TopoTrace-AD diagnoses microservice failures using service metrics and operation-level trace evidence. It estimates operation persistence relative to normal observations, assigns evidence to services through normal call roles, and combines metric and trace evidence within the observed trace support.

## Contents

| Directory | Contents |
|---|---|
| `code/` | Method implementation, ranking metrics, and main experiment runner |
| `data/` | Prepared method inputs and separate evaluation labels |
| `datasets/` | Source dataset archives, checksums, and event mappings |
| `experiments/care/` | CARE input adapter and comparison runner |
| `experiments/coverage/` | Trace-support retention experiment and stratified analysis |
| `experiments/normal_roles/` | Normal-call-role observation retention experiment |

Generated scores and tables are not part of the repository. Each runner writes new outputs locally.

## Installation

Use Python 3.12. A GPU is not required. Run commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment using `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

CARE has additional dependencies:

```bash
python -m pip install -r experiments/care/requirements.txt
python experiments/care/code/fetch_upstream.py
```

The CARE downloader uses the commit and SHA-256 hashes in `experiments/care/parameters.json`. It retrieves the author's source files without changing them. The local adapter disables interactive figures and supplies benchmark feature inputs. Source attribution and adaptation details are in `experiments/care/README.md`.

## Data

The main benchmark contains 270 incidents: 90 each from Online Boutique, Sock Shop, and Train Ticket. Generalization uses a separate set of 30 RE3 Online Boutique incidents. The 570 prepared records represent two observation windows for the 270 main incidents plus 30 generalization records; they are not 570 independent incidents.

Download the data assets from this repository's Releases page. Place the four source archives in `datasets/`. Extract `Experiment_Inputs.zip` into the repository root, preserving its `experiments/` directory structure. The local distribution already includes these inputs.

| Archive | Data |
|---|---|
| `TORAI_Main_270.zip` | Author-published preprocessed main benchmark |
| `RE2_OB_Raw_90.zip` | Online Boutique raw telemetry |
| `RE2_TT_Raw_90.zip` | Train Ticket request traces; matching metrics are in the main archive |
| `RE3_OB_Generalization_30.zip` | Generalization telemetry |
| `Experiment_Inputs.zip` | CARE feature inputs and compact normal-role observations |

The main runner starts from prepared observations; supplementary runners start from their documented experiment inputs. The source telemetry archives are included for access to the underlying data. They do not replace the prepared inputs used by these commands. `datasets/README.md` describes the representations and sources.

## Main experiments

```bash
python code/run_main.py
```

This computes all six configurations on all 570 records, including the five- and ten-minute main windows and the generalization set. Results are written to `outputs/main/case_results.jsonl`, `summary.csv`, and `pooled_summary.csv`.

| Configuration | Definition |
|---|---|
| `L1_FULL` | Complete TopoTrace-AD method |
| `MEAN_FULL` | Mean operation persistence in place of median persistence |
| `NO_TRACE` | Metric evidence alone |
| `NO_METRIC_VALUES` | Uniform supported metric evidence |
| `IDENTITY_DEST` | Evidence remains at its source service |
| `MEAN_IDENTITY_DEST` | Mean persistence with source-service attribution |

AC@1, AC@3, Avg@5, MRR, and AC@5 are evaluated with uniform expectation within score ties. Avg@5 is the mean of AC@1 through AC@5. Ordinary ordered-list metrics are also written per incident. Labels are read only after scoring finishes.

## Supplementary experiments

### CARE comparison

```bash
python experiments/care/code/run_care.py --workers 2
```

This evaluates CARE on the Online Boutique and Train Ticket benchmark inputs. It does not use CARE's original evaluation dataset. Sock Shop lacks the request-level traces required by CARE. The runner writes per-event scores, rankings, and aggregate metrics under `experiments/care/reproduced/`.

### Trace-support coverage

```bash
python experiments/coverage/code/run_coverage.py
python experiments/coverage/code/run_stratification.py
```

The experiment retains 100%, 75%, 50%, 25%, or 0% of the observed support. Partial retentions use seeds 17, 29, and 43; endpoints are evaluated once. Stratification uses each condition's own support membership and averages seed conditions within incidents. Outputs are written under `experiments/coverage/reproduced/`.

### Normal-call-role robustness

```bash
python experiments/normal_roles/code/run_normal.py
```

The experiment retains 100%, 50%, or 25% of complete normal-role observation units. It covers 180 incidents and 1,260 retention/seed conditions. Responsibility transport is rebuilt after sampling, while incident evidence and metric inputs remain fixed. Outputs are written under `experiments/normal_roles/reproduced/`.

## Sources and attribution

- [RCAEval and TORAI benchmark](https://github.com/phamquiluan/RCAEval#for-torai-paper)
- [TORAI data, version 1](https://doi.org/10.6084/m9.figshare.31925976.v1)
- [RE2 and RE3 datasets](https://doi.org/10.5281/zenodo.14590730)
- [CARE author implementation](https://github.com/M-panahandeh/CARE-Context-Aware-Root-Cause-Identification-Using-Distributed-Traces-and-Profiling-Metrics)

Third-party data retain their source licenses and attribution. CARE source files are downloaded separately under the author's terms. See `THIRD_PARTY.md`. A license for the original project code should be selected by the authors before public release.
