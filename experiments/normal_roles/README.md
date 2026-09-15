# Normal-call-role robustness

Run `code/run_normal.py`, or use the repository-root command in the main README.

The experiment covers 90 Online Boutique and 90 Train Ticket incidents. Normal observation retention is 1, .5, or .25. Reduced retentions use seeds 17, 29, and 43, giving 1,260 incident/retention/seed conditions in total.

`data/compact_roles/` stores complete inbound/outbound role relations in canonical observation order. A unit is a normal span together with its role relations; sampling retains `floor(retention * unit_count)` units. The code rebuilds responsibility transport after sampling. Metric evidence and incident operation energies remain fixed. Identity attribution is recomputed from the operation energies, without loading a reference prediction.

The inputs are supplied in `Experiment_Inputs.zip`. Two CPU processes execute the experiment. New score vectors, rankings, and metrics are written to `reproduced/`; the input data do not contain published Full or Identity result tables.
