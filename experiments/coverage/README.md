# Trace-support coverage

Run `code/run_coverage.py` followed by `code/run_stratification.py`, or use the repository-root commands in the main README.

`data/fixed_evidence.jsonl` contains the 270 incident inputs: the complete metric distribution, transported trace evidence, support, candidate order, and evaluation labels. The complete-method reference is computed from these inputs rather than loaded from saved results.

The retention levels are 1, .75, .5, .25, and 0. Partial levels use seeds 17, 29, and 43; endpoints use one condition. The same seeded service permutation gives nested retained supports. Metric evidence remains fixed. The three comparison arms are support-conditional consensus, full-domain arithmetic averaging, and metric-only ranking.

The 2,970 conditions are averaged within incidents before aggregate analysis. Stratification uses each condition's own support membership. The bootstrap resamples root-service/fault-category clusters, with 10,000 draws. All comparison outcomes are generated in `reproduced/`.
