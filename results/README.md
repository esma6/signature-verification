# Results

This folder contains the experimental outputs for all six experiments
(3 datasets × 2 scenarios), produced under one identical training protocol.

## Files

- `results.csv` — all metrics in one machine-readable table.
- `metrics/all_biometric_metrics.json` — aggregate biometric metrics (FAR/FRR/AER/EER) per dataset/scenario.
- `metrics/<dataset>_<scenario>.json` — per-experiment classification metrics.
- `figures/comparison_<dataset>.png` — WD vs WI biometric comparison per dataset.
- `figures/overall_eer.png` — EER comparison across all three datasets.

## Summary

Difficulty ordering by writing system (lower EER = easier):
**CEDAR (Latin) < BHSig (Devanagari) < UTSig (Perso-Arabic)**.

The dataset effect dominates the writer-dependent/independent scenario effect.
