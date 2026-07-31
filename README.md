# Code Availability — Revision Notes

This repository accompanies "Evaluation of Quantum Machine Learning Architectures for
Clinical Insulin Prediction under NISQ Hardware Constraints." Following peer review,
the analysis pipeline went through several corrections; this note documents what
changed and which script is canonical.

## Canonical analysis script (v4 — current)

**`analysis_v4_FINAL.py`** is the source of truth for every number, table, and figure
in the revised manuscript. Input: `Diabetes_data_FINAL.csv` (100 records).

Three corrections are included, in order of discovery:

1. **Leakage fix:** SMOTE, feature scaling, and PCA are fit exclusively on training
   data — splitting first, then resampling only the training partition, and
   re-fitting all three independently inside each cross-validation fold.
2. **Data correction (not exclusion):** three values inconsistent with the rest of
   their feature distributions, all consistent with decimal-point transcription
   errors, were corrected prior to any modelling:
   - Patient 240619190: WEIGHT 6 kg -> corrected to 60 kg
   - Patient 240621190: WEIGHT 699 kg -> corrected to 69.9 kg
   - Patient 240701234: FBS 1993 mg/dL -> corrected to 199.3 mg/dL

   No records are excluded. All 100 original patients are retained in every
   analysis reported in the manuscript.
3. **VQC reproducibility fix (partial):** `qiskit-machine-learning`'s VQC draws its
   random initial parameter vector from `qiskit_algorithms.utils.algorithm_globals.random`
   -- a separate RNG from numpy's global state that defaults to unseeded,
   system-entropy-initialised behaviour. This was the primary source of large
   run-to-run F1 swings observed for VQC-Paper and VQC-HEA (e.g. VQC-Paper F1
   ranging 0.32-0.78 across identical-seed reruns). `analysis_v4_FINAL.py` sets
   `algorithm_globals.random_seed` explicitly, which fixes the initial parameter
   vector but does NOT fully eliminate run-to-run variability -- a residual
   source (likely in gradient estimation) was traced but not fully isolated.
   Consequently, VQC results are reported as mean +/- SD across 5 independent
   runs rather than a single point estimate (see `vqc_reproducibility_check_FINAL.py`
   / `vqc_reproducibility_FROZEN.json`, manuscript Section 3.2.1), and this is
   documented as an explicit, quantified limitation rather than hidden.

Full configuration (seeds, package versions, exact preprocessing order, the full
correction record above) is documented in the v4 file's header docstring.

## Hardware execution script

**`quantum_diabetes_hardware_SANITIZED.py`** contains the actual IBM Quantum
`ibm_fez` hardware-execution code referenced in Manuscript Section 3.7. Notes:

1. This script executes ONE quantum architecture (VQC with `ZZFeatureMap(reps=2)`
   + `RealAmplitudes(reps=3, entanglement='full')`) -- not all four models evaluated
   elsewhere in the manuscript, and not the same ansatz reps count as the primary
   VQC-Paper architecture (reps=4) reported in Table 2. See manuscript Table 2a.
2. The hardware run returns predicted labels only; raw measurement counts/quasi-
   distributions were not retained, and shot count / transpiler optimization level /
   error-mitigation settings were not explicitly configured or logged beyond Qiskit
   Runtime defaults. This is disclosed as a limitation in the manuscript (Sections
   2.4, 3.7, 4) rather than reconstructed after the fact.

Do not hardcode API credentials. Set your IBM Quantum Cloud token as an
environment variable before running:

```bash
export IBM_QUANTUM_TOKEN="your_token_here"
python3 quantum_diabetes_hardware_SANITIZED.py
```
## Supplementary analyses

- `pca_sensitivity_FINAL.py` / `pca_sensitivity_results_FROZEN.json` --
  dimensionality sensitivity sweep on the corrected 100-sample dataset (Manuscript
  Section 3.5, Table 4, Figure 7), evaluating classical and quantum-kernel models
  across PCA component counts k = 2, 4, 6, 8, 10, 11, on the single held-out test
  split.
- `vqc_reproducibility_check_FINAL.py` / `vqc_reproducibility_FROZEN.json` --
  characterizes VQC-Paper and VQC-HEA F1 variability across 5 independent training
  runs on the identical data partition (Manuscript Section 3.2.1, Table S3, Figure 8).

## Reproducing the reported results

```bash
pip install qiskit==1.3.2 qiskit-aer==0.15.1 qiskit-machine-learning==0.8.2 \
            qiskit-algorithms xgboost lightgbm imbalanced-learn scikit-learn \
            scipy statsmodels pandas numpy

python3 analysis_v4_FINAL.py                # -> FINAL_results_corrected.json
python3 pca_sensitivity_FINAL.py            # -> pca_sensitivity_results.json
python3 vqc_reproducibility_check_FINAL.py  # -> vqc_reproducibility_runs.json
```

All three scripts read `Diabetes_data_FINAL.csv` from the working directory.
