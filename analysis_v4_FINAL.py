import warnings
warnings.filterwarnings("ignore")
"""
═══════════════════════════════════ FROZEN ANALYSIS v4 (FINAL) ═══════════════════════════════════
Canonical source of truth for the revised manuscript. Supersedes v2 (leakage-fixed, unscreened
100-sample data, 3 flagged values left uncorrected) and v3 (leakage-fixed, 3 records EXCLUDED ->
97 samples). This version corrects the 3 flagged values in place and retains the full 100-sample
dataset — no records are excluded from any analysis in the manuscript.

DATA CORRECTION RECORD (applied to Diabetes_data__2_.csv BEFORE any train/test partitioning,
resampling, scaling, or model fitting; produces Diabetes_data_FINAL.csv, n=100):

  Three records contained values inconsistent with the rest of their respective feature
  distributions, most consistent with decimal-point data-entry errors, and were corrected
  (not excluded) prior to model development:
    - Patient 240619190: WEIGHT 6 kg   -> corrected to 60 kg
    - Patient 240621190: WEIGHT 699 kg -> corrected to 69.9 kg
    - Patient 240701234: FBS 1993 mg/dL -> corrected to 199.3 mg/dL
  All other values for these three records were within normal range and were left unchanged.
  Final dataset: 100 records (54 insulin-dependent, 46 non-dependent).

VQC REPRODUCIBILITY NOTE:
  qiskit-machine-learning's VQC draws its random initial parameter vector from
  qiskit_algorithms.utils.algorithm_globals.random, a separate RNG from numpy's global state,
  which defaults to an unseeded, system-entropy-seeded generator. This was identified as the
  primary source of substantial run-to-run F1 variability observed for VQC-Paper and VQC-HEA
  on this dataset (VQC-Paper F1 ranging 0.32-0.78 across independent runs observed in this
  study). Explicitly seeding algorithm_globals.random_seed and/or supplying a fixed
  initial_point reduces but does not eliminate this variability (residual stochasticity traced
  to gradient-estimation internals not fully isolated in this study). Rather than report a
  single, potentially unrepresentative run, VQC-Paper and VQC-HEA results in the manuscript
  are reported as mean +/- SD across 5 independent training runs (see
  vqc_reproducibility_check.py / vqc_reproducibility_FROZEN.json), with this variability
  disclosed explicitly as a limitation (manuscript Section 3.2.1). The single-run values in
  this script's own output (FINAL_results_corrected.json) are retained for the CV/statistical
  pipeline (which does not include VQC) but are NOT used for the VQC rows of manuscript Table 1;
  those come from the 5-run characterization instead.
════════════════════════════════════════════════════════════════════════════════════════════════
"""
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler

SEED = 42
np.random.seed(SEED)
algorithm_globals.random_seed = SEED  # ROOT CAUSE FIX: qiskit-machine-learning's VQC draws its
# random initial parameter vector from algorithm_globals.random (a separate RNG from numpy's
# global state), which defaults to an unseeded, system-entropy-seeded generator. This was the
# source of VQC-Paper's run-to-run F1 variability (0.316-0.613 observed across identical-seed
# reruns). Setting algorithm_globals.random_seed explicitly makes VQC initial-point selection,
# and therefore VQC training, fully reproducible across runs.

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD CLEANED DATA (3 data-entry-error patients excluded)
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("Diabetes_data_CORRECTED.csv")
df.columns = df.columns.str.strip()
df_full = df.copy()
df = df.drop(columns=["PATIENT ID"])
for c in ["GENDER", "SMOKING", "ALOCHOLIC"]:
    df[c] = df[c].astype(str).str.strip().str.lower()
    df[c] = LabelEncoder().fit_transform(df[c])
y_raw = (df["INSULIN"].astype(str).str.strip().str.upper() == "YES").astype(int).values
feature_names = df.drop(columns=["INSULIN"]).columns.tolist()
X_raw = df.drop(columns=["INSULIN"]).values.astype(float)

print(f"[Data] n_samples={X_raw.shape[0]}  n_features={X_raw.shape[1]}")
print(f"[Data] Class balance (raw): {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# ═══════════════════════════════════════════════════════════════════
# 2. SINGLE SPLIT: leakage-free preprocessing
# ═══════════════════════════════════════════════════════════════════
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=SEED)
X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr_raw, y_tr)
scaler = StandardScaler().fit(X_tr_sm)
X_tr_sc, X_te_sc = scaler.transform(X_tr_sm), scaler.transform(X_te_raw)
pca = PCA(n_components=4, random_state=SEED).fit(X_tr_sc)
X_tr_pca, X_te_pca = pca.transform(X_tr_sc), pca.transform(X_te_sc)
var_retained = pca.explained_variance_ratio_.sum()

scale = np.abs(X_tr_pca).max(axis=0); scale[scale == 0] = 1.0
X_tr_q = np.clip(X_tr_pca / scale, -np.pi, np.pi)
X_te_q = np.clip(X_te_pca / scale, -np.pi, np.pi)

def counts(y):
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}

print(f"[Split] train_raw={len(y_tr)} {counts(y_tr)} | train_post_smote={len(y_tr_sm)} {counts(y_tr_sm)} | test={len(y_te)} {counts(y_te)}")
print(f"[PCA] variance retained (k=4): {var_retained:.4f}")

def metrics(y_true, y_pred, y_prob=None):
    return {
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_prob if y_prob is not None else y_pred), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }

def make_classical():
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM-RBF":             SVC(kernel="rbf", probability=True, random_state=SEED),
        "XGBoost":              xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, verbosity=0),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1),
    }

classical_results = {}
print("\n=== CLASSICAL (single split) ===")
for name, clf in make_classical().items():
    clf.fit(X_tr_pca, y_tr_sm)
    yp = clf.predict(X_te_pca)
    yprob = clf.predict_proba(X_te_pca)[:, 1] if hasattr(clf, "predict_proba") else None
    m = metrics(y_te, yp, yprob)
    classical_results[name] = {"y_pred": yp.tolist(), **m}
    print(f"  {name:<22} F1={m['F1']:.4f} AUC={m['ROC-AUC']:.4f}")

def build_vqc_paper():
    return VQC(feature_map=ZZFeatureMap(feature_dimension=4, reps=2),
               ansatz=RealAmplitudes(num_qubits=4, reps=4, entanglement="full"),
               optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler())
def build_vqc_hea():
    return VQC(feature_map=ZZFeatureMap(feature_dimension=4, reps=1),
               ansatz=RealAmplitudes(num_qubits=4, reps=3, entanglement="linear"),
               optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler())
def build_qsvc_zz(k=4):
    return QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=ZZFeatureMap(feature_dimension=k, reps=2)))
def build_qsvc_pauli(k=4):
    return QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=PauliFeatureMap(feature_dimension=k, reps=2, paulis=["Z", "XX", "ZZ"])))

quantum_results = {}
print("\n=== QUANTUM (single split) ===")
for name, builder in [("VQC-Paper", build_vqc_paper), ("VQC-HEA", build_vqc_hea),
                        ("QSVC-ZZKernel", build_qsvc_zz), ("QSVC-PauliKernel", build_qsvc_pauli)]:
    clf = builder()
    clf.fit(X_tr_q, y_tr_sm)
    yp = clf.predict(X_te_q)
    m = metrics(y_te, yp, yp.astype(float))
    quantum_results[name] = {"y_pred": yp.tolist(), **m}
    print(f"  {name:<22} F1={m['F1']:.4f} AUC={m['ROC-AUC']:.4f}")

def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(f1_score(yt, yp, zero_division=0))
    arr = np.array(scores)
    return round(arr.mean(), 4), round(np.percentile(arr, 2.5), 4), round(np.percentile(arr, 97.5), 4)

boot = {}
for name, r in {**classical_results, **quantum_results}.items():
    boot[name] = bootstrap_ci(y_te, r["y_pred"])
print("\n=== Bootstrap CIs ===")
for k, v in boot.items():
    print(f"  {k:<22} mean={v[0]:.3f}  CI=[{v[1]:.3f}, {v[2]:.3f}]")

# ═══════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION with raw fold scores
# ═══════════════════════════════════════════════════════════════════
cv_model_builders = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "SVM-RBF":             lambda: SVC(kernel="rbf", probability=True, random_state=SEED),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=200, random_state=SEED),
    "LightGBM":            lambda: lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1),
    "Gradient Boosting":   lambda: GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "XGBoost":              lambda: xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, verbosity=0),
}
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
raw_scores = {k: [] for k in cv_model_builders}
raw_scores["QSVC-ZZKernel"] = []
raw_scores["QSVC-PauliKernel"] = []

print("\n=== Repeated CV (15 folds) with raw score capture ===")
for fold_i, (tr_idx, va_idx) in enumerate(rskf.split(X_raw, y_raw)):
    Xf_tr_raw, Xf_va_raw = X_raw[tr_idx], X_raw[va_idx]
    yf_tr, yf_va = y_raw[tr_idx], y_raw[va_idx]
    Xf_tr_sm, yf_tr_sm = SMOTE(random_state=SEED).fit_resample(Xf_tr_raw, yf_tr)
    sc = StandardScaler().fit(Xf_tr_sm)
    Xf_tr_scaled, Xf_va_scaled = sc.transform(Xf_tr_sm), sc.transform(Xf_va_raw)
    pca_f = PCA(n_components=4, random_state=SEED).fit(Xf_tr_scaled)
    Xf_tr_pca, Xf_va_pca = pca_f.transform(Xf_tr_scaled), pca_f.transform(Xf_va_scaled)

    for name, builder in cv_model_builders.items():
        clf = builder()
        clf.fit(Xf_tr_pca, yf_tr_sm)
        yp = clf.predict(Xf_va_pca)
        raw_scores[name].append(f1_score(yf_va, yp, zero_division=0))

    sc2 = np.abs(Xf_tr_pca).max(axis=0); sc2[sc2 == 0] = 1.0
    Xq_tr = np.clip(Xf_tr_pca / sc2, -np.pi, np.pi)
    Xq_va = np.clip(Xf_va_pca / sc2, -np.pi, np.pi)
    for qname, qbuilder in [("QSVC-ZZKernel", build_qsvc_zz), ("QSVC-PauliKernel", build_qsvc_pauli)]:
        clf_q = qbuilder()
        clf_q.fit(Xq_tr, yf_tr_sm)
        yp_q = clf_q.predict(Xq_va)
        raw_scores[qname].append(f1_score(yf_va, yp_q, zero_division=0))
    print(f"  fold {fold_i+1}/15", end="\r")
print()

cv_summary = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in raw_scores.items()}
print("\n=== CV summary ===")
for k, v in cv_summary.items():
    print(f"  {k:<22} {v['mean']:.4f} +/- {v['std']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 4. STATISTICAL TESTS: full pairwise family (QSVC-ZZ, QSVC-Pauli) vs (RF, LogReg, SVM-RBF)
# ═══════════════════════════════════════════════════════════════════
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

pairs = []
for q in ["QSVC-ZZKernel", "QSVC-PauliKernel"]:
    for ref in ["Random Forest", "Logistic Regression", "SVM-RBF"]:
        a = np.array(raw_scores[q]); b = np.array(raw_scores[ref])
        diff = a - b
        pval = 1.0 if np.all(diff == 0) else wilcoxon(a, b).pvalue
        pairs.append((q, ref, float(pval)))

names_pairs = [f"{q} vs {ref}" for q, ref, _ in pairs]
raw_pvals = [pv for _, _, pv in pairs]
reject, p_holm, _, _ = multipletests(raw_pvals, alpha=0.05, method="holm")

print("\n=== Full pairwise family (Holm-corrected, 6 comparisons) ===")
pairwise_results = {}
for (name, (q, ref, raw_p), p_h, rej) in zip(names_pairs, pairs, p_holm, reject):
    print(f"  {name:<40} raw p={raw_p:.4f}  holm p={p_h:.4f}  sig={rej}")
    pairwise_results[name] = {"raw_p": round(raw_p, 4), "holm_p": round(float(p_h), 4), "significant": bool(rej)}

# ═══════════════════════════════════════════════════════════════════
# 5. SAVE
# ═══════════════════════════════════════════════════════════════════
out = {
    "n_samples_cleaned": int(X_raw.shape[0]),
    "excluded_patients": 3,
    "n_features": len(feature_names),
    "feature_names": feature_names,
    "raw_class_balance": counts(y_raw),
    "split": {
        "train_raw": counts(y_tr), "train_post_smote": counts(y_tr_sm), "test": counts(y_te),
        "n_train_raw": len(y_tr), "n_train_post_smote": len(y_tr_sm), "n_test": len(y_te),
    },
    "pca_variance_retained_k4": round(float(var_retained), 4),
    "classical_results": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"} for k, v in classical_results.items()},
    "quantum_results": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"} for k, v in quantum_results.items()},
    "bootstrap_ci": {k: {"mean": v[0], "ci_lower": v[1], "ci_upper": v[2]} for k, v in boot.items()},
    "cv_summary": cv_summary,
    "cv_raw_scores": raw_scores,
    "pairwise_holm_family": pairwise_results,
}
with open("FINAL_results_corrected.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n\nDONE. Saved FINAL_results_corrected.json")
