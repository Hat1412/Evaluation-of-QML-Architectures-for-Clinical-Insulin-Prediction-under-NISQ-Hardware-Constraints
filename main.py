#Evaluation of Quantum Machine Learning Architectures for Clinical Insulin Prediction under NISQ Hardware Constraints

"""
Requirements
pip install qiskit==1.3.2 qiskit-aer==0.15.1 qiskit-machine-learning==0.8.2
pip install qiskit-algorithms scikit-learn imbalanced-learn
pip install xgboost lightgbm scipy statsmodels matplotlib numpy pandas
"""
import warnings
warnings.filterwarnings("ignore")
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, roc_curve)

from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import xgboost as xgb
import lightgbm as lgb

# ── Qiskit ──────────────────────────────────────────────────────────────────
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# ── Plotting ────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
# ═══════════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(X, y, n_pca=4, test_size=0.20, seed=SEED):
    """SMOTE → StandardScaler → PCA → train/test split."""
    X_sm, y_sm = SMOTE(random_state=seed).fit_resample(X, y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sm, y_sm, test_size=test_size, stratify=y_sm, random_state=seed)

    scaler = StandardScaler().fit(X_tr)
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    pca = PCA(n_components=n_pca, random_state=seed).fit(X_tr_sc)
    X_tr_pca = pca.transform(X_tr_sc)
    X_te_pca  = pca.transform(X_te_sc)

    var_retained = pca.explained_variance_ratio_.sum()
    print(f"  PCA variance retained : {var_retained:.3f}  "
          f"({n_pca} components, {len(y_sm)} samples after SMOTE)")

    # Quantum-scaled inputs: clip to [-π, π]
    scale = np.abs(X_tr_pca).max(axis=0)
    scale[scale == 0] = 1.0
    X_tr_q = np.clip(X_tr_pca / scale, -np.pi, np.pi)
    X_te_q  = np.clip(X_te_pca  / scale, -np.pi, np.pi)

    return (X_tr_pca, X_te_pca, X_tr_q, X_te_q,
            y_tr, y_te, X_sm, y_sm, scaler, pca)


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVALUATION HELPER
# ═══════════════════════════════════════════════════════════════════════════
def metrics(y_true, y_pred, y_prob=None):
    m = {
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob if y_prob is not None
                                         else y_pred), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 4. CLASSICAL MODELS
# ═══════════════════════════════════════════════════════════════════════════
def train_classical(X_tr, X_te, y_tr, y_te):
    models = {
        "Random Forest":     RandomForestClassifier(n_estimators=200,
                                                     random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=SEED),
        "SVM-RBF":           SVC(kernel="rbf", probability=True,
                                  random_state=SEED),
        "XGBoost":           xgb.XGBClassifier(n_estimators=200,
                                                eval_metric="logloss",
                                                random_state=SEED,
                                                verbosity=0),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                         random_state=SEED),
        "LightGBM":          lgb.LGBMClassifier(n_estimators=200,
                                                  random_state=SEED,
                                                  verbose=-1),
    }
    results = {}
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = (clf.predict_proba(X_te)[:, 1]
                  if hasattr(clf, "predict_proba") else None)
        m = metrics(y_te, y_pred, y_prob)
        results[name] = {"model": clf, "y_pred": y_pred,
                         "y_prob": y_prob, **m}
        print(f"  {name:<25} F1={m['F1']:.4f}  AUC={m['ROC-AUC']:.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. QUANTUM MODELS
# ═══════════════════════════════════════════════════════════════════════════
def build_vqc_paper():
    """VQC: ZZFeatureMap(reps=2) + RealAmplitudes(reps=4, full), COBYLA."""
    return VQC(
        feature_map=ZZFeatureMap(feature_dimension=4, reps=2),
        ansatz=RealAmplitudes(num_qubits=4, reps=4, entanglement="full"),
        optimizer=COBYLA(maxiter=150),
        sampler=StatevectorSampler(),
    )

def build_vqc_hea():
    """VQC HW-efficient: ZZFeatureMap(reps=1) + RealAmplitudes(reps=3,
    linear), COBYLA.  Fewer CX gates → higher estimated hardware fidelity."""
    return VQC(
        feature_map=ZZFeatureMap(feature_dimension=4, reps=1),
        ansatz=RealAmplitudes(num_qubits=4, reps=3, entanglement="linear"),
        optimizer=COBYLA(maxiter=150),
        sampler=StatevectorSampler(),
    )

def build_qsvc_zz():
    """QSVC with ZZFeatureMap quantum kernel."""
    fm = ZZFeatureMap(feature_dimension=4, reps=2)
    return QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm))

def build_qsvc_pauli():
    """QSVC with PauliFeatureMap quantum kernel (Z + XX + ZZ terms)."""
    fm = PauliFeatureMap(feature_dimension=4, reps=2, paulis=["Z", "XX", "ZZ"])
    return QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm))


def train_quantum(X_tr_q, X_te_q, y_tr, y_te):
    q_models = {
        "VQC-Paper":         build_vqc_paper,
        "VQC-HEA":           build_vqc_hea,
        "QSVC-ZZKernel":     build_qsvc_zz,
        "QSVC-PauliKernel":  build_qsvc_pauli,
    }
    results = {}
    for name, builder in q_models.items():
        print(f"  {name:<25}", end="", flush=True)
        clf = builder()
        t0 = time.time()
        clf.fit(X_tr_q, y_tr)
        elapsed = time.time() - t0
        y_pred = clf.predict(X_te_q)
        y_prob = y_pred.astype(float)   # no predict_proba for QSVC/VQC
        m = metrics(y_te, y_pred, y_prob)
        results[name] = {"model": clf, "y_pred": y_pred,
                         "y_prob": y_prob, **m}
        print(f" F1={m['F1']:.4f}  AUC={m['ROC-AUC']:.4f}  "
              f"({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. CIRCUIT FIDELITY ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════
CIRCUIT_INFO = {
    # name: (cx_gates, sq_gates, depth, n_params)
    "VQC-Paper":        (48, 28, 52, 20),
    "VQC-HEA":          (16, 12, 28, 12),
    "QSVC-ZZKernel":    (14, 8,  24,  0),
    "QSVC-PauliKernel": (14, 10, 20,  0),
}
CX_ERR, SQ_ERR = 0.015, 0.003   # IBM ibm_fez calibration

def fidelity(cx, sq):
    return round((1 - CX_ERR) ** cx * (1 - SQ_ERR) ** sq, 4)


def print_fidelity_table():
    print("\n  Circuit complexity and fidelity:")
    print(f"  {'Model':<25} {'CX':>4} {'Depth':>6} {'Params':>7} "
          f"{'Fidelity':>10}")
    print("  " + "-" * 56)
    for name, (cx, sq, depth, params) in CIRCUIT_INFO.items():
        f = fidelity(cx, sq)
        print(f"  {name:<25} {cx:>4} {depth:>6} {params:>7} {f:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
def run_cv(X_pca, y_sm, n_splits=5, n_repeats=3, seed=SEED):
    """
    Repeated stratified k-fold CV for classical models and quantum kernels.
    VQC models are excluded from CV due to computational cost.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=seed)
    cv_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   random_state=seed),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=200,
                                                    random_state=seed,
                                                    verbose=-1),
        "SVM-RBF":             SVC(kernel="rbf", probability=True,
                                    random_state=seed),
        "XGBoost":             xgb.XGBClassifier(n_estimators=200,
                                                   eval_metric="logloss",
                                                   random_state=seed,
                                                   verbosity=0),
        "Random Forest":       RandomForestClassifier(n_estimators=200,
                                                       random_state=seed),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200,
                                                           random_state=seed),
    }
    cv_scores = {k: [] for k in cv_models}

    # Quantum kernel CV (using fresh kernel per fold)
    cv_scores["QSVC-ZZKernel"]    = []
    cv_scores["QSVC-PauliKernel"] = []

    print("\n  Running cross-validation …")
    for fold_i, (tr_idx, va_idx) in enumerate(rskf.split(X_pca, y_sm)):
        Xf_tr, Xf_va = X_pca[tr_idx], X_pca[va_idx]
        yf_tr, yf_va = y_sm[tr_idx],  y_sm[va_idx]

        # Classical
        for name, clf in cv_models.items():
            clf.fit(Xf_tr, yf_tr)
            yp = clf.predict(Xf_va)
            cv_scores[name].append(f1_score(yf_va, yp, zero_division=0))

        # Quantum kernels
        sc = np.abs(Xf_tr).max(axis=0); sc[sc == 0] = 1.0
        Xq_tr = np.clip(Xf_tr / sc, -np.pi, np.pi)
        Xq_va = np.clip(Xf_va / sc, -np.pi, np.pi)

        for qname, qbuilder in [("QSVC-ZZKernel",    build_qsvc_zz),
                                  ("QSVC-PauliKernel", build_qsvc_pauli)]:
            clf_q = qbuilder()
            clf_q.fit(Xq_tr, yf_tr)
            yp_q = clf_q.predict(Xq_va)
            cv_scores[qname].append(f1_score(yf_va, yp_q, zero_division=0))

        print(f"  fold {fold_i+1}/{n_splits*n_repeats}", end="\r")

    print()
    return cv_scores


def print_cv_table(cv_scores):
    rows = sorted(cv_scores.items(),
                  key=lambda kv: np.mean(kv[1]), reverse=True)
    print(f"\n  {'Model':<25} {'CV-F1 Mean':>12} {'Std Dev':>10}")
    print("  " + "-" * 50)
    for name, scores in rows:
        print(f"  {name:<25} {np.mean(scores):>12.4f} "
              f"{np.std(scores):>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════
def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(f1_score(yt, yp, zero_division=0))
    arr = np.array(scores)
    return (round(arr.mean(), 4),
            round(np.percentile(arr, 2.5), 4),
            round(np.percentile(arr, 97.5), 4))


# ═══════════════════════════════════════════════════════════════════════════
# 9. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════
def wilcoxon_vs_rf(cv_scores):
    """Pairwise Wilcoxon signed-rank test against Random Forest."""
    rf = np.array(cv_scores["Random Forest"])
    print(f"\n  Wilcoxon signed-rank tests vs Random Forest (α=0.05):")
    print(f"  {'Model':<25} {'p-value':>10}  {'Significant':>12}")
    print("  " + "-" * 52)
    for name, scores in cv_scores.items():
        if name == "Random Forest":
            continue
        diff = np.array(scores) - rf
        if np.all(diff == 0):
            p = 1.0
        else:
            _, p = wilcoxon(np.array(scores), rf)
        sig = "Yes *" if p < 0.05 else "No"
        print(f"  {name:<25} {p:>10.4f}  {sig:>12}")


def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):
    """McNemar's exact test between two classifiers."""
    b = int(np.sum((pred_a == y_true) & (pred_b != y_true)))
    c = int(np.sum((pred_a != y_true) & (pred_b == y_true)))
    table = [[int(np.sum((pred_a == y_true) & (pred_b == y_true))), b],
             [c, int(np.sum((pred_a != y_true) & (pred_b != y_true)))]]
    result = mcnemar(table, exact=True)
    sig = "Yes *" if result.pvalue < 0.05 else "No"
    print(f"  {name_a:<22} vs {name_b:<22}  "
          f"b={b}  c={c}  p={result.pvalue:.4f}  {sig}")


if __name__ == "__main__":
    main()
