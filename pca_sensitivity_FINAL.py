"""
PCA sensitivity analysis: does restricting to 4 components (the 4-qubit
hardware limit) artificially handicap model performance vs. using more
components? Classical models can use any n_components; QSVC/VQC are
capped at what maps onto real qubit counts. VQC is excluded here (cost).
"""
import warnings
warnings.filterwarnings("ignore")
import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

SEED = 42
np.random.seed(SEED)

df = pd.read_csv("Diabetes_data_CORRECTED.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["PATIENT ID"])
for c in ["GENDER", "SMOKING", "ALOCHOLIC"]:
    df[c] = df[c].astype(str).str.strip().str.lower()
    df[c] = LabelEncoder().fit_transform(df[c])
y_raw = (df["INSULIN"].astype(str).str.strip().str.upper() == "YES").astype(int).values
X_raw = df.drop(columns=["INSULIN"]).values.astype(float)

N_QUBITS_MAX = 8  # QSVC beyond 8 qubits/features becomes too slow for this sweep
COMPONENT_GRID = [2, 4, 6, 8, 10, 11]

def make_classical():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM-RBF":             SVC(kernel="rbf", probability=True, random_state=SEED),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=SEED),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "XGBoost":              xgb.XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=SEED, verbosity=0),
        "LightGBM":            lgb.LGBMClassifier(n_estimators=200, random_state=SEED, verbose=-1),
    }

def run_for_k(k, X_raw, y_raw, run_quantum):
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=SEED)
    X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr_raw, y_tr)
    sc = StandardScaler().fit(X_tr_sm)
    X_tr_sc, X_te_sc = sc.transform(X_tr_sm), sc.transform(X_te_raw)
    pca = PCA(n_components=k, random_state=SEED).fit(X_tr_sc)
    X_tr_pca, X_te_pca = pca.transform(X_tr_sc), pca.transform(X_te_sc)
    var = pca.explained_variance_ratio_.sum()

    row = {"n_components": k, "variance_retained": round(float(var), 4)}

    for name, clf in make_classical().items():
        clf.fit(X_tr_pca, y_tr_sm)
        yp = clf.predict(X_te_pca)
        row[f"{name}_F1"] = round(f1_score(y_te, yp, zero_division=0), 4)

    if run_quantum and k <= N_QUBITS_MAX:
        scale = np.abs(X_tr_pca).max(axis=0); scale[scale == 0] = 1.0
        Xq_tr = np.clip(X_tr_pca / scale, -np.pi, np.pi)
        Xq_te = np.clip(X_te_pca / scale, -np.pi, np.pi)

        fm_zz = ZZFeatureMap(feature_dimension=k, reps=2)
        qsvc_zz = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm_zz))
        qsvc_zz.fit(Xq_tr, y_tr_sm)
        row["QSVC-ZZKernel_F1"] = round(f1_score(y_te, qsvc_zz.predict(Xq_te), zero_division=0), 4)

        fm_pauli = PauliFeatureMap(feature_dimension=k, reps=2, paulis=["Z", "XX", "ZZ"])
        qsvc_pauli = QSVC(quantum_kernel=FidelityQuantumKernel(feature_map=fm_pauli))
        qsvc_pauli.fit(Xq_tr, y_tr_sm)
        row["QSVC-PauliKernel_F1"] = round(f1_score(y_te, qsvc_pauli.predict(Xq_te), zero_division=0), 4)
    else:
        row["QSVC-ZZKernel_F1"] = None
        row["QSVC-PauliKernel_F1"] = None

    return row

results = []
for k in COMPONENT_GRID:
    print(f"\n--- n_components = {k} ---")
    row = run_for_k(k, X_raw, y_raw, run_quantum=True)
    for key, val in row.items():
        print(f"  {key}: {val}")
    results.append(row)

with open("pca_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\n=== DONE ===")
