import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler
from sklearn.metrics import f1_score, roc_auc_score

SEED = 42
N_RUNS = 5

df = pd.read_csv("Diabetes_data_CORRECTED.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=["PATIENT ID"])
for c in ["GENDER", "SMOKING", "ALOCHOLIC"]:
    df[c] = df[c].astype(str).str.strip().str.lower()
    df[c] = LabelEncoder().fit_transform(df[c])
y = (df["INSULIN"].astype(str).str.strip().str.upper() == "YES").astype(int).values
X = df.drop(columns=["INSULIN"]).values.astype(float)

# Deterministic preprocessing (identical for every run)
np.random.seed(SEED)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
X_tr_sm, y_tr_sm = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
sc = StandardScaler().fit(X_tr_sm)
pca = PCA(n_components=4, random_state=SEED).fit(sc.transform(X_tr_sm))
Xtr_p, Xte_p = pca.transform(sc.transform(X_tr_sm)), pca.transform(sc.transform(X_te))
scale = np.abs(Xtr_p).max(axis=0); scale[scale == 0] = 1
Xtr_q, Xte_q = np.clip(Xtr_p / scale, -np.pi, np.pi), np.clip(Xte_p / scale, -np.pi, np.pi)

def build_vqc_paper():
    return VQC(feature_map=ZZFeatureMap(feature_dimension=4, reps=2),
               ansatz=RealAmplitudes(num_qubits=4, reps=4, entanglement="full"),
               optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler())

def build_vqc_hea():
    return VQC(feature_map=ZZFeatureMap(feature_dimension=4, reps=1),
               ansatz=RealAmplitudes(num_qubits=4, reps=3, entanglement="linear"),
               optimizer=COBYLA(maxiter=150), sampler=StatevectorSampler())

results = {"VQC-Paper": [], "VQC-HEA": []}
for run_i in range(N_RUNS):
    algorithm_globals.random_seed = SEED + run_i  # distinct, but recorded, seed per run
    for name, builder in [("VQC-Paper", build_vqc_paper), ("VQC-HEA", build_vqc_hea)]:
        clf = builder()
        clf.fit(Xtr_q, y_tr_sm)
        yp = clf.predict(Xte_q)
        f1 = f1_score(y_te, yp, zero_division=0)
        auc = roc_auc_score(y_te, yp.astype(float))
        results[name].append({"run": run_i, "F1": round(f1, 4), "ROC_AUC": round(auc, 4)})
        print(f"run {run_i} {name}: F1={f1:.4f} AUC={auc:.4f}", flush=True)

summary = {}
for name, runs in results.items():
    f1s = [r["F1"] for r in runs]
    summary[name] = {
        "runs": runs,
        "mean_F1": round(float(np.mean(f1s)), 4),
        "std_F1": round(float(np.std(f1s)), 4),
        "min_F1": round(float(np.min(f1s)), 4),
        "max_F1": round(float(np.max(f1s)), 4),
    }
    print(f"\n{name}: mean={summary[name]['mean_F1']:.4f} sd={summary[name]['std_F1']:.4f} "
          f"range=[{summary[name]['min_F1']:.4f}, {summary[name]['max_F1']:.4f}]")

with open("vqc_reproducibility_runs.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE.")
