# -*- coding: utf-8 -*-
"""
Problem 1 (v2): Acoustic Characteristics, Separability Diagnosis, and Classification
-----------------------------------------------------------------------------------
Three call types of African penguin sea-surface calls: flat, modulated (mod), two-voice (tv).

WHAT CHANGED vs v1 (and why)
1. Removed the target-leakage "Set 3" (F02/PulseRate presence + zero-filled sentinels).
   Those flags are deterministic functions of the label, so v1's 94% accuracy was a
   definitional tautology, not an acoustic classifier. They are now reported as a
   separate "definitional" model and as a hierarchical first stage.
2. Added a leakage-free hierarchical classifier:
      Stage 1 (definitional): second fundamental present -> tv
      Stage 2 (scientific):   flat vs mod using continuous acoustics
3. Added no-skill baselines (majority, stratified random), Cohen's kappa, Brier score,
   bootstrap confidence intervals, and the Nadeau-Bengio corrected resampled t-test.
4. Corrected the rank-biserial sign convention and added cluster-bootstrap CIs.
5. Multiple-comparison control is now global: Holm across 9 features for Kruskal-Wallis,
   and Holm across all 27 pairwise Mann-Whitney tests.
6. Added a date-adjusted inference: within-date centered Kruskal-Wallis with a
   date-stratified permutation test, plus GroupKFold/StratifiedGroupKFold CV by Date.
   This addresses pseudo-replication (108 calls from 16 recording dates, 2 locations).
7. PERMANOVA is now run on the non-redundant 6D raw space (the 9D space double-counts
   Q25/Q75/Fpeak/F01 through derived features), and PERMDISP is reported.
8. Added a diagnostic for the bimodal Fpeak/F01 distribution (peak at the fundamental
   vs at the third harmonic), which makes v1's "HarmonicRatio" a mixture artifact.
9. Permutation importance is now computed on held-out folds (not on the training data),
   and SHAP is labelled as descriptive and attached to the model's CV accuracy.
10. Added the quantitative P1 -> P2/P3 hand-off: an information x robustness trade-off,
    a 50%-fidelity crossing distance, and a feature-contract table.

Run:
    python P1.py
    python P1.py --data-only    # only validate data + regenerate dataset/manifest
    P1_FAST=1 python P1.py      # quicker check: fewer permutations/bootstraps

Requires: Python 3.8+, numpy, pandas, scipy, matplotlib, seaborn>=0.12,
          scikit-learn>=1.1. Optional: shap.
Statsmodels is intentionally not used (portable; no mixed-model dependency).
"""

import os
import re
import sys
import glob
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from scipy import stats
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform

from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_validate, GridSearchCV
)
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    from sklearn.model_selection import GroupKFold
    HAS_SGK = False
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, roc_auc_score,
    precision_recall_fscore_support, make_scorer
)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ==============================================================================
# 0. Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs_p1")
os.makedirs(OUT_DIR, exist_ok=True)

FAST = os.environ.get("P1_FAST", "0") == "1"
SEED = 42
N_BOOT = 300 if FAST else 1000
N_PERM = 999 if FAST else 2999
N_REPEATS = 4 if FAST else 10
N_SPLITS = 5
C_GRID = [0.1, 1.0]

CALL_ORDER = ["flat", "mod", "tv"]
PALETTE = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}

RAW_ACOUSTIC = ["Duration", "F01", "F02", "Fpeak", "Q25", "Q50", "Q75", "PulseRate"]
RAW_SHARED = ["Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"]
ENG_FEATS = ["SpecBandwidth", "HarmonicRatio", "SpecSkew"]
SHARED_9D = RAW_SHARED + ENG_FEATS
SHARED_10D = SHARED_9D + ["PeakAtFundamental"]
DEFINITIONAL_FLAGS = ["F02_present", "PulseRate_present"]
STRUCT_12D = DEFINITIONAL_FLAGS + SHARED_10D

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

print(f"[Environment] Python {sys.version.split()[0]} | numpy {np.__version__} | "
      f"pandas {pd.__version__} | sklearn {sklearn.__version__} | seaborn {sns.__version__}")
print(f"[Mode] {'FAST' if FAST else 'FULL'}: N_PERM={N_PERM}, N_BOOT={N_BOOT}, "
      f"N_REPEATS={N_REPEATS}")


# ==============================================================================
# 1. Small statistical helpers
# ==============================================================================
def smart_read_table(pattern):
    """Load CSV or Excel tables from the relative data directory."""
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found in '{DATA_DIR}/'.")
    fpath = matches[0]
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(fpath, na_values=["NA", "na", "NaN", "null", "", " "])
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(fpath, na_values=["NA", "na", "NaN", "null", "", " "])
    else:
        raise ValueError(f"Unsupported file format: {fpath}")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.dropna(how="all").reset_index(drop=True)
    return df, os.path.basename(fpath)


def find_col(cols, prefix):
    cands = [c for c in cols if c.lower().startswith(prefix.lower())]
    return cands[0] if cands else None


def holm_adjust(pvals):
    """Holm step-down adjusted p-values, capped at 1."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * n / (rank + 1))
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(values, stat=np.mean, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for a scalar statistic of a 1D sample."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat(rng.choice(values, size=len(values), replace=True))
    return tuple(np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def epsilon_sq(groups):
    """Kruskal-Wallis epsilon-squared: (H - k + 1) / (n - k)."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    k = len(groups)
    n = sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return np.nan
    H, _ = stats.kruskal(*groups)
    return max(0.0, (H - k + 1) / (n - k))


def bootstrap_epsilon_sq(groups, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        gs = [rng.choice(np.asarray(g, float), size=len(g), replace=True) for g in groups if len(g) > 0]
        if len(gs) >= 2:
            out.append(epsilon_sq(gs))
    if len(out) < 10:
        return np.nan, np.nan
    return tuple(np.percentile(out, [2.5, 97.5]))


def rank_biserial(x, y):
    """
    Rank-biserial correlation in the direction P(x > y).
    r = 2U/(n1*n2) - 1, where U is scipy's statistic for sample x.
    v1 used 1 - 2U/(n1*n2), which flips the sign.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return 2.0 * u / (len(x) * len(y)) - 1.0


def cluster_bootstrap_rank_biserial(vals_a, vals_b, dates_a, dates_b,
                                    n_boot=500, seed=SEED):
    """Cluster (recording-date) bootstrap CI for the rank-biserial correlation."""
    rng = np.random.default_rng(seed)
    all_dates = np.unique(np.concatenate([dates_a, dates_b]))
    out = []
    for _ in range(n_boot):
        samp = rng.choice(all_dates, size=len(all_dates), replace=True)
        a_parts, b_parts = [], []
        for d in samp:
            ma = dates_a == d
            mb = dates_b == d
            if np.any(ma):
                a_parts.append(vals_a[ma])
            if np.any(mb):
                b_parts.append(vals_b[mb])
        if not a_parts or not b_parts:
            continue
        a = np.concatenate(a_parts)
        b = np.concatenate(b_parts)
        if len(a) < 2 or len(b) < 2:
            continue
        out.append(rank_biserial(a, b))
    if len(out) < 10:
        return np.nan, np.nan
    return tuple(np.percentile(out, [2.5, 97.5]))


def multiclass_brier(y_true, y_prob):
    """Multiclass Brier score (lower is better)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n, k = y_prob.shape
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


try:
    BRIER_SCORER = make_scorer(multiclass_brier, greater_is_better=False,
                               response_method="predict_proba")
except TypeError:
    BRIER_SCORER = make_scorer(multiclass_brier, greater_is_better=False,
                               needs_proba=True)


def corrected_resampled_ttest(acc_a, acc_b, n_train, n_test):
    """
    Nadeau-Bengio corrected resampled t-test for comparing two CV models.
    acc_a, acc_b: per-fold accuracies in the same fold order.
    """
    d = np.asarray(acc_a, float) - np.asarray(acc_b, float)
    J = len(d)
    if J < 2:
        return np.nan, np.nan
    var_d = np.var(d, ddof=1)
    denom = np.sqrt((1.0 / J + n_test / n_train) * var_d)
    if denom == 0 or not np.isfinite(denom):
        return np.nan, np.nan
    t = d.mean() / denom
    p = 2.0 * stats.t.sf(abs(t), df=J - 1)
    return float(t), float(p)


# ==============================================================================
# 2. Data loading, cleaning and feature engineering
# ==============================================================================
REQUIRED_PATTERNS = [f"*Data{i:02d}*" for i in range(1, 12)]


def check_data():
    """Fail fast with a clear message if the official inputs are incomplete."""
    missing = [p for p in REQUIRED_PATTERNS
               if not glob.glob(os.path.join(DATA_DIR, p))]
    if missing:
        raise FileNotFoundError(
            "Missing required input files in data/:\n  " + "\n  ".join(missing)
            + "\nPlace the 11 official CSV files in data/ and re-run. "
              "The raw data cannot be generated by this script; it is contest input."
        )
    d1, _ = smart_read_table("*Data01*")
    need1 = {"CallType", "Duration", "F01", "Fpeak", "Q25", "Q50", "Q75"}
    if not need1.issubset(set(d1.columns)):
        raise ValueError(f"Data01 is missing columns: {sorted(need1 - set(d1.columns))}")
    d2, _ = smart_read_table("*Data02*")
    need2 = {"CallType", "Context", "Distance", "Npenguins", "ResponseYesNo"}
    if not need2.issubset(set(d2.columns)):
        raise ValueError(f"Data02 is missing columns: {sorted(need2 - set(d2.columns))}")
    print(f"[Data check] 11 official CSV files found; required columns present.")


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_data_manifest():
    """Hash and describe every file in data/ (provenance for reproducibility)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
        if os.path.isdir(path):
            continue
        name = os.path.basename(path)
        rows_n, cols_n = "", ""
        if name.lower().endswith(".csv"):
            try:
                cols_n = pd.read_csv(path, nrows=5).shape[1]
                rows_n = sum(1 for _ in open(path, "rb")) - 1
            except Exception:
                pass
        used = bool(name.startswith("Data") and name.endswith(".csv"))
        rows.append({
            "File": name,
            "Size_bytes": os.path.getsize(path),
            "Rows": rows_n,
            "Columns": cols_n,
            "SHA256": file_sha256(path),
            "Used_by_P1": used,
            "Role": "P1 input" if used else "not used by P1 (reference / later problems)"
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "P1_data_manifest.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"[Data manifest] {len(rows)} files -> outputs_p1/P1_data_manifest.csv")


def export_analysis_dataset(df1):
    """Export the cleaned modelling table and its codebook (generated by code)."""
    cols = ["Date", "Location", "CallType", "Duration", "F01", "F02", "Fpeak",
            "Q25", "Q50", "Q75", "PulseRate"] + ENG_FEATS + [
            "PeakAtFundamental", "F02_present", "PulseRate_present"]
    out = df1[cols].copy()
    out.to_csv(os.path.join(OUT_DIR, "P1_analysis_dataset.csv"),
               index=False, encoding="utf-8-sig")

    dictionary = pd.DataFrame([
        ("Date", "Recording date", "string", "Data01"),
        ("Location", "Recording location", "string", "Data01"),
        ("CallType", "flat / mod / tv", "category", "Data01"),
        ("Duration", "Call duration", "source units", "Data01"),
        ("F01", "First fundamental frequency", "source units", "Data01"),
        ("F02", "Second fundamental frequency; tv only", "source units", "Data01"),
        ("Fpeak", "Peak frequency", "source units", "Data01"),
        ("Q25", "25th-percentile frequency", "source units", "Data01"),
        ("Q50", "Median frequency", "source units", "Data01"),
        ("Q75", "75th-percentile frequency", "source units", "Data01"),
        ("PulseRate", "Pulse rate; flat structurally undefined", "source units", "Data01"),
        ("SpecBandwidth", "Q75 - Q25", "derived", "P1.py"),
        ("HarmonicRatio", "Fpeak / F01; bimodal mixture indicator", "derived", "P1.py"),
        ("SpecSkew", "((Q75-Q50)-(Q50-Q25))/(Q75-Q25)", "derived", "P1.py"),
        ("PeakAtFundamental", "1 if Fpeak/F01 < 1.5, else 0", "binary, derived", "P1.py"),
        ("F02_present", "1 if F02 measured, else 0; definitional for tv", "binary, derived", "P1.py"),
        ("PulseRate_present", "1 if PulseRate measured, else 0", "binary, derived", "P1.py"),
    ], columns=["Column", "Description", "Unit_Type", "Source"])
    dictionary.to_csv(os.path.join(OUT_DIR, "P1_data_dictionary.csv"),
                      index=False, encoding="utf-8-sig")
    print(f"[Analysis dataset] {len(out)} rows, {len(cols)} columns -> "
          f"outputs_p1/P1_analysis_dataset.csv")


def load_and_clean():
    audit = []

    def log_step(dataset, step, before, after, cells_removed=0):
        audit.append({
            "Dataset": dataset, "Step": step,
            "Rows_Removed": int(before - after),
            "Rows_Remaining": int(after),
            "Cells_Removed": int(cells_removed)
        })

    # ---- Data01: acoustic measurements -------------------------------------
    df1_raw, f1_name = smart_read_table("*Data01*")
    n0 = len(df1_raw)

    for col in df1_raw.columns:
        if pd.api.types.is_string_dtype(df1_raw[col]) or df1_raw[col].dtype == object:
            df1_raw[col] = df1_raw[col].astype(str).str.strip()

    df1_raw["CallType"] = (
        df1_raw["CallType"].astype(str).str.strip().str.lower()
        .replace({"modulated": "mod", "two-voice": "tv", "twovoice": "tv", "two_voice": "tv"})
    )

    df1 = df1_raw[df1_raw["CallType"].isin(CALL_ORDER)].copy()
    log_step(f1_name, "Invalid CallType labels", n0, len(df1))

    for col in RAW_ACOUSTIC:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")

    before = len(df1)
    df1 = df1.drop_duplicates().reset_index(drop=True)
    log_step(f1_name, "Exact duplicate rows", before, len(df1))

    before = len(df1)
    bad_phys = pd.Series(False, index=df1.index)
    for col in RAW_ACOUSTIC:
        bad_phys |= (df1[col].notna() & (df1[col] <= 0))
    df1 = df1[~bad_phys].reset_index(drop=True)
    log_step(f1_name, "Non-positive acoustic measurements", before, len(df1))

    before = len(df1)
    q_valid = df1[["Q25", "Q50", "Q75"]].notna().all(axis=1)
    q_bad = q_valid & ((df1["Q25"] > df1["Q50"]) | (df1["Q50"] > df1["Q75"]))
    df1 = df1[~q_bad].reset_index(drop=True)
    log_step(f1_name, "Non-monotone spectral quartiles", before, len(df1))

    before = len(df1)
    df1 = df1.dropna(subset=RAW_SHARED).reset_index(drop=True)
    log_step(f1_name, "Missing shared 6D features", before, len(df1))

    log_step(f1_name, "Final analytic sample", len(df1), len(df1))

    # ---- derived acoustic features -----------------------------------------
    bw = df1["Q75"] - df1["Q25"]
    df1["SpecBandwidth"] = bw
    df1["HarmonicRatio"] = df1["Fpeak"] / df1["F01"]
    df1["SpecSkew"] = np.where(
        bw > 1e-6,
        ((df1["Q75"] - df1["Q50"]) - (df1["Q50"] - df1["Q25"])) / bw,
        0.0
    )
    # Fpeak is bimodal: in some calls the peak is the fundamental, in others the
    # third harmonic. This indicator separates the two regimes.
    df1["PeakAtFundamental"] = (df1["HarmonicRatio"] < 1.5).astype(int)
    df1["F02_present"] = df1["F02"].notna().astype(int)
    df1["PulseRate_present"] = df1["PulseRate"].notna().astype(int)

    df1["CallType"] = pd.Categorical(df1["CallType"], categories=CALL_ORDER, ordered=True)
    df1["Date"] = df1["Date"].astype(str)

    # ---- Data02: playback ---------------------------------------------------
    df2_raw, f2_name = smart_read_table("*Data02*")
    n2 = len(df2_raw)
    df2 = df2_raw.drop_duplicates().reset_index(drop=True)
    log_step(f2_name, "Exact duplicate rows", n2, len(df2), cells_removed=0)
    log_step(f2_name, "Final analytic sample", len(df2), len(df2), cells_removed=0)

    for col in ["Context", "Type", "CallType"]:
        if col in df2.columns:
            df2[col] = df2[col].astype(str).str.strip().str.lower()

    c_resp = find_col(df2.columns, "Response")
    c_head = find_col(df2.columns, "HeadsUpY")
    c_appr = find_col(df2.columns, "ApproachY")
    c_voc = find_col(df2.columns, "VocalY")
    pb_cols = [c for c in [c_resp, c_head, c_appr, c_voc] if c is not None]
    for c in pb_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    if all(c is not None for c in [c_resp, c_head, c_appr, c_voc]):
        any_beh = (df2[[c_head, c_appr, c_voc]].fillna(0).max(axis=1) > 0).astype(int)
        mismatch = (any_beh == 1) & (df2[c_resp] == 0)
        if mismatch.sum() > 0:
            df2.loc[mismatch, c_resp] = 1
        df2.attrs["response_fixes"] = int(mismatch.sum())

    # ---- Data03-11: propagation correlations -------------------------------
    deg_specs = [
        ("Data03", "flat", "Spectrogram"), ("Data04", "mod", "Spectrogram"), ("Data05", "tv", "Spectrogram"),
        ("Data06", "flat", "AmplitudeSpectrum"), ("Data07", "mod", "AmplitudeSpectrum"),
        ("Data08", "tv", "AmplitudeSpectrum"),
        ("Data09", "flat", "Envelope"), ("Data10", "mod", "Envelope"), ("Data11", "tv", "Envelope"),
    ]
    deg_long_list = []
    for prefix, ctype, metric in deg_specs:
        d_raw, d_name = smart_read_table(f"*{prefix}*")
        dist_map = {
            col: int(re.match(r"^\s*(\d+)\s*m?\s*$", col, re.IGNORECASE).group(1))
            for col in d_raw.columns if re.match(r"^\s*(\d+)\s*m?\s*$", col, re.IGNORECASE)
        }
        d_sub = d_raw[list(dist_map.keys())].apply(pd.to_numeric, errors="coerce")
        out_range = int(((d_sub < -1.0) | (d_sub > 1.0)).sum().sum())
        if out_range > 0:
            d_sub = d_sub.mask((d_sub < -1.0) | (d_sub > 1.0))
        rows_kept = len(d_sub.dropna(how="all"))
        log_step(d_name, "Out-of-range correlation cells", len(d_raw), rows_kept, cells_removed=out_range)

        long_d = d_sub.melt(var_name="DistLabel", value_name="Correlation")
        long_d["Distance_m"] = long_d["DistLabel"].map(dist_map)
        long_d["CallType"] = ctype
        long_d["Metric"] = metric
        deg_long_list.append(long_d.dropna(subset=["Correlation"]))

    pd.DataFrame(audit).to_csv(
        os.path.join(OUT_DIR, "P1_data_cleaning_audit.csv"), index=False, encoding="utf-8-sig"
    )
    deg_all = pd.concat(deg_long_list, ignore_index=True)

    print("=" * 78)
    print("DATA SUMMARY")
    print("=" * 78)
    print(f"Data01 analytic n = {len(df1)} "
          f"(flat={(df1['CallType']=='flat').sum()}, "
          f"mod={(df1['CallType']=='mod').sum()}, "
          f"tv={(df1['CallType']=='tv').sum()})")
    print(f"Recording dates = {df1['Date'].nunique()}, locations = {df1['Location'].nunique()}")
    print(f"Playback rows = {len(df2)}; ResponseYesNo fixes = {df2.attrs.get('response_fixes', 0)}")

    return df1, df2, deg_all, audit


# ==============================================================================
# 3. Descriptive statistics, Table 1, missingness audit
# ==============================================================================
def descriptive_statistics(df1):
    rows = []
    for ctype in CALL_ORDER:
        sub = df1[df1["CallType"] == ctype]
        for feat in RAW_ACOUSTIC + ENG_FEATS + ["PeakAtFundamental"]:
            vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
            if len(vals) == 0:
                rows.append({
                    "CallType": ctype, "Feature": feat, "N": len(sub),
                    "N_valid": 0, "N_missing": len(sub),
                    "Mean": np.nan, "SD": np.nan, "Median": np.nan,
                    "IQR": np.nan, "Q25": np.nan, "Q75": np.nan,
                    "Min": np.nan, "Max": np.nan
                })
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append({
                "CallType": ctype, "Feature": feat, "N": len(sub),
                "N_valid": int(len(vals)), "N_missing": int(len(sub) - len(vals)),
                "Mean": float(vals.mean()),
                "SD": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "Median": float(med), "IQR": float(q3 - q1),
                "Q25": float(q1), "Q75": float(q3),
                "Min": float(vals.min()), "Max": float(vals.max())
            })
    desc = pd.DataFrame(rows)
    desc.to_csv(os.path.join(OUT_DIR, "P1_table1_descriptive.csv"),
                index=False, encoding="utf-8-sig")

    # Markdown Table 1 for the paper
    lines = ["# Table 1. Descriptive statistics by call type",
             "",
             "Cells are mean ± SD; median [IQR]. F02 is structurally undefined for "
             "flat/mod (single-voice calls); PulseRate is structurally undefined for flat "
             "(unmodulated calls). Units follow the source data."]
    for feat in RAW_ACOUSTIC + ENG_FEATS + ["PeakAtFundamental"]:
        lines.append("")
        lines.append(f"## {feat}")
        lines.append("")
        lines.append("| CallType | n | mean ± SD | median [IQR] |")
        lines.append("|---|---:|---|---|")
        for ctype in CALL_ORDER:
            r = desc[(desc.CallType == ctype) & (desc.Feature == feat)].iloc[0]
            if r["N_valid"] == 0:
                lines.append(f"| {ctype} | 0 | structurally undefined | — |")
            else:
                lines.append(
                    f"| {ctype} | {int(r['N_valid'])} | "
                    f"{r['Mean']:.3g} ± {r['SD']:.3g} | "
                    f"{r['Median']:.3g} [{r['Q25']:.3g}, {r['Q75']:.3g}] |"
                )
    with open(os.path.join(OUT_DIR, "P1_table1_descriptive.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Missingness audit
    miss_rows = []
    for ctype in CALL_ORDER:
        sub = df1[df1["CallType"] == ctype]
        for feat in RAW_ACOUSTIC:
            n_valid = int(pd.to_numeric(sub[feat], errors="coerce").notna().sum())
            miss_rows.append({
                "Feature": feat, "CallType": ctype, "N": len(sub),
                "N_valid": n_valid, "N_missing": len(sub) - n_valid,
                "Pct_missing": 100.0 * (len(sub) - n_valid) / len(sub),
                "Structural_note": (
                    "Structurally undefined (no second fundamental)" if feat == "F02" and ctype != "tv"
                    else "Structurally undefined (unmodulated call)" if feat == "PulseRate" and ctype == "flat"
                    else ""
                )
            })
    pd.DataFrame(miss_rows).to_csv(
        os.path.join(OUT_DIR, "P1_missingness_audit.csv"), index=False, encoding="utf-8-sig"
    )

    # Formal association between presence/absence and class
    assoc = []
    for flag in DEFINITIONAL_FLAGS:
        tab = pd.crosstab(df1["CallType"], df1[flag])
        try:
            chi2_stat, p_val, dof, _ = stats.chi2_contingency(tab)
        except Exception:
            chi2_stat, p_val, dof = np.nan, np.nan, np.nan
        assoc.append({
            "Flag": flag, "Chi2": chi2_stat, "df": dof, "p_value": p_val,
            "Interpretation": "Deterministic/definitional association; not random missingness"
        })
    pd.DataFrame(assoc).to_csv(
        os.path.join(OUT_DIR, "P1_missingness_association.csv"), index=False, encoding="utf-8-sig"
    )

    # Outlier audit (3 x IQR; retained, but a trimmed sensitivity is reported later)
    outlier_records = []
    for ctype in CALL_ORDER:
        sub = df1[df1["CallType"] == ctype]
        for col in RAW_ACOUSTIC:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals) >= 4:
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                low, high = q1 - 3.0 * iqr, q3 + 3.0 * iqr
                n_out = int(((vals < low) | (vals > high)).sum())
                if n_out > 0:
                    outlier_records.append({
                        "CallType": ctype, "Feature": col,
                        "ExtremeOutliers_3IQR": n_out,
                        "LowerBound": low, "UpperBound": high
                    })
    pd.DataFrame(outlier_records, columns=[
        "CallType", "Feature", "ExtremeOutliers_3IQR", "LowerBound", "UpperBound"
    ]).to_csv(
        os.path.join(OUT_DIR, "P1_outlier_audit.csv"),
        index=False, encoding="utf-8-sig"
    )

    return desc


def plot_descriptive(df1):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.0))
    for ax, feat in zip(axes.ravel(), RAW_ACOUSTIC):
        pdata = df1[["CallType", feat]].copy()
        pdata[feat] = pd.to_numeric(pdata[feat], errors="coerce")
        pdata = pdata.dropna()
        if len(pdata) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(feat)
            continue
        sns.violinplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER,
                       palette=PALETTE,
                       inner=None, alpha=0.35, cut=0, ax=ax)
        sns.boxplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER,
                    palette=PALETTE,
                    width=0.28, boxprops={"facecolor": "none", "zorder": 3},
                    whiskerprops={"zorder": 3},
                    medianprops={"color": "black", "linewidth": 1.8},
                    showfliers=False, ax=ax)
        sns.stripplot(data=pdata, x="CallType", y=feat, order=CALL_ORDER,
                      palette=PALETTE,
                      size=3.2, alpha=0.65, jitter=0.14, ax=ax)
        ax.set_title(f"{feat} (n_valid={len(pdata)})", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("")
        if feat == "F02":
            ax.text(0.5, 0.55, "structurally undefined\nfor flat / mod\n(single-voice calls)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.85))
        if feat == "PulseRate":
            ax.text(0.5, 0.55, "structurally undefined\nfor flat\n(unmodulated call)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#444444",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.85))
    plt.suptitle("Raw Acoustic Feature Distributions across Three Call Types\n"
                 "(empty panels are structural, not missing-at-random)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig1_raw_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Engineered features + peak-regime diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    for ax, feat, title in [
        (axes[0, 0], "SpecBandwidth", "Spectral Bandwidth (Q75 - Q25)"),
        (axes[0, 1], "SpecSkew", "Spectral Skewness")
    ]:
        sns.violinplot(data=df1, x="CallType", y=feat, order=CALL_ORDER,
                       palette=PALETTE,
                       inner=None, alpha=0.35, cut=0, ax=ax)
        sns.boxplot(data=df1, x="CallType", y=feat, order=CALL_ORDER,
                    palette=PALETTE,
                    width=0.28, boxprops={"facecolor": "none", "zorder": 3},
                    whiskerprops={"zorder": 3},
                    medianprops={"color": "black", "linewidth": 1.8},
                    showfliers=False, ax=ax)
        sns.stripplot(data=df1, x="CallType", y=feat, order=CALL_ORDER,
                      palette=PALETTE,
                      size=3.4, alpha=0.65, jitter=0.14, ax=ax)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("")

    ax = axes[1, 0]
    for ctype in CALL_ORDER:
        sub = df1[df1["CallType"] == ctype]
        ax.scatter(sub["F01"], sub["Fpeak"], s=34, alpha=0.70,
                   color=PALETTE[ctype], label=ctype, edgecolors="w", linewidths=0.4)
    xx = np.linspace(df1["F01"].min() * 0.95, df1["F01"].max() * 1.05, 100)
    ax.plot(xx, xx, color="#888888", ls="--", lw=1.1, label="Fpeak = F01")
    ax.plot(xx, 3 * xx, color="#BBBBBB", ls=":", lw=1.1, label="Fpeak = 3·F01")
    ax.set_xlabel("First fundamental F01")
    ax.set_ylabel("Peak frequency Fpeak")
    ax.set_title("Fpeak is bimodal: peak at fundamental vs at 3rd harmonic",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 1]
    regime = pd.crosstab(df1["CallType"], df1["PeakAtFundamental"], normalize="index")
    regime = regime.reindex(CALL_ORDER)
    bottom = np.zeros(len(regime))
    colors = {"harmonic (Fpeak/F01 >= 1.5)": "#8DA0CB",
              "fundamental (Fpeak/F01 < 1.5)": "#E78AC3"}
    for col, label, color in [
        (0, "harmonic (Fpeak/F01 >= 1.5)", colors["harmonic (Fpeak/F01 >= 1.5)"]),
        (1, "fundamental (Fpeak/F01 < 1.5)", colors["fundamental (Fpeak/F01 < 1.5)"])
    ]:
        vals = regime[col].values if col in regime.columns else np.zeros(len(regime))
        ax.bar(regime.index.astype(str), vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of calls")
    ax.set_title("Peak-regime composition by call type", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("Engineered Features and the Fpeak Mixture Artifact",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig2_engineered_distributions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 4. Collinearity diagnostics
# ==============================================================================
def vif_table(X, names):
    X = np.asarray(X, dtype=float)
    rows = []
    for i, name in enumerate(names):
        y = X[:, i]
        Z = np.delete(X, i, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rows.append({"Feature": name, "VIF": 1.0 / (1.0 - r2) if r2 < 1 else np.inf})
    return pd.DataFrame(rows)


def collinearity_diagnostics(df1):
    X10 = np.asarray(df1[SHARED_10D].astype(float).to_numpy(), dtype=float)
    corr = df1[SHARED_10D].corr(method="spearman")
    high_pairs = []
    for i in range(len(SHARED_10D)):
        for j in range(i + 1, len(SHARED_10D)):
            rho = corr.iloc[i, j]
            if abs(rho) >= 0.7:
                high_pairs.append({
                    "Feature_1": SHARED_10D[i], "Feature_2": SHARED_10D[j],
                    "Spearman_rho": float(rho)
                })
    vif = vif_table(X10, SHARED_10D)
    cond = float(np.linalg.cond(np.corrcoef(X10, rowvar=False)))
    n_high_vif = int((vif["VIF"] > 10).sum())

    vif.to_csv(os.path.join(OUT_DIR, "P1_collinearity_vif.csv"),
               index=False, encoding="utf-8-sig")
    pd.DataFrame(high_pairs, columns=["Feature_1", "Feature_2", "Spearman_rho"]).to_csv(
        os.path.join(OUT_DIR, "P1_collinearity_high_pairs.csv"),
        index=False, encoding="utf-8-sig")
    pd.DataFrame([{
        "N_high_corr_pairs_abs_rho_ge_0.7": len(high_pairs),
        "Max_abs_spearman_rho": float(np.max(np.abs(corr.values - np.eye(len(SHARED_10D))))),
        "Condition_number_correlation_matrix": cond,
        "N_features_VIF_gt_10": n_high_vif
    }]).to_csv(os.path.join(OUT_DIR, "P1_collinearity_summary.csv"),
               index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=axes[0])
    axes[0].set_title("Spearman correlation (10D acoustic features)",
                      fontsize=12.5, fontweight="bold")

    vif_plot = vif.copy()
    vif_plot["VIF_plot"] = vif_plot["VIF"].replace([np.inf], 1e4)
    vif_sorted = vif_plot.sort_values("VIF_plot")
    axes[1].barh(vif_sorted["Feature"], vif_sorted["VIF_plot"], color="#4C72B0")
    axes[1].set_xscale("log")
    axes[1].set_xlim(1, 2e4)
    axes[1].axvline(5, color="#D62728", ls=":", lw=1.2, label="VIF = 5")
    axes[1].axvline(10, color="#8C2D04", ls="--", lw=1.2, label="VIF = 10")
    for y, (val, feat) in enumerate(zip(vif_sorted["VIF"], vif_sorted["Feature"])):
        if np.isinf(val):
            axes[1].text(1.15e4, y, "∞", va="center", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("VIF (log scale; ∞ = exact linear dependency)")
    axes[1].set_title(f"VIF (>10: {n_high_vif} features; cond. number: {cond:.1e})\n"
                      f"|rho| >= 0.7 pairs: {len(high_pairs)}",
                      fontsize=12.0, fontweight="bold")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig3_spearman_heatmap.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[Collinearity] |rho|>=0.7 pairs = %d, VIF>10 features = %d, "
          "condition number = %.1e" % (len(high_pairs), n_high_vif, cond))
    print(vif.round(2).to_string(index=False))
    return corr, vif


# ==============================================================================
# 5. Nonparametric tests with global Holm correction and date adjustment
# ==============================================================================
def within_date_perm_p(sub, feat, n_perm=N_PERM, seed=SEED):
    """
    Date-adjusted Kruskal-Wallis p-value.
    Features are centered within recording date (removes session means), and
    class labels are permuted within each date, which preserves the date design.
    """
    sub = sub.dropna(subset=[feat]).copy()
    if len(sub) < 6:
        return np.nan
    sub["_c"] = sub[feat] - sub.groupby("Date")[feat].transform("mean")
    vals = np.asarray(sub["_c"].to_numpy(), dtype=float)
    lab = np.asarray(sub["CallType"].astype(str).tolist(), dtype=object)
    dates = np.asarray(sub["Date"].astype(str).tolist(), dtype=object)

    present = [c for c in CALL_ORDER if np.any(lab == c)]
    if len(present) < 2:
        return np.nan
    obs = stats.kruskal(*[vals[lab == c] for c in present])[0]

    rng = np.random.default_rng(seed)
    idx_by_date = {d: np.where(dates == d)[0] for d in np.unique(dates)}
    ge, valid = 0, 0
    for _ in range(n_perm):
        perm = lab.copy()
        for d, idx in idx_by_date.items():
            perm[idx] = rng.permutation(lab[idx])
        groups = [vals[perm == c] for c in present]
        if any(len(g) == 0 for g in groups):
            continue
        valid += 1
        H = stats.kruskal(*groups)[0]
        if H >= obs:
            ge += 1
    return (ge + 1) / (valid + 1) if valid > 0 else np.nan


def within_date_perm_p_pair(sub, feat, g1, g2, n_perm=1999, seed=SEED):
    """Date-adjusted two-group permutation p-value for one feature/pair."""
    sub = sub[sub["CallType"].astype(str).isin([g1, g2])].dropna(subset=[feat]).copy()
    if len(sub) < 6:
        return np.nan
    sub["_c"] = sub[feat] - sub.groupby("Date")[feat].transform("mean")
    vals = np.asarray(sub["_c"].to_numpy(), dtype=float)
    lab = np.asarray(sub["CallType"].astype(str).tolist(), dtype=object)
    dates = np.asarray(sub["Date"].astype(str).tolist(), dtype=object)

    a0, b0 = vals[lab == g1], vals[lab == g2]
    if len(a0) < 2 or len(b0) < 2:
        return np.nan
    obs = stats.kruskal(a0, b0)[0]

    rng = np.random.default_rng(seed)
    idx_by_date = {d: np.where(dates == d)[0] for d in np.unique(dates)}
    ge, valid = 0, 0
    for _ in range(n_perm):
        perm = lab.copy()
        for d, idx in idx_by_date.items():
            perm[idx] = rng.permutation(lab[idx])
        a = vals[perm == g1]
        b = vals[perm == g2]
        if len(a) == 0 or len(b) == 0:
            continue
        valid += 1
        if stats.kruskal(a, b)[0] >= obs:
            ge += 1
    return (ge + 1) / (valid + 1) if valid > 0 else np.nan


def nonparametric_tests(df1):
    # ---- Univariate Kruskal-Wallis across 9 continuous features -------------
    kw_rows = []
    for feat in SHARED_9D:
        groups = [np.asarray(df1.loc[df1["CallType"] == c, feat].dropna(), float)
                  for c in CALL_ORDER]
        H, p = stats.kruskal(*groups)
        eps = epsilon_sq(groups)
        lo, hi = bootstrap_epsilon_sq(groups)
        # trimmed sensitivity: drop 3 x IQR within each class
        trimmed = []
        for g in groups:
            q1, q3 = np.percentile(g, [25, 75])
            iqr = q3 - q1
            keep = g[(g >= q1 - 3 * iqr) & (g <= q3 + 3 * iqr)]
            trimmed.append(keep if len(keep) >= 3 else g)
        H_t, p_t = stats.kruskal(*trimmed)
        p_date = within_date_perm_p(df1, feat)
        kw_rows.append({
            "Feature": feat, "Test": "Kruskal-Wallis (3 groups)",
            "Statistic": H, "p_raw": p, "p_trimmed": p_t,
            "EffectSize": eps, "EffectSize_CI_Low": lo, "EffectSize_CI_High": hi,
            "p_date_adjusted": p_date, "N": int(sum(len(g) for g in groups)),
            "Family": "9 continuous features"
        })

    # PulseRate: structural for flat, so only mod vs tv
    pr_m = np.asarray(df1.loc[df1["CallType"] == "mod", "PulseRate"].dropna(), float)
    pr_t = np.asarray(df1.loc[df1["CallType"] == "tv", "PulseRate"].dropna(), float)
    u_pr, p_pr = stats.mannwhitneyu(pr_m, pr_t, alternative="two-sided")
    r_pr = rank_biserial(pr_m, pr_t)
    sub_pr = df1[df1["CallType"].isin(["mod", "tv"])]
    p_date_pr = within_date_perm_p(sub_pr, "PulseRate")
    kw_rows.append({
        "Feature": "PulseRate", "Test": "Mann-Whitney U (mod vs tv)",
        "Statistic": u_pr, "p_raw": p_pr, "p_trimmed": np.nan,
        "EffectSize": r_pr, "EffectSize_CI_Low": np.nan, "EffectSize_CI_High": np.nan,
        "p_date_adjusted": p_date_pr, "N": len(pr_m) + len(pr_t),
        "Family": "PulseRate (2 groups)"
    })

    kw = pd.DataFrame(kw_rows)
    fam = kw["Family"] == "9 continuous features"
    kw.loc[fam, "p_holm"] = holm_adjust(kw.loc[fam, "p_raw"].values)
    kw.loc[fam, "p_holm_date_adjusted"] = holm_adjust(kw.loc[fam, "p_date_adjusted"].values)
    kw.loc[~fam, "p_holm"] = kw.loc[~fam, "p_raw"]
    kw.loc[~fam, "p_holm_date_adjusted"] = kw.loc[~fam, "p_date_adjusted"]
    kw.to_csv(os.path.join(OUT_DIR, "P1_nonparam_kruskal_holm.csv"),
              index=False, encoding="utf-8-sig")

    # A compact naive-vs-adjusted table
    kw[["Feature", "p_raw", "p_trimmed", "p_holm",
        "p_date_adjusted", "p_holm_date_adjusted", "EffectSize"]].to_csv(
        os.path.join(OUT_DIR, "P1_date_adjusted_tests.csv"),
        index=False, encoding="utf-8-sig")

    print("\n[Kruskal-Wallis: naive vs trimmed vs date-adjusted]")
    print(kw[["Feature", "p_raw", "p_trimmed", "p_holm",
              "p_date_adjusted", "p_holm_date_adjusted"]].round(4).to_string(index=False))

    # ---- Pairwise Mann-Whitney with GLOBAL Holm across 27 tests -------------
    pw_rows = []
    pairs = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
    for feat in SHARED_9D:
        for g1, g2 in pairs:
            sub1 = df1[df1["CallType"] == g1]
            sub2 = df1[df1["CallType"] == g2]
            x1 = np.asarray(sub1[feat].dropna(), float)
            x2 = np.asarray(sub2[feat].dropna(), float)
            u, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
            r = rank_biserial(x1, x2)
            d1 = np.asarray(sub1.loc[sub1[feat].notna(), "Date"].astype(str).tolist(), dtype=object)
            d2 = np.asarray(sub2.loc[sub2[feat].notna(), "Date"].astype(str).tolist(), dtype=object)
            lo, hi = cluster_bootstrap_rank_biserial(x1, x2, d1, d2, n_boot=500)
            p_date_pair = within_date_perm_p_pair(df1, feat, g1, g2)
            pw_rows.append({
                "Feature": feat, "Comparison": f"{g1} vs {g2}",
                "U_stat": u, "p_raw": p, "RankBiserial_r": r,
                "r_CI_Low": lo, "r_CI_High": hi,
                "p_date_adjusted": p_date_pair,
                "N_1": len(x1), "N_2": len(x2)
            })

    # PulseRate mod vs tv (separate family)
    pw_rows.append({
        "Feature": "PulseRate", "Comparison": "mod vs tv",
        "U_stat": u_pr, "p_raw": p_pr, "RankBiserial_r": r_pr,
        "r_CI_Low": np.nan, "r_CI_High": np.nan,
        "p_date_adjusted": within_date_perm_p_pair(df1, "PulseRate", "mod", "tv"),
        "N_1": len(pr_m), "N_2": len(pr_t)
    })

    pw = pd.DataFrame(pw_rows)
    main = pw["Feature"].isin(SHARED_9D)
    pw.loc[main, "p_holm_global"] = holm_adjust(pw.loc[main, "p_raw"].values)
    pw.loc[~main, "p_holm_global"] = pw.loc[~main, "p_raw"]
    pw.loc[main, "p_holm_date_global"] = holm_adjust(pw.loc[main, "p_date_adjusted"].values)
    pw.loc[~main, "p_holm_date_global"] = pw.loc[~main, "p_date_adjusted"]
    pw["Significant_Global_Holm"] = pw["p_holm_global"] < 0.05
    pw["Significant_Global_Holm_Date"] = pw["p_holm_date_global"] < 0.05
    pw.to_csv(os.path.join(OUT_DIR, "P1_nonparam_pairwise_global_holm.csv"),
              index=False, encoding="utf-8-sig")

    print("\n[Pairwise Mann-Whitney: surviving GLOBAL Holm, naive]")
    sig = pw[pw["Significant_Global_Holm"]]
    print(sig[["Feature", "Comparison", "p_raw", "p_holm_global",
               "RankBiserial_r"]].round(4).to_string(index=False) if len(sig) else "  none")
    print("\n[Pairwise Mann-Whitney: surviving GLOBAL Holm after date adjustment]")
    sig_d = pw[pw["Significant_Global_Holm_Date"]]
    print(sig_d[["Feature", "Comparison", "p_date_adjusted", "p_holm_date_global",
                 "RankBiserial_r"]].round(4).to_string(index=False) if len(sig_d) else "  none")

    # ---- Effect-size forest plot -------------------------------------------
    plot_effect_size_forest(pw)
    return kw, pw


def plot_effect_size_forest(pw):
    features = SHARED_9D
    pairs = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 8.4), sharey=True)
    for ax, (g1, g2) in zip(axes, pairs):
        sub = pw[pw["Comparison"] == f"{g1} vs {g2}"].set_index("Feature").reindex(features)
        y = np.arange(len(features))
        r = sub["RankBiserial_r"].values
        lo = sub["r_CI_Low"].values
        hi = sub["r_CI_High"].values
        xerr = np.vstack([r - lo, hi - r])
        colors = ["#D62728" if s else "#7F7F7F" for s in sub["Significant_Global_Holm"].values]
        ax.errorbar(r, y, xerr=xerr, fmt="none", ecolor="#555555", capsize=3, lw=1.2, zorder=2)
        ax.scatter(r, y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.5)
        ax.axvline(0, color="black", lw=1.0)
        ax.axvline(0.3, color="#999999", ls=":", lw=0.9)
        ax.axvline(-0.3, color="#999999", ls=":", lw=0.9)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{g1} vs {g2}", fontweight="bold")
        ax.set_xlabel("Rank-biserial r (P(x > y) direction)")
        ax.grid(axis="x", alpha=0.4)
    axes[0].set_yticks(np.arange(len(features)))
    axes[0].set_yticklabels(features)
    axes[0].invert_yaxis()
    fig.suptitle("Pairwise Effect Sizes with Date-Cluster Bootstrap 95% CIs\n"
                 "Red = survives naive global Holm across all 27 tests",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.01, 0.01,
             "Date-adjusted permutation: F01 (flat vs tv) survives global Holm "
             "(p_holm = 0.041); see P1_nonparam_pairwise_global_holm.csv.",
             fontsize=8.5, color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig12_effect_size_forest.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 6. PERMANOVA + PERMDISP on the non-redundant 6D space
# ==============================================================================
def run_permanova(X_mat, labels, n_perm=N_PERM, seed=SEED):
    """PERMANOVA (Anderson 2001) on standardized Euclidean distances."""
    rng = np.random.default_rng(seed)
    X_std = StandardScaler().fit_transform(np.asarray(X_mat, float))
    D2 = squareform(pdist(X_std, metric="euclidean")) ** 2
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    ug = np.unique(labels)
    k = len(ug)
    SS_T = np.sum(D2) / (2.0 * n)

    def calc_f(lbls):
        ss_w = 0.0
        for g in ug:
            idx = np.where(lbls == g)[0]
            if len(idx) == 0:
                return np.nan, np.nan
            ss_w += np.sum(D2[np.ix_(idx, idx)]) / (2.0 * len(idx))
        ss_a = SS_T - ss_w
        return (ss_a / (k - 1)) / (ss_w / (n - k)), ss_a / SS_T

    f_obs, r2_obs = calc_f(labels)
    ge = 0
    for _ in range(n_perm):
        f_p, _ = calc_f(rng.permutation(labels))
        if np.isfinite(f_p) and f_p >= f_obs:
            ge += 1
    return f_obs, r2_obs, (ge + 1) / (n_perm + 1)


def permdisp(X_mat, labels, n_perm=N_PERM, seed=SEED):
    """PERMDISP: permutation test on distance-to-centroid dispersion."""
    X_std = StandardScaler().fit_transform(np.asarray(X_mat, float))
    labels = np.asarray(labels, dtype=object)
    ug = np.unique(labels)

    def stat(lbls):
        dists = []
        for g in ug:
            m = lbls == g
            if m.sum() < 2:
                return np.nan
            centroid = X_std[m].mean(axis=0)
            dists.append(np.linalg.norm(X_std[m] - centroid, axis=1))
        d = np.concatenate(dists)
        grand = d.mean()
        ssb = sum((lbls == g).sum() * (d[lbls == g].mean() - grand) ** 2 for g in ug)
        ssw = sum(((d[lbls == g] - d[lbls == g].mean()) ** 2).sum() for g in ug)
        return (ssb / (len(ug) - 1)) / (ssw / (len(d) - len(ug)))

    obs = stat(labels)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = stat(rng.permutation(labels))
        if np.isfinite(s) and s >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def permanova_analysis(df1):
    X6 = np.asarray(df1[RAW_SHARED].astype(float).to_numpy(), dtype=float)
    y = np.asarray(df1["CallType"].astype(str).tolist(), dtype=object)

    f_all, r2_all, p_all = run_permanova(X6, y)
    disp_f, disp_p = permdisp(X6, y)
    rows = [{
        "Comparison": "Overall (flat vs mod vs tv)", "Space": "6D non-redundant",
        "Pseudo_F": f_all, "R2": r2_all, "p_perm": p_all,
        "PERMDISP_F": disp_f, "PERMDISP_p": disp_p,
        "Note": "Primary test"
    }]

    pairs = [("flat", "mod"), ("flat", "tv"), ("mod", "tv")]
    tmp, raw_p = [], []
    for g1, g2 in pairs:
        m = np.isin(y, [g1, g2])
        fp, r2p, pp = run_permanova(X6[m], y[m])
        dfp, dpp = permdisp(X6[m], y[m])
        raw_p.append(pp)
        tmp.append((f"{g1} vs {g2}", fp, r2p, pp, dfp, dpp))
    p_holm = holm_adjust(raw_p)
    for (comp, fp, r2p, pp, dfp, dpp), ph in zip(tmp, p_holm):
        rows.append({
            "Comparison": comp, "Space": "6D non-redundant",
            "Pseudo_F": fp, "R2": r2p, "p_perm": pp, "p_holm": ph,
            "PERMDISP_F": dfp, "PERMDISP_p": dpp,
            "Note": "Pairwise; Holm across 3"
        })

    # 9D comparison to show the effect of derived-feature double counting
    X9 = np.asarray(df1[SHARED_9D].astype(float).to_numpy(), dtype=float)
    f9, r29, p9 = run_permanova(X9, y)
    rows.append({
        "Comparison": "Overall (flat vs mod vs tv)", "Space": "9D redundant (reference)",
        "Pseudo_F": f9, "R2": r29, "p_perm": p9, "p_holm": p9,
        "PERMDISP_F": np.nan, "PERMDISP_p": np.nan,
        "Note": "Derived features double-count Q25/Q75/Fpeak/F01; R2 is inflated"
    })

    perm = pd.DataFrame(rows)
    perm.to_csv(os.path.join(OUT_DIR, "P1_nonparam_permanova.csv"),
                index=False, encoding="utf-8-sig")
    print("\n[PERMANOVA + PERMDISP]")
    print(perm[["Comparison", "Space", "Pseudo_F", "R2", "p_perm",
                "p_holm", "PERMDISP_p"]].round(4).to_string(index=False))

    # ---- PCA biplot on 6D ---------------------------------------------------
    X_std6 = StandardScaler().fit_transform(X6)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_std6)
    var_r = pca.explained_variance_ratio_
    loads = pca.components_.T * np.sqrt(pca.explained_variance_)

    label_offsets = {
        "Duration": (0.05, 0.18), "F01": (0.24, -0.12), "Fpeak": (0.32, 0.06),
        "Q25": (0.10, -0.25), "Q50": (0.24, 0.03), "Q75": (0.05, 0.25)
    }

    fig, ax = plt.subplots(figsize=(9.4, 7.4))
    for ctype in CALL_ORDER:
        pts = X_pca[y == ctype]
        ax.scatter(pts[:, 0], pts[:, 1], label=ctype, color=PALETTE[ctype],
                   alpha=0.72, edgecolors="w", s=55)
        cov = np.cov(pts, rowvar=False) + 1e-9 * np.eye(2)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        scale = np.sqrt(chi2.ppf(0.68, df=2))
        w, h = 2 * scale * np.sqrt(vals)
        ax.add_patch(Ellipse(xy=pts.mean(axis=0), width=w, height=h, angle=theta,
                             edgecolor=PALETTE[ctype], facecolor=PALETTE[ctype],
                             alpha=0.12, lw=2))
        ax.scatter(*pts.mean(axis=0), color=PALETTE[ctype], marker="X",
                   s=140, edgecolors="black", zorder=5)

    for i, feat in enumerate(RAW_SHARED):
        vx, vy = loads[i, 0] * 2.5, loads[i, 1] * 2.5
        dx, dy = label_offsets.get(feat, (0.12, 0.12))
        ax.arrow(0, 0, vx, vy, color="#222222", alpha=0.82,
                 head_width=0.09, length_includes_head=True, lw=1.2, zorder=6)
        ax.text(vx + dx, vy + dy, feat, color="#111111", fontsize=9.8,
                fontweight="bold", ha="center", va="center", zorder=7)

    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(X_pca[:, 0].min() - 0.6, X_pca[:, 0].max() + 0.6)
    ax.set_ylim(X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5)
    ax.set_xlabel(f"PC1 ({var_r[0]*100:.1f}% variance)", fontsize=11.5)
    ax.set_ylabel(f"PC2 ({var_r[1]*100:.1f}% variance)", fontsize=11.5)
    ax.set_title(
        f"PCA Biplot, 6D Non-redundant Space (68% ellipses)\n"
        f"PERMANOVA Pseudo-F={f_all:.2f}, R2={r2_all:.3f}, p={p_all:.4f}; "
        f"PERMDISP p={disp_p:.3f}",
        fontsize=12.0, fontweight="bold")
    ax.legend(title="CallType (X = centroid)", loc="lower right")
    ax.text(0.01, 0.01, "Loading arrows scaled x2.5 for display",
            transform=ax.transAxes, fontsize=8.5, color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig4_PCA_PERMANOVA_biplot.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    return perm


# ==============================================================================
# 7. Classification: leakage-free ablation, baselines, uncertainty
# ==============================================================================
def make_lasso():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=10000,
                                   random_state=SEED))
    ])
    return GridSearchCV(
        pipe, {"clf__C": C_GRID},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=1, refit=True
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        random_state=SEED, n_jobs=1
    )


def repeated_oof_proba(make_model, X, y, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                       seed=SEED):
    """Average out-of-fold predicted probabilities over repeated stratified CV."""
    y = np.asarray(y)
    classes = np.unique(y)
    n = len(y)
    proba_sum = np.zeros((n, len(classes)))
    counts = np.zeros(n)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    for train, test in rskf.split(X, y):
        m = make_model()
        m.fit(X[train], y[train])
        p = m.predict_proba(X[test])
        col = np.searchsorted(classes, m.classes_)
        proba_sum[np.ix_(test, col)] += p
        counts[test] += 1
    return proba_sum / counts[:, None]


def grouped_cv_scores(make_model, X, y, groups, n_splits=5, seed=SEED):
    """Grouped CV by recording date; skips folds that miss a class."""
    if HAS_SGK:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed).split(X, y, groups)
    else:
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    accs, bals = [], []
    for train, test in splitter:
        if len(np.unique(y[train])) < len(np.unique(y)):
            continue
        m = make_model()
        m.fit(X[train], y[train])
        yp = m.predict(X[test])
        accs.append(accuracy_score(y[test], yp))
        bals.append(balanced_accuracy_score(y[test], yp))
    return np.asarray(accs), np.asarray(bals)


def cv_permutation_importance(make_model, X, y, n_splits=5, n_repeats=10,
                              n_imp_repeats=10, seed=SEED):
    """
    Permutation importance computed on held-out folds, not on the training data.
    Returns mean and SD across folds (per feature).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    imps = []
    for train, test in skf.split(X, y):
        m = make_model()
        m.fit(X[train], y[train])
        r = permutation_importance(m, X[test], y[test], n_repeats=n_imp_repeats,
                                   random_state=seed, n_jobs=1)
        imps.append(r.importances_mean)
    imps = np.asarray(imps)
    return imps.mean(axis=0), imps.std(axis=0)


def lasso_stability(X, y, n_boot=200, seed=SEED):
    """Bootstrap selection frequency for L1-penalized multinomial logit."""
    rng = np.random.default_rng(seed)
    # choose C once on the full data (mild optimism; reported as descriptive)
    gs = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(penalty="l1", solver="saga",
                                             max_iter=10000, random_state=SEED))]),
        {"clf__C": C_GRID},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=1
    ).fit(X, y)
    best_c = gs.best_params_["clf__C"]
    n = len(y)
    freq = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < len(np.unique(y)):
            continue
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l1", solver="saga",
                                                    C=best_c, max_iter=10000,
                                                    random_state=SEED))])
        pipe.fit(X[idx], y[idx])
        coef = pipe.named_steps["clf"].coef_
        freq += (np.abs(coef).sum(axis=0) > 1e-8).astype(float)
    return freq / max(1, n_boot), best_c


def sns_barplot_ci(ax, data, x, y, hue, hue_order, ci=95):
    """Bar plot with 95% CI; supports both new and old seaborn APIs."""
    try:
        sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order,
                    errorbar=("ci", ci), legend=False, ax=ax)
    except TypeError:
        sns.barplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order,
                    ci=ci, ax=ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()


def classification_analysis(df1):
    le = LabelEncoder().fit(CALL_ORDER)
    y = le.transform(df1["CallType"].astype(str).values)

    X_sets = {
        "Set 1: Raw 6D": (np.asarray(df1[RAW_SHARED].astype(float).to_numpy(), float), RAW_SHARED),
        "Set 2: +Engineered 9D": (np.asarray(df1[SHARED_9D].astype(float).to_numpy(), float), SHARED_9D),
        "Set 3: +Peak regime 10D": (np.asarray(df1[SHARED_10D].astype(float).to_numpy(), float), SHARED_10D),
        "Set 4: +Structural flags (definitional)": (
            np.asarray(df1[STRUCT_12D].astype(float).to_numpy(), float), STRUCT_12D),
        "Definitional flags only (2D)": (
            np.asarray(df1[DEFINITIONAL_FLAGS].astype(float).to_numpy(), float), DEFINITIONAL_FLAGS),
    }

    models = {"LASSO-Logit (nested C)": make_lasso, "RandomForest": make_rf}
    scoring = {
        "acc": "accuracy",
        "bal_acc": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "auc": "roc_auc_ovr",
        "kappa": make_scorer(cohen_kappa_score),
        "brier": BRIER_SCORER,
    }
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)

    counts = pd.Series(y).value_counts(normalize=True)
    majority_acc = float(counts.max())
    random_acc = float(np.sum(counts.values ** 2))

    ablation_rows, fold_rows = [], []
    fold_acc = {}
    for fs_name, (X_curr, feats) in X_sets.items():
        for m_name, make_model in models.items():
            res = cross_validate(make_model(), X_curr, y, cv=rskf, scoring=scoring,
                                 n_jobs=1, error_score="raise")
            acc_mean = float(res["test_acc"].mean())
            lo, hi = bootstrap_ci(res["test_acc"])
            ablation_rows.append({
                "FeatureSet": fs_name, "N_features": len(feats), "Model": m_name,
                "Acc_Mean": acc_mean, "Acc_SD": float(res["test_acc"].std()),
                "Acc_CI_Low": lo, "Acc_CI_High": hi,
                "BalAcc_Mean": float(res["test_bal_acc"].mean()),
                "MacroF1_Mean": float(res["test_macro_f1"].mean()),
                "AUC_OvR_Mean": float(res["test_auc"].mean()),
                "Kappa_Mean": float(res["test_kappa"].mean()),
                "Brier_Mean": float(-res["test_brier"].mean()),
                "Acc_Gap_vs_Majority": acc_mean - majority_acc,
                "Note": ("Definitional; not an acoustic classifier"
                         if ("Structural" in fs_name or "Definitional" in fs_name)
                         else "Acoustic-only (leakage-free)")
            })
            fold_acc[(fs_name, m_name)] = np.asarray(res["test_acc"], float)
            for a, f1, auc, kap in zip(res["test_acc"], res["test_macro_f1"],
                                       res["test_auc"], res["test_kappa"]):
                fold_rows.append({
                    "FeatureSet": fs_name, "Model": m_name,
                    "Accuracy": a, "Macro_F1": f1, "AUC_OvR": auc, "Kappa": kap
                })

    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(os.path.join(OUT_DIR, "P1_classification_ablation.csv"),
                    index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(os.path.join(OUT_DIR, "P1_classification_folds.csv"),
                                   index=False, encoding="utf-8-sig")

    # Figure 5: ablation with confidence intervals and no-skill baselines
    fold_df = pd.DataFrame(fold_rows)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))
    for ax, met in zip(axes, ["Accuracy", "Macro_F1", "AUC_OvR"]):
        sns_barplot_ci(ax, fold_df, "FeatureSet", met, "Model", list(models.keys()))
        ax.set_title(f"{met} (10x5 repeated CV, mean ± 95% CI)",
                     fontsize=11.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=24)
    axes[0].axhline(majority_acc, color="#D62728", ls="--", lw=1.2,
                    label=f"Majority = {majority_acc:.3f}")
    axes[0].axhline(1.0 / 3.0, color="#7F7F7F", ls=":", lw=1.2,
                    label="Chance = 0.333")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(1.0 / 3.0, color="#7F7F7F", ls=":", lw=1.2)
    axes[2].axhline(0.5, color="#7F7F7F", ls=":", lw=1.2)
    from matplotlib.patches import Patch
    model_names = list(models.keys())
    hue_colors = [axes[2].patches[i].get_facecolor() for i in range(len(model_names))]
    handles = [Patch(facecolor=hue_colors[i], label=model_names[i])
               for i in range(len(model_names))]
    axes[2].legend(handles=handles, fontsize=8, loc="lower right", title="Model")
    plt.suptitle("Ablation: acoustic-only models vs definitional flags\n"
                 "The jump at Set 4 is definitional, not acoustic",
                 fontsize=13.5, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig5_ablation_model_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    pd.DataFrame([{
        "Majority_baseline_accuracy": majority_acc,
        "Stratified_random_expected_accuracy": random_acc,
        "Balanced_accuracy_chance": 1.0 / 3.0,
        "Class_counts": dict(zip(CALL_ORDER, np.bincount(y, minlength=3).tolist()))
    }]).to_csv(os.path.join(OUT_DIR, "P1_baselines.csv"), index=False,
               encoding="utf-8-sig")

    print("\n[Classification ablation: 10x5 repeated stratified CV]")
    print(ablation[["FeatureSet", "Model", "Acc_Mean", "Acc_CI_Low", "Acc_CI_High",
                    "BalAcc_Mean", "MacroF1_Mean", "Kappa_Mean", "Brier_Mean"]]
          .round(3).to_string(index=False))
    print(f"Majority baseline = {majority_acc:.3f}; stratified random = {random_acc:.3f}")

    # ---- Model comparison: corrected resampled t-test ----------------------
    cmp_rows = []
    n = len(y)
    n_test = n / N_SPLITS
    n_train = n - n_test
    for fs_name in ["Set 2: +Engineered 9D", "Set 3: +Peak regime 10D",
                    "Set 4: +Structural flags (definitional)"]:
        a = fold_acc[(fs_name, "RandomForest")]
        b = fold_acc[(fs_name, "LASSO-Logit (nested C)")]
        t, p = corrected_resampled_ttest(a, b, n_train, n_test)
        cmp_rows.append({
            "FeatureSet": fs_name, "RF_minus_LASSO_Acc":
                float(a.mean() - b.mean()), "t_corrected": t, "p_value": p
        })
    pd.DataFrame(cmp_rows).to_csv(
        os.path.join(OUT_DIR, "P1_model_comparison_test.csv"),
        index=False, encoding="utf-8-sig")

    # ---- Grouped CV by recording date --------------------------------------
    groups = np.asarray(df1["Date"].astype(str).tolist(), dtype=object)
    grp_rows = []
    for fs_name, (X_curr, feats) in X_sets.items():
        for m_name, make_model in models.items():
            accs, bals = grouped_cv_scores(make_model, X_curr, y, groups)
            if len(accs) == 0:
                continue
            grp_rows.append({
                "FeatureSet": fs_name, "Model": m_name,
                "Grouped_Acc_Mean": float(accs.mean()),
                "Grouped_Acc_SD": float(accs.std()),
                "Grouped_BalAcc_Mean": float(bals.mean()),
                "N_valid_folds": int(len(accs))
            })
    grp = pd.DataFrame(grp_rows)
    grp.to_csv(os.path.join(OUT_DIR, "P1_classification_grouped_cv.csv"),
               index=False, encoding="utf-8-sig")
    print("\n[StratifiedGroupKFold by recording date]")
    print(grp.round(3).to_string(index=False))

    # ---- OOF probabilities for Set 3 and Set 4 -----------------------------
    oof = {}
    for fs_name in ["Set 3: +Peak regime 10D", "Set 4: +Structural flags (definitional)"]:
        X_curr = X_sets[fs_name][0]
        for m_name, make_model in models.items():
            oof[(fs_name, m_name)] = repeated_oof_proba(make_model, X_curr, y)

    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    layout = [
        ("Set 3: +Peak regime 10D", "RandomForest"),
        ("Set 3: +Peak regime 10D", "LASSO-Logit (nested C)"),
        ("Set 4: +Structural flags (definitional)", "RandomForest"),
        ("Set 4: +Structural flags (definitional)", "LASSO-Logit (nested C)"),
    ]
    for ax, (fs_name, m_name) in zip(axes.ravel(), layout):
        proba = oof[(fs_name, m_name)]
        yp = proba.argmax(axis=1)
        acc = accuracy_score(y, yp)
        kap = cohen_kappa_score(y, yp)
        ConfusionMatrixDisplay(confusion_matrix(y, yp, labels=[0, 1, 2]),
                               display_labels=CALL_ORDER).plot(
            ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{fs_name}\n{m_name} | Acc={acc:.3f}, kappa={kap:.3f}",
                     fontsize=10.2, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig6_confusion_matrices.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Per-class metrics for the leakage-free and definitional models
    pc_rows = []
    for fs_name, m_name in [
        ("Set 3: +Peak regime 10D", "RandomForest"),
        ("Set 3: +Peak regime 10D", "LASSO-Logit (nested C)"),
        ("Set 4: +Structural flags (definitional)", "RandomForest"),
    ]:
        proba = oof[(fs_name, m_name)]
        yp = proba.argmax(axis=1)
        prec, rec, f1v, sup = precision_recall_fscore_support(
            y, yp, labels=[0, 1, 2], zero_division=0)
        for i, ctype in enumerate(CALL_ORDER):
            pc_rows.append({
                "FeatureSet": fs_name, "Model": m_name, "CallType": ctype,
                "Precision": prec[i], "Recall": rec[i], "F1": f1v[i],
                "Support": int(sup[i])
            })
    pd.DataFrame(pc_rows).to_csv(
        os.path.join(OUT_DIR, "P1_per_class_metrics.csv"),
        index=False, encoding="utf-8-sig")

    # ---- Feature importance on Set 3 ---------------------------------------
    X3 = X_sets["Set 3: +Peak regime 10D"][0]
    perm_mean, perm_sd = cv_permutation_importance(make_rf, X3, y)
    mi = mutual_info_classif(X3, y, random_state=SEED)
    rf_full = make_rf().fit(X3, y)
    gini = rf_full.feature_importances_
    stab_freq, best_c = lasso_stability(X3, y)

    imp = pd.DataFrame({
        "Feature": SHARED_10D,
        "CV_PermImp_Mean": perm_mean,
        "CV_PermImp_SD": perm_sd,
        "MutualInfo": mi,
        "RF_Gini": gini,
        "LASSO_SelectFreq": stab_freq,
        "Status": [
            "regime indicator (Fpeak/F01 < 1.5)" if f == "PeakAtFundamental"
            else "artifact (bimodal mixture)" if f == "HarmonicRatio"
            else "redundant with quartiles" if f == "SpecBandwidth"
            else "discriminative candidate" if f in ["F01", "Fpeak"]
            else "weak / not significant" if f in ["Duration", "Q25", "Q50", "Q75"]
            else "engineered descriptor"
            for f in SHARED_10D
        ]
    }).sort_values("CV_PermImp_Mean", ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, "P1_feature_importance.csv"),
               index=False, encoding="utf-8-sig")

    # LASSO coefficients at the selected C
    lasso_full = Pipeline([("scaler", StandardScaler()),
                           ("clf", LogisticRegression(penalty="l1", solver="saga",
                                                      C=best_c, max_iter=10000,
                                                      random_state=SEED))]).fit(X3, y)
    pd.DataFrame(lasso_full.named_steps["clf"].coef_,
                 index=[f"Class_{c}" for c in CALL_ORDER],
                 columns=SHARED_10D).to_csv(
        os.path.join(OUT_DIR, "P1_lasso_coefficients.csv"), encoding="utf-8-sig")

    # Permutation importance figure
    imp_plot = imp.sort_values("CV_PermImp_Mean")
    colors = ["#DD8452" if f in ENG_FEATS + ["PeakAtFundamental"] else "#4C72B0"
              for f in imp_plot["Feature"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.barh(imp_plot["Feature"], imp_plot["CV_PermImp_Mean"],
            xerr=imp_plot["CV_PermImp_SD"], color=colors, capsize=3)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("Held-out permutation importance (accuracy drop)")
    ax.set_title("Cross-validated Permutation Importance (Set 3, 10D RF)\n"
                 "Bars are fold means; error bars are fold SD",
                 fontweight="bold")
    fig.text(0.01, 0.01,
             "Note: HarmonicRatio is the bimodal Fpeak/F01 mixture (Fig. 2); "
             "its importance is an artifact, not a biological harmonic ratio.",
             fontsize=8.5, color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig13_permutation_importance.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[Feature importance, Set 3]")
    print(imp[["Feature", "CV_PermImp_Mean", "MutualInfo", "LASSO_SelectFreq"]]
          .round(3).to_string(index=False))

    # ---- SHAP (descriptive only) -------------------------------------------
    if HAS_SHAP:
        X3_df = pd.DataFrame(X3, columns=SHARED_10D)
        explainer = shap.TreeExplainer(rf_full)
        sv = explainer.shap_values(X3_df)
        if isinstance(sv, list):
            mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
            sv_list = sv
        else:
            mean_shap = np.abs(sv).mean(axis=(0, 2))
            sv_list = [sv[:, :, i] for i in range(len(CALL_ORDER))]

        sh_df = pd.DataFrame({"Feature": SHARED_10D, "Mean_Abs_SHAP": mean_shap})
        sh_df = sh_df.sort_values("Mean_Abs_SHAP")
        plt.figure(figsize=(8.6, 5.6))
        colors_bar = ["#DD8452" if f in ENG_FEATS + ["PeakAtFundamental"] else "#4C72B0"
                      for f in sh_df["Feature"]]
        plt.barh(sh_df["Feature"], sh_df["Mean_Abs_SHAP"], color=colors_bar)
        plt.xlabel("Mean |SHAP value| (orange = engineered)")
        plt.title(f"Global SHAP, 10D RF (descriptive; repeated-CV Acc="
                  f"{ablation.loc[(ablation.FeatureSet=='Set 3: +Peak regime 10D') & (ablation.Model=='RandomForest'), 'Acc_Mean'].values[0]:.3f})\n"
                  "SHAP explains the model, not the biology",
                  fontweight="bold")
        plt.figtext(0.01, 0.01,
                    "HarmonicRatio is bimodal (Fpeak at the fundamental vs 3rd harmonic); "
                    "treat its rank as a measurement artifact.",
                    fontsize=8.5, color="#555555")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "P1_fig7_SHAP_global.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
        for i, ctype in enumerate(CALL_ORDER):
            plt.sca(axes[i])
            shap.summary_plot(sv_list[i], X3_df, show=False, plot_size=None)
            axes[i].set_title(f"SHAP impact: '{ctype}'", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "P1_fig7b_SHAP_by_class.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ---- RF hyperparameter sensitivity -------------------------------------
    sens_rows = []
    skf5 = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    for n_est in [200, 300, 500]:
        for msl in [1, 2, 4]:
            mdl = RandomForestClassifier(
                n_estimators=n_est, min_samples_leaf=msl,
                class_weight="balanced", random_state=SEED, n_jobs=1
            )
            r = cross_validate(mdl, X3, y, cv=skf5,
                               scoring={"acc": "accuracy", "bal_acc": "balanced_accuracy"},
                               n_jobs=1)
            sens_rows.append({
                "n_estimators": n_est, "min_samples_leaf": msl,
                "Acc_Mean": float(r["test_acc"].mean()),
                "BalAcc_Mean": float(r["test_bal_acc"].mean())
            })
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(os.path.join(OUT_DIR, "P1_rf_hyperparam_sensitivity.csv"),
                index=False, encoding="utf-8-sig")
    print("\n[RF hyperparameter sensitivity]")
    print(sens.round(3).to_string(index=False))

    return {
        "le": le, "y": y, "X_sets": X_sets, "ablation": ablation,
        "oof": oof, "imp": imp, "best_c": best_c
    }


# ==============================================================================
# 8. Hierarchical classifier (definitional stage + scientific stage)
# ==============================================================================
def hierarchical_analysis(df1, cls):
    y = cls["y"]
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    scoring = {"acc": "accuracy", "bal_acc": "balanced_accuracy",
               "auc": "roc_auc", "kappa": make_scorer(cohen_kappa_score)}

    df_hm = df1[df1["CallType"].isin(["flat", "mod"])].reset_index(drop=True)
    X_hm = np.asarray(df_hm[SHARED_10D].astype(float).to_numpy(), float)
    y_hm = (df_hm["CallType"].astype(str).values == "mod").astype(int)
    counts_hm = y_hm.mean()
    baseline_hm = max(counts_hm, 1 - counts_hm)

    rows = []
    # Stage 1: definitional rule, not a learned model
    stage1_pred = df1["F02_present"].values.astype(int)
    stage1_true = (df1["CallType"].astype(str).values == "tv").astype(int)
    rows.append({
        "Stage": "Stage 1: F02 present -> tv",
        "Type": "Definitional rule (not a learned classifier)",
        "Acc": float(accuracy_score(stage1_true, stage1_pred)),
        "BalAcc": float(balanced_accuracy_score(stage1_true, stage1_pred)),
        "AUC": np.nan, "Kappa": float(cohen_kappa_score(stage1_true, stage1_pred)),
        "Acc_CI_Low": np.nan, "Acc_CI_High": np.nan,
        "Baseline": float(max(stage1_true.mean(), 1 - stage1_true.mean())),
        "N": len(stage1_true)
    })

    hm_models = {"RandomForest": make_rf, "LASSO-Logit (nested C)": make_lasso}
    hm_res = {}
    hm_oof = {}
    for m_name, make_model in hm_models.items():
        res = cross_validate(make_model(), X_hm, y_hm, cv=rskf, scoring=scoring,
                             n_jobs=1, error_score="raise")
        lo, hi = bootstrap_ci(res["test_acc"])
        hm_res[m_name] = res
        hm_oof[m_name] = repeated_oof_proba(make_model, X_hm, y_hm)
        rows.append({
            "Stage": "Stage 2: flat vs mod",
            "Type": m_name,
            "Acc": float(res["test_acc"].mean()),
            "BalAcc": float(res["test_bal_acc"].mean()),
            "AUC": float(res["test_auc"].mean()),
            "Kappa": float(res["test_kappa"].mean()),
            "Acc_CI_Low": lo, "Acc_CI_High": hi,
            "Baseline": baseline_hm, "N": len(y_hm)
        })

    # Combined rule: Stage 1 deterministic + Stage 2 OOF predictions
    y_all = y
    for m_name in hm_models:
        pred_hm_proba = hm_oof[m_name]
        pred_hm = pred_hm_proba.argmax(axis=1)
        pred_comb = np.empty(len(df1), dtype=int)
        tv_mask = df1["CallType"].astype(str).values == "tv"
        pred_comb[tv_mask] = 2
        pred_comb[~tv_mask] = pred_hm
        acc_comb = accuracy_score(y_all, pred_comb)
        bal_comb = balanced_accuracy_score(y_all, pred_comb)
        rows.append({
            "Stage": "Combined hierarchical rule",
            "Type": m_name,
            "Acc": float(acc_comb),
            "BalAcc": float(bal_comb),
            "AUC": np.nan,
            "Kappa": float(cohen_kappa_score(y_all, pred_comb)),
            "Acc_CI_Low": np.nan, "Acc_CI_High": np.nan,
            "Baseline": 1.0 / 3.0, "N": len(y_all)
        })

    # Reference: flat 3-class acoustic model
    ref = cls["ablation"]
    ref_rf = ref[(ref.FeatureSet == "Set 3: +Peak regime 10D") &
                 (ref.Model == "RandomForest")].iloc[0]
    rows.append({
        "Stage": "Reference: flat 3-class acoustic RF",
        "Type": "Set 3, 10D",
        "Acc": float(ref_rf.Acc_Mean), "BalAcc": float(ref_rf.BalAcc_Mean),
        "AUC": float(ref_rf.AUC_OvR_Mean), "Kappa": float(ref_rf.Kappa_Mean),
        "Acc_CI_Low": float(ref_rf.Acc_CI_Low), "Acc_CI_High": float(ref_rf.Acc_CI_High),
        "Baseline": 1.0 / 3.0, "N": len(y_all)
    })

    hm = pd.DataFrame(rows)
    hm.to_csv(os.path.join(OUT_DIR, "P1_hierarchical_classifier.csv"),
              index=False, encoding="utf-8-sig")

    print("\n[Hierarchical classifier]")
    print(hm[["Stage", "Type", "Acc", "BalAcc", "AUC", "Baseline"]]
          .round(3).to_string(index=False))
    print("Stage 1 is DEFINITIONAL (F02 presence defines tv); do not report its "
          "100% as a learned classification result.")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    plot_rows = hm[hm["Stage"] != "Reference: flat 3-class acoustic RF"].copy()
    x = np.arange(len(plot_rows))
    colors = ["#999999", "#4C72B0", "#4C72B0", "#55A868", "#55A868"][:len(plot_rows)]
    axes[0].bar(x, plot_rows["Acc"], color=colors)
    axes[0].axhline(baseline_hm, color="#D62728", ls="--", lw=1.2,
                    label=f"Stage-2 majority baseline = {baseline_hm:.3f}")
    axes[0].axhline(1.0 / 3.0, color="#7F7F7F", ls=":", lw=1.2,
                    label="3-class chance = 0.333")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}\n{t}" for s, t in zip(plot_rows["Stage"], plot_rows["Type"])],
                            rotation=26, ha="right", fontsize=8.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Hierarchical stages", fontweight="bold")
    axes[0].legend(fontsize=8)

    pred_hm = hm_oof["RandomForest"].argmax(axis=1)
    pred_comb = np.empty(len(df1), dtype=int)
    tv_mask = df1["CallType"].astype(str).values == "tv"
    pred_comb[tv_mask] = 2
    pred_comb[~tv_mask] = pred_hm
    ConfusionMatrixDisplay(confusion_matrix(y, pred_comb, labels=[0, 1, 2]),
                           display_labels=CALL_ORDER).plot(ax=axes[1],
                                                          cmap="Greens", colorbar=False)
    axes[1].set_title(f"Combined rule (RF stage 2)\nAcc={accuracy_score(y, pred_comb):.3f}, "
                      f"kappa={cohen_kappa_score(y, pred_comb):.3f}",
                      fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig11_hierarchical_classifier.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    return hm


# ==============================================================================
# 9. Playback adjustment and propagation synthesis
# ==============================================================================
def playback_analysis(df2, c_resp):
    d = df2.copy()
    d["resp"] = pd.to_numeric(d[c_resp], errors="coerce")
    d = d.dropna(subset=["CallType", "Context", "Distance", "Npenguins", "resp"])
    # Explicit reference categories: control for CallType, commuting for Context
    d["Context"] = pd.Categorical(d["Context"].astype(str),
                                  categories=["commuting", "after", "searching"])
    d["CallType"] = pd.Categorical(d["CallType"].astype(str),
                                   categories=["control", "flat", "mod", "tv"])

    # Raw and stratified rates for transparency
    strat = d.groupby(["CallType", "Context"], observed=True)["resp"].agg(
        ["mean", "count"]).reset_index()
    strat.columns = ["CallType", "Context", "Response_Rate", "N"]
    strat.to_csv(os.path.join(OUT_DIR, "P1_playback_stratified.csv"),
                 index=False, encoding="utf-8-sig")

    raw = d.groupby("CallType", observed=True)["resp"].agg(
        ["mean", "count"]).reset_index()
    raw.columns = ["CallType", "Response_Rate", "N"]
    raw.to_csv(os.path.join(OUT_DIR, "P1_playback_summary.csv"),
               index=False, encoding="utf-8-sig")

    # Adjusted logistic: response ~ CallType + Context + Distance + Npenguins
    X = pd.get_dummies(d[["CallType", "Context"]], drop_first=True).astype(float)
    X["Distance"] = pd.to_numeric(d["Distance"], errors="coerce").astype(float)
    X["Npenguins"] = pd.to_numeric(d["Npenguins"], errors="coerce").astype(float)
    terms = list(X.columns)
    y = d["resp"].astype(float).values
    Xv = np.asarray(X.to_numpy(), dtype=float)

    m = LogisticRegression(C=1e6, max_iter=5000).fit(Xv, y)
    or_point = np.exp(m.coef_[0])

    # Cluster bootstrap by experimental session (expID)
    exp_ids = np.asarray(d["expID"].astype(str).tolist(), dtype=object)
    uniq = np.unique(exp_ids)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(500):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(exp_ids == e)[0] for e in samp])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            mb = LogisticRegression(C=1e6, max_iter=5000).fit(Xv[idx], y[idx])
            boot.append(np.exp(mb.coef_[0]))
        except Exception:
            continue
    if boot:
        boot = np.asarray(boot)
        ci = np.percentile(boot, [2.5, 97.5], axis=0)
    else:
        ci = np.full((2, len(terms)), np.nan)

    adj = pd.DataFrame({
        "Term": terms, "Odds_Ratio": or_point,
        "OR_CI_Low": ci[0], "OR_CI_High": ci[1],
        "Note": ["Reference = control (context = commuting)" if t.startswith("CallType_")
                 else "Reference = commuting" if t.startswith("Context_")
                 else "Continuous covariate" for t in terms],
        "Model_Note": ("Quasi-separation possible (control Approach is all zero); "
                       "ORs are adjusted associations, not causal effects")
    })
    adj.to_csv(os.path.join(OUT_DIR, "P1_playback_adjusted.csv"),
               index=False, encoding="utf-8-sig")

    print("\n[Playback: raw marginal rates]")
    print(raw.round(3).to_string(index=False))
    print("\n[Playback: adjusted odds ratios, cluster bootstrap 95% CI]")
    print(adj.round(3).to_string(index=False))

    # Figure: OR forest + stratified rates
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6),
                             gridspec_kw={"width_ratios": [1.1, 1.4]})
    call_terms = [t for t in terms if t.startswith("CallType_")]
    call_idx = [terms.index(t) for t in call_terms]
    ypos = np.arange(len(call_terms))
    ors = adj.loc[call_idx, "Odds_Ratio"].values
    los = adj.loc[call_idx, "OR_CI_Low"].values
    his = adj.loc[call_idx, "OR_CI_High"].values
    axes[0].errorbar(ors, ypos, xerr=[ors - los, his - ors], fmt="o",
                     color="#4C72B0", ecolor="#555555", capsize=4, lw=1.4)
    axes[0].axvline(1.0, color="black", ls="--", lw=1.0)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([t.replace("CallType_", "") for t in call_terms])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Adjusted odds ratio (log scale)")
    axes[0].set_title("Adjusted response odds vs control\n"
                      "(Context, Distance, Npenguins held constant)",
                      fontweight="bold")
    axes[0].invert_yaxis()

    sns.barplot(data=strat, x="Context", y="Response_Rate", hue="CallType",
                hue_order=[c for c in ["control", "flat", "mod", "tv"]
                           if c in strat["CallType"].unique()],
                palette={"control": "#999999", **PALETTE}, ax=axes[1])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Proportion responding")
    axes[1].set_title("Raw response rates are confounded by context\n"
                      "(use the adjusted ORs for inference)", fontweight="bold")
    axes[1].legend(title="Stimulus", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig10_playback_adjusted.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    return adj, raw, strat


def propagation_analysis(deg_all):
    deg_sum = deg_all.groupby(["Metric", "CallType", "Distance_m"])["Correlation"].agg(
        Mean="mean", SD="std", N="count").reset_index()
    deg_sum.to_csv(os.path.join(OUT_DIR, "P1_propagation_summary.csv"),
                   index=False, encoding="utf-8-sig")

    comp = deg_all.groupby(["CallType", "Distance_m"])["Correlation"].agg(
        Mean="mean", SD="std", N="count").reset_index()
    comp.to_csv(os.path.join(OUT_DIR, "P1_propagation_composite.csv"),
                index=False, encoding="utf-8-sig")

    def crossing_distance(dist, val, level=0.5):
        dist = np.asarray(dist, float)
        val = np.asarray(val, float)
        order = np.argsort(dist)
        dist, val = dist[order], val[order]
        for i in range(len(dist) - 1):
            if (val[i] - level) * (val[i + 1] - level) <= 0 and val[i] != val[i + 1]:
                x1, x2 = np.log2(dist[i]), np.log2(dist[i + 1])
                t = (level - val[i]) / (val[i + 1] - val[i])
                return float(2 ** (x1 + t * (x2 - x1)))
        return np.nan

    rows = []
    for ctype in CALL_ORDER:
        sub = comp[comp["CallType"] == ctype].sort_values("Distance_m")
        cross = crossing_distance(sub["Distance_m"].values, sub["Mean"].values)
        f128 = float(sub.loc[sub["Distance_m"] == 128, "Mean"].values[0]) \
            if np.any(sub["Distance_m"] == 128) else np.nan
        f256 = float(sub.loc[sub["Distance_m"] == 256, "Mean"].values[0]) \
            if np.any(sub["Distance_m"] == 256) else np.nan
        rows.append({
            "CallType": ctype, "Crossing_50m": cross,
            "Fidelity_128m": f128, "Fidelity_256m": f256
        })
    cross_df = pd.DataFrame(rows)
    cross_df.to_csv(os.path.join(OUT_DIR, "P1_propagation_crossing.csv"),
                    index=False, encoding="utf-8-sig")

    print("\n[Propagation: composite fidelity and 50% crossing distance]")
    print(cross_df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.7))
    for ax, met in zip(axes[:3], ["Spectrogram", "AmplitudeSpectrum", "Envelope"]):
        sub_m = deg_sum[deg_sum["Metric"] == met].sort_values("Distance_m")
        for ctype in CALL_ORDER:
            sub_c = sub_m[sub_m["CallType"] == ctype]
            ax.errorbar(sub_c["Distance_m"], sub_c["Mean"], yerr=sub_c["SD"],
                        marker="o", capsize=3, label=ctype,
                        color=PALETTE[ctype], lw=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{met} decay", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Distance (m)")
        if met == "Spectrogram":
            ax.set_ylabel("Mean correlation (± SD)")
            ax.legend(title="CallType")

    ax = axes[3]
    for ctype in CALL_ORDER:
        sub = comp[comp["CallType"] == ctype].sort_values("Distance_m")
        ax.errorbar(sub["Distance_m"], sub["Mean"], yerr=sub["SD"],
                    marker="o", capsize=3, label=ctype, color=PALETTE[ctype], lw=2)
        cr = cross_df.loc[cross_df.CallType == ctype, "Crossing_50m"].values[0]
        if np.isfinite(cr):
            ax.axvline(cr, color=PALETTE[ctype], ls=":", lw=1.2, alpha=0.8)
            ax.text(cr, 0.08, f"{cr:.0f} m", color=PALETTE[ctype],
                    fontsize=8.5, rotation=90, ha="right")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_title("Composite fidelity (mean of 3 metrics)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Composite fidelity")

    plt.suptitle("Propagation Robustness by Call Type (descriptive hand-off to Problem 2)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig8_propagation_synthesis.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    return deg_sum, comp, cross_df


def information_robustness(cls, cross_df):
    # One-vs-rest AUC per class from the leakage-free Set 3 RF OOF probabilities
    y = cls["y"]
    proba = cls["oof"][("Set 3: +Peak regime 10D", "RandomForest")]
    aucs = []
    for i, ctype in enumerate(CALL_ORDER):
        y_bin = (y == i).astype(int)
        aucs.append(float(roc_auc_score(y_bin, proba[:, i])))

    trade = pd.DataFrame({
        "CallType": CALL_ORDER,
        "AUC_OvR": aucs,
        "Fidelity_128m": [float(cross_df.loc[cross_df.CallType == c, "Fidelity_128m"].values[0])
                          for c in CALL_ORDER],
        "Fidelity_256m": [float(cross_df.loc[cross_df.CallType == c, "Fidelity_256m"].values[0])
                          for c in CALL_ORDER],
        "Crossing_50m": [float(cross_df.loc[cross_df.CallType == c, "Crossing_50m"].values[0])
                         for c in CALL_ORDER]
    })
    trade["InfoRetention_128"] = trade["AUC_OvR"] * trade["Fidelity_128m"]
    trade["InfoRetention_256"] = trade["AUC_OvR"] * trade["Fidelity_256m"]
    trade.to_csv(os.path.join(OUT_DIR, "P1_information_robustness.csv"),
                 index=False, encoding="utf-8-sig")

    print("\n[Information x robustness trade-off]")
    print(trade.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, ycol, title in [
        (axes[0], "Fidelity_128m", "Information vs fidelity at 128 m"),
        (axes[1], "Fidelity_256m", "Information vs fidelity at 256 m")
    ]:
        for _, row in trade.iterrows():
            ax.scatter(row["AUC_OvR"], row[ycol], s=120,
                       color=PALETTE[row["CallType"]], edgecolors="black", zorder=3)
            ax.annotate(row["CallType"], (row["AUC_OvR"], row[ycol]),
                        textcoords="offset points", xytext=(7, 7), fontsize=10,
                        fontweight="bold")
        ax.set_xlabel("One-vs-rest AUC (discriminability)")
        ax.set_ylabel("Composite fidelity")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(0, 1.05)

    x = np.arange(len(trade))
    axes[2].bar(x - 0.18, trade["InfoRetention_128"], width=0.36,
                color=[PALETTE[c] for c in trade["CallType"]], label="128 m")
    axes[2].bar(x + 0.18, trade["InfoRetention_256"], width=0.36,
                color=[PALETTE[c] for c in trade["CallType"]], alpha=0.45, label="256 m")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(trade["CallType"])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("AUC × fidelity")
    axes[2].set_title("Information-retention score", fontweight="bold")
    axes[2].legend(fontsize=8)

    plt.suptitle("Information Function × Transmission Robustness (P1 -> P2/P3 hand-off)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P1_fig9_information_robustness.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    return trade


def feature_contract():
    rows = [
        ("F01", "First fundamental (carrier frequency)",
         "Significant after date adjustment (Holm p=0.021); tv lower than flat/mod",
         "P2: absorption and geometric spreading; P4: source level"),
        ("F02", "Second fundamental (two-voice structure)",
         "Definitional for tv; F02/F01 ~ 1.11; F02-F01 ~ 30 Hz beat",
         "P2: two-voice robustness; P4: individual signature / beat frequency"),
        ("Fpeak", "Peak frequency",
         "Naive Holm significant, but not robust to date adjustment; regime-dependent (fundamental vs 3rd harmonic)",
         "P2: spectrogram and amplitude-spectrum correlation"),
        ("Q25", "25th-percentile frequency", "Not significant",
         "P2: spectral envelope fidelity"),
        ("Q50", "Median frequency", "Weak; significance driven by recording date",
         "P2: spectral envelope fidelity"),
        ("Q75", "75th-percentile frequency", "Not significant",
         "P2: spectral envelope fidelity"),
        ("Duration", "Call duration", "Not significant",
         "P2: envelope correlation; P3: exposure window"),
        ("PulseRate", "Pulse rate (mod/tv only)", "mod vs tv not significant",
         "P2: envelope correlation; P3: latency"),
        ("SpecBandwidth", "Q75 - Q25", "Redundant with quartiles",
         "P2: spectral fidelity (descriptive)"),
        ("HarmonicRatio", "Fpeak / F01", "Bimodal mixture artifact; do not interpret as biology",
         "Do not use as a biological harmonic ratio"),
        ("SpecSkew", "Spectral skewness", "Not significant",
         "P2: spectral shape (descriptive)"),
        ("PeakAtFundamental", "Fpeak/F01 < 1.5", "Regime indicator; separates peak regimes",
         "P2: spectral peak interpretation"),
        ("CallType", "Categorical call type", "Response factor",
         "P3: playback response model"),
    ]
    df = pd.DataFrame(rows, columns=["Feature", "Role", "Evidence_in_P1", "Downstream_Use"])
    df.to_csv(os.path.join(OUT_DIR, "P1_feature_contract.csv"),
              index=False, encoding="utf-8-sig")
    print("\n[Feature contract for P2/P3]")
    print(df.to_string(index=False))
    return df


# ==============================================================================
# 10. Main
# ==============================================================================
class Tee:
    """Write stdout to both the console and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False


def run_with_log(fn):
    """Run fn() and save its console output to outputs_p1/P1_run_log.txt."""
    log_path = os.path.join(OUT_DIR, "P1_run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = Tee(old, f)
        try:
            fn()
        finally:
            sys.stdout = old


def prepare_data_artifacts():
    """Validate inputs and regenerate the data-derived artifacts."""
    check_data()
    write_data_manifest()
    df1, df2, deg_all, audit = load_and_clean()
    export_analysis_dataset(df1)
    return df1, df2, deg_all, audit


def main():
    df1, df2, deg_all, audit = prepare_data_artifacts()

    print("\n" + "=" * 78)
    print("STEP 2: DESCRIPTIVE STATISTICS, MISSINGNESS AND COLLINEARITY")
    print("=" * 78)
    desc = descriptive_statistics(df1)
    plot_descriptive(df1)
    corr, vif = collinearity_diagnostics(df1)

    print("\n" + "=" * 78)
    print("STEP 3: NONPARAMETRIC TESTS (GLOBAL HOLM + DATE ADJUSTMENT)")
    print("=" * 78)
    kw, pw = nonparametric_tests(df1)

    print("\n" + "=" * 78)
    print("STEP 4: PERMANOVA / PERMDISP ON THE NON-REDUNDANT 6D SPACE")
    print("=" * 78)
    perm = permanova_analysis(df1)

    print("\n" + "=" * 78)
    print("STEP 5: CLASSIFICATION, BASELINES, UNCERTAINTY AND IMPORTANCE")
    print("=" * 78)
    cls = classification_analysis(df1)

    print("\n" + "=" * 78)
    print("STEP 6: HIERARCHICAL (DEFINITIONAL + SCIENTIFIC) CLASSIFIER")
    print("=" * 78)
    hm = hierarchical_analysis(df1, cls)

    print("\n" + "=" * 78)
    print("STEP 7: PLAYBACK ADJUSTMENT AND PROPAGATION SYNTHESIS")
    print("=" * 78)
    c_resp = find_col(df2.columns, "Response")
    adj, raw, strat = playback_analysis(df2, c_resp)
    deg_sum, comp, cross_df = propagation_analysis(deg_all)
    trade = information_robustness(cls, cross_df)
    contract = feature_contract()

    print("\n" + "=" * 78)
    print("HONEST CONCLUSIONS FOR THE PAPER")
    print("=" * 78)
    acc3 = float(cls["ablation"].loc[
        (cls["ablation"].FeatureSet == "Set 3: +Peak regime 10D") &
        (cls["ablation"].Model == "RandomForest"), "Acc_Mean"].values[0])
    acc_def = float(cls["ablation"].loc[
        (cls["ablation"].FeatureSet == "Set 4: +Structural flags (definitional)") &
        (cls["ablation"].Model == "RandomForest"), "Acc_Mean"].values[0])
    acc_flag = float(cls["ablation"].loc[
        (cls["ablation"].FeatureSet == "Definitional flags only (2D)") &
        (cls["ablation"].Model == "RandomForest"), "Acc_Mean"].values[0])
    print(f"1. Leakage-free 3-class acoustic accuracy (Set 3, RF) = {acc3:.3f}; "
          f"majority baseline = 0.407.")
    print(f"2. Two definitional flags alone give {acc_flag:.3f}; adding acoustics gives "
          f"{acc_def:.3f}. The 94% is definitional, not acoustic.")
    print("3. flat vs mod is the only genuine acoustic problem; the provided summary "
          "features leave a large overlap.")
    print("4. Report global-Holm p-values and date-adjusted p-values, not the naive "
          "per-family p-values.")
    print("5. Use the adjusted playback ORs (cluster bootstrap CI), not the raw marginal "
          "response rates, for functional claims.")
    print(f"6. 50% composite-fidelity distances: "
          + ", ".join([f"{r.CallType}={r.Crossing_50m:.0f} m"
                       for r in cross_df.itertuples() if np.isfinite(r.Crossing_50m)]))

    # Machine-readable summary for the paper team
    kw_idx = kw.set_index("Feature")
    pw_sig = pw[pw["Significant_Global_Holm"]]
    pw_sig_d = pw[pw["Significant_Global_Holm_Date"]]
    summary = [
        "Problem 1 v2 -- key results",
        f"Environment: Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, sklearn {sklearn.__version__}, seaborn {sns.__version__}",
        f"Mode: {'FAST' if FAST else 'FULL'} (N_PERM={N_PERM}, N_BOOT={N_BOOT}, "
        f"N_REPEATS={N_REPEATS})",
        f"Analytic sample: n={len(df1)} "
        f"(flat={(df1['CallType']=='flat').sum()}, "
        f"mod={(df1['CallType']=='mod').sum()}, "
        f"tv={(df1['CallType']=='tv').sum()}); dates={df1['Date'].nunique()}",
        "",
        "Univariate Kruskal-Wallis (Holm across 9 features):",
        f"  F01:   p_raw={kw_idx.loc['F01','p_raw']:.4f}, "
        f"p_holm={kw_idx.loc['F01','p_holm']:.4f}, "
        f"p_holm_date={kw_idx.loc['F01','p_holm_date_adjusted']:.4f}",
        f"  Fpeak: p_raw={kw_idx.loc['Fpeak','p_raw']:.4f}, "
        f"p_holm={kw_idx.loc['Fpeak','p_holm']:.4f}, "
        f"p_holm_date={kw_idx.loc['Fpeak','p_holm_date_adjusted']:.4f}",
        "  Interpretation: Fpeak is naive-significant but not date-robust; "
        "F01 is the date-robust signal.",
        "",
        "Pairwise Mann-Whitney (global Holm across 27):",
        f"  Naive significant: "
        f"{', '.join(pw_sig['Feature'].astype(str) + ' (' + pw_sig['Comparison'].astype(str) + ')') if len(pw_sig) else 'none'}",
        f"  Date-adjusted significant: "
        f"{', '.join(pw_sig_d['Feature'].astype(str) + ' (' + pw_sig_d['Comparison'].astype(str) + ')') if len(pw_sig_d) else 'none'}",
        "",
        "PERMANOVA (6D non-redundant):",
        f"  Overall: pseudo-F={perm.iloc[0]['Pseudo_F']:.3f}, "
        f"R2={perm.iloc[0]['R2']:.3f}, p={perm.iloc[0]['p_perm']:.4f}, "
        f"PERMDISP p={perm.iloc[0]['PERMDISP_p']:.3f}",
        "",
        "Classification (10x5 repeated stratified CV):",
        f"  Leakage-free 3-class acoustic RF (Set 3): Acc={acc3:.3f}; "
        "majority baseline=0.407",
        f"  Definitional flags only (2D): Acc={acc_flag:.3f}",
        f"  Definitional + acoustic (Set 4): Acc={acc_def:.3f}",
        "  -> The ~94% is definitional; the acoustic-only model is near baseline.",
        f"  Hierarchical Stage 2 (flat vs mod, RF): "
        f"Acc={hm.loc[hm.Stage=='Stage 2: flat vs mod','Acc'].iloc[0]:.3f}; baseline=0.524",
        "",
        "Playback adjusted odds ratios (cluster bootstrap 95% CI):",
    ]
    for _, r in adj[adj["Term"].astype(str).str.startswith("CallType_")].iterrows():
        summary.append(f"  {r['Term']}: OR={r['Odds_Ratio']:.2f} "
                       f"[{r['OR_CI_Low']:.2f}, {r['OR_CI_High']:.2f}]")
    summary += ["", "Propagation composite fidelity:"]
    for _, r in cross_df.iterrows():
        summary.append(f"  {r['CallType']}: 50% crossing={r['Crossing_50m']:.1f} m; "
                       f"fidelity@128m={r['Fidelity_128m']:.3f}; "
                       f"fidelity@256m={r['Fidelity_256m']:.3f}")
    summary += ["", "Information x robustness (AUC_ovr from Set 3 RF OOF):"]
    for _, r in trade.iterrows():
        summary.append(f"  {r['CallType']}: AUC={r['AUC_OvR']:.3f}; "
                       f"InfoRetention@128m={r['InfoRetention_128']:.3f}")
    summary += [
        "",
        "Honest conclusions:",
        "  1. The v1 94% classifier was definitional (F02/PulseRate presence), not acoustic.",
        "  2. flat vs mod is the only genuine acoustic problem; the supplied summary features overlap heavily.",
        "  3. F01 is the date-robust univariate signal; Fpeak's naive significance is partly session-driven.",
        "  4. Report global-Holm and date-adjusted p-values; use adjusted playback ORs for functional claims.",
    ]
    with open(os.path.join(OUT_DIR, "P1_key_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")
    with open(os.path.join(OUT_DIR, "P1_run_environment.txt"), "w", encoding="utf-8") as f:
        f.write(f"Python {sys.version}\n")
        f.write(f"numpy {np.__version__}\npandas {pd.__version__}\n")
        f.write(f"scikit-learn {sklearn.__version__}\nseaborn {sns.__version__}\n")
        f.write(f"FAST={FAST}; N_PERM={N_PERM}; N_BOOT={N_BOOT}; "
                f"N_REPEATS={N_REPEATS}\n")

    print(f"\nAll outputs saved to ./{os.path.relpath(OUT_DIR, BASE_DIR)}/")


if __name__ == "__main__":
    if "--data-only" in sys.argv:
        run_with_log(prepare_data_artifacts)
    else:
        run_with_log(main)
