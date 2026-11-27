# -*- coding: utf-8 -*-
"""
Problem 2 (v2): Sound-Signal Propagation Degradation and Effective Communication Range
--------------------------------------------------------------------------------------
P0/P1 revision of the first P2 version. All outputs in outputs_p2/ are generated
by this script.

What changed
------------
P0
1. Informative missingness is now audited and bounded. Missing correlations are
   trial-level (the same trials are missing for all three metrics) and strongly
   distance-dependent (e.g. tv: 110/300 missing at 128 m, 90/300 at 256 m).
   We report:
     - a missingness audit table/figure;
     - complete-case, worst-case (missing F = 0) and best-case (missing F = 1)
       degradation fits -> partial-identification bounds for the effective range.
2. The main range model is now the logistic curve (bounded, monotone,
   interpretable half-decay distance). The physical log-exponential is retained
   as a mechanistic comparator, reparameterised at 4 m; its log-distance term is
   flagged when it collapses to the boundary. A model-averaged range (weights
   from trial-level CV-RMSE) and a model-spread interval are reported, so the
   tau = 0.70 ranking is not presented as a single-model certainty.
3. Probabilistic effective range: R_tau(p) = max{d : P(F(d) >= tau) >= p},
   estimated from the trial-level empirical distribution, with p = 0.50/0.75/0.90.
4. Threshold-dependent dominance: the tv and mod range curves cross at
   tau* ~ 0.59; the crossover is estimated with a trial bootstrap and annotated.
P1
5. Physical model reparameterised at 4 m: F(d) = F4 - b ln(d/4) - c (d-4).
6. AIC/BIC removed from the main comparison (row-level n ignores trial
   clustering); trial-level CV-RMSE is the primary criterion.
7. Fusion diagnostics: PCA loadings, communalities and uniqueness; metric
   correlation matrix; PCA/equal/entropy/geometric sensitivity retained.
8. Bootstrap now re-estimates the PCA fusion weights in every resample
   (fusion-weight uncertainty is propagated).
9. Metric-specific effective ranges (spectrogram / amplitude spectrum /
   envelope) identify the limiting channel.
10. Interpolation sensitivity (all zeros in v1) removed; replaced by
    missingness bounds, weighted-fit sensitivity and GAM-knot sensitivity.
11. Playback consistency check upgraded to an adjusted logistic calibration
    P(response) ~ F_model + Context + Npenguins, with cluster bootstrap by expID;
    it reports the fidelity at which response probability is 0.5/0.7 and
    compares it with the acoustic thresholds.

Run:
    python P2.py
    P2_FAST=1 python P2.py      # fewer bootstrap replicates

Requires: Python 3.8+, numpy, pandas, scipy, matplotlib, seaborn>=0.12,
          scikit-learn>=1.1. No statsmodels/R dependency.
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

from scipy import stats
from scipy.optimize import curve_fit, minimize
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


# ==============================================================================
# 0. Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs_p2")
os.makedirs(OUT_DIR, exist_ok=True)

FAST = os.environ.get("P2_FAST", "0") == "1"
SEED = 42
N_BOOT = 300 if FAST else 1000
N_SPLITS = 5

CALL_ORDER = ["flat", "mod", "tv"]
DISTANCES = [4, 8, 16, 32, 64, 128, 256]
METRICS = [("spec", "Spectrogram"), ("amp", "Amplitude spectrum"), ("env", "Envelope")]
PALETTE = {"flat": "#4C72B0", "mod": "#DD8452", "tv": "#55A868"}

LEVELS = {"Potential": 0.50, "Recognition": 0.60, "Reliable": 0.70}
TAU_GRID = np.round(np.arange(0.40, 0.801, 0.05), 2)
TAU_GRID_FINE = np.round(np.arange(0.30, 0.901, 0.01), 2)
GRID = np.exp(np.linspace(np.log(4.0), np.log(256.0), 400))

FILE_MAP = {
    ("flat", "spec"): "*Data03*", ("mod", "spec"): "*Data04*", ("tv", "spec"): "*Data05*",
    ("flat", "amp"): "*Data06*", ("mod", "amp"): "*Data07*", ("tv", "amp"): "*Data08*",
    ("flat", "env"): "*Data09*", ("mod", "env"): "*Data10*", ("tv", "env"): "*Data11*",
}
FUSIONS = [("F_pca", "PCA"), ("F_equal", "Equal"),
           ("F_entropy", "Entropy"), ("F_geo", "Geometric")]
MAIN_MODEL_PREFERENCE = ["Isotonic (monotone)", "GAM spline", "Logistic", "Physical log-exp"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

print(f"[Environment] Python {sys.version.split()[0]} | numpy {np.__version__} | "
      f"pandas {pd.__version__} | sklearn {sklearn.__version__} | seaborn {sns.__version__}")
print(f"[Mode] {'FAST' if FAST else 'FULL'}: N_BOOT={N_BOOT}, N_SPLITS={N_SPLITS}")


# ==============================================================================
# 1. Helpers
# ==============================================================================
class Tee:
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
    log_path = os.path.join(OUT_DIR, "P2_run_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = Tee(old, f)
        try:
            fn()
        finally:
            sys.stdout = old


def smart_read_table(pattern):
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {DATA_DIR}")
    path = matches[0]
    df = pd.read_csv(path, na_values=["NA", "na", "NaN", "null", "", " "])
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed", case=False)]
    df = df.dropna(how="all").reset_index(drop=True)
    return df, os.path.basename(path)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def archive_deprecated_outputs():
    stale = ["P2_interpolation_sensitivity.csv",
             "P2_playback_consistency.csv",
             "P2_playback_consistency_corr.csv",
             "P2_playback_calibration.csv",
             "P2_fig13_playback_calibration.png"]
    dep = os.path.join(OUT_DIR, "_deprecated_v1")
    existing = [f for f in stale if os.path.exists(os.path.join(OUT_DIR, f))]
    if existing:
        os.makedirs(dep, exist_ok=True)
        for f in existing:
            os.replace(os.path.join(OUT_DIR, f), os.path.join(dep, f))
        print(f"[Archived {len(existing)} v1 outputs to {os.path.relpath(dep, BASE_DIR)}/]")


def check_data():
    required = [FILE_MAP[(ct, m)] for ct in CALL_ORDER for m, _ in METRICS]
    missing = [p for p in required if not glob.glob(os.path.join(DATA_DIR, p))]
    if missing:
        raise FileNotFoundError(
            "Missing required propagation files in data/:\n  " + "\n  ".join(missing)
            + "\nPlace Data03-Data11 in data/ and re-run."
        )
    print("[Data check] 9 propagation CSV files found.")


def write_data_manifest():
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
        is_prop = bool(re.match(r"Data(0[3-9]|1[01])", name))
        is_playback = bool(name.startswith("Data02"))
        rows.append({
            "File": name, "Size_bytes": os.path.getsize(path),
            "Rows": rows_n, "Columns": cols_n, "SHA256": file_sha256(path),
            "Used_by_P2": is_prop or is_playback,
            "Role": ("P2 propagation input" if is_prop
                     else "P2 playback calibration" if is_playback
                     else "not used by P2")
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "P2_data_manifest.csv"),
                              index=False, encoding="utf-8-sig")
    print(f"[Data manifest] {len(rows)} files -> outputs_p2/P2_data_manifest.csv")


# ==============================================================================
# 2. Data loading, missingness and fusion
# ==============================================================================
def _read_metric(ct, metric, pattern):
    raw, name = smart_read_table(pattern)
    dist_cols = {}
    for c in raw.columns:
        m = re.match(r"^\s*(\d+)\s*m?\s*$", str(c))
        if m:
            dist_cols[int(m.group(1))] = c
    missing = [d for d in DISTANCES if d not in dist_cols]
    if missing:
        raise ValueError(f"{name}: missing distance columns {missing}")
    out = pd.DataFrame({"TrialID": np.arange(len(raw))})
    n_out = 0
    for d in DISTANCES:
        v = pd.to_numeric(raw[dist_cols[d]], errors="coerce")
        bad = (v < -1.0) | (v > 1.0)
        n_out += int(bad.sum())
        out[f"{metric}_{d}"] = v.mask(bad)
    miss = int(out[[f"{metric}_{d}" for d in DISTANCES]].isna().sum().sum())
    return out, name, n_out, miss


def load_propagation_data():
    audit = []
    wide_by_ct = {}
    for ct in CALL_ORDER:
        frames = {}
        for metric, label in METRICS:
            f, name, n_out, n_miss = _read_metric(ct, metric, FILE_MAP[(ct, metric)])
            frames[metric] = f
            audit.append({"CallType": ct, "Metric": label, "File": name,
                          "Rows": len(f), "OutOfRangeCells": n_out,
                          "MissingCells": n_miss})
        n = min(len(frames[m]) for m, _ in METRICS)
        wide = pd.DataFrame({"TrialID": np.arange(n)})
        for metric, _ in METRICS:
            for d in DISTANCES:
                wide[f"{metric}_{d}"] = frames[metric][f"{metric}_{d}"].iloc[:n].values
        wide["CallType"] = ct
        wide_by_ct[ct] = wide

    rows = []
    for ct, wide in wide_by_ct.items():
        for d in DISTANCES:
            sub = wide[["TrialID"]].copy()
            sub["CallType"] = ct
            sub["Distance"] = d
            for metric, _ in METRICS:
                sub[f"r_{metric}"] = wide[f"{metric}_{d}"].values
            rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    for m in ["r_spec", "r_amp", "r_env"]:
        long[m + "_pos"] = long[m].clip(0.0, 1.0)
    long["CallType"] = pd.Categorical(long["CallType"], categories=CALL_ORDER, ordered=True)
    long["n_metrics"] = long[["r_spec_pos", "r_amp_pos", "r_env_pos"]].notna().sum(axis=1)
    return long, wide_by_ct, pd.DataFrame(audit)


def pca_weights_from_matrix(X):
    """PCA weights from an n x 3 matrix with possible NaN; returns w, var, loadings."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 10:
        return np.ones(3) / 3.0, np.nan, np.ones(3) / np.sqrt(3)
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1).fit(Xz)
    load = pca.components_[0]
    if load.sum() < 0:
        load = -load
    load = np.clip(load, 0.0, None)
    if load.sum() <= 0:
        load = np.ones(3)
        var = np.nan
    else:
        var = float(pca.explained_variance_ratio_[0])
    w = load / load.sum()
    return w, var, load / np.linalg.norm(load)


def compute_fusion_weights(long):
    X = long[["r_spec_pos", "r_amp_pos", "r_env_pos"]].values
    w_pca, pca_var, loadings = pca_weights_from_matrix(X)

    Xc = X[np.isfinite(X).all(axis=1)]
    P = Xc / np.maximum(Xc.sum(axis=0, keepdims=True), 1e-12)
    P = np.clip(P, 1e-12, None)
    e = -(P * np.log(P)).sum(axis=0) / np.log(len(P))
    d = 1.0 - e
    w_ent = d / d.sum() if d.sum() > 0 else np.ones(3) / 3.0

    w_eq = np.ones(3) / 3.0
    metric_labels = ["Spectrogram", "Amplitude spectrum", "Envelope"]
    rows = []
    for i, lab in enumerate(metric_labels):
        rows.append(("PCA", lab, w_pca[i], loadings[i], loadings[i] ** 2,
                     1 - loadings[i] ** 2,
                     f"PC1 explains {pca_var*100:.1f}% of standardised variance"))
    for i, lab in enumerate(metric_labels):
        rows.append(("Entropy", lab, w_ent[i], np.nan, np.nan, np.nan,
                     "Entropy-weight method"))
    for i, lab in enumerate(metric_labels):
        rows.append(("Equal", lab, w_eq[i], np.nan, np.nan, np.nan,
                     "Equal weights"))
    for i, lab in enumerate(metric_labels):
        rows.append(("Geometric", lab, w_pca[i], loadings[i], loadings[i] ** 2,
                     1 - loadings[i] ** 2,
                     "PCA weights used in weighted geometric mean"))
    table = pd.DataFrame(rows, columns=["Method", "Metric", "Weight", "Loading_PC1",
                                        "Communality", "Uniqueness", "Note"])

    # Metric correlation table (pooled and by call type)
    corr_rows = []
    for ct in ["pooled"] + CALL_ORDER:
        sub = long if ct == "pooled" else long[long["CallType"] == ct]
        Xs = sub[["r_spec_pos", "r_amp_pos", "r_env_pos"]].dropna()
        if len(Xs) < 3:
            continue
        c = Xs.corr(method="spearman")
        for i in range(3):
            for j in range(i + 1, 3):
                corr_rows.append({"Group": ct, "Metric_1": metric_labels[i],
                                  "Metric_2": metric_labels[j],
                                  "Spearman_rho": float(c.iloc[i, j]),
                                  "N": len(Xs)})
    corr_table = pd.DataFrame(corr_rows)

    weights = {"pca": w_pca, "entropy": w_ent, "equal": w_eq}
    return weights, pca_var, table, corr_table


def fuse_rows(X, w, impute=None, geo=False):
    """Fuse the three metric columns row-wise; impute None uses available metrics."""
    X = np.asarray(X, float)
    n = len(X)
    out = np.full(n, np.nan)
    w = np.asarray(w, float)
    for i in range(n):
        row = X[i]
        m = np.isfinite(row)
        if impute is None:
            if not m.any():
                continue
            v = row[m]
            ww = w[m]
        else:
            v = np.where(m, row, impute)
            ww = w
        ww = ww / ww.sum()
        if geo:
            out[i] = np.exp(np.sum(ww * np.log(np.clip(v, 1e-6, 1.0))))
        else:
            out[i] = np.sum(ww * v)
    return np.clip(out, 0.0, 1.0)


def apply_fusion(long, weights):
    X = long[["r_spec_pos", "r_amp_pos", "r_env_pos"]].values
    long["F_pca"] = fuse_rows(X, weights["pca"])
    long["F_equal"] = fuse_rows(X, weights["equal"])
    long["F_entropy"] = fuse_rows(X, weights["entropy"])
    long["F_geo"] = fuse_rows(X, weights["pca"], geo=True)
    long["F_main"] = long["F_pca"]
    long["F_worst"] = fuse_rows(X, weights["pca"], impute=0.0)
    long["F_best"] = fuse_rows(X, weights["pca"], impute=1.0)
    return long


def summarize_fidelity(long):
    rows = []
    for (ct, d), g in long.groupby(["CallType", "Distance"], observed=True):
        row = {"CallType": str(ct), "Distance": int(d), "N_total": int(len(g)),
               "N_valid_F": int(g["F_main"].notna().sum())}
        for m in ["r_spec", "r_amp", "r_env", "F_main", "F_equal", "F_entropy", "F_geo"]:
            v = g[m].dropna().values.astype(float)
            row[f"{m}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_audit(long):
    rows = []
    for ct in CALL_ORDER:
        for d in DISTANCES:
            sub = long[(long["CallType"] == ct) & (long["Distance"] == d)]
            row = {"CallType": ct, "Distance": d, "N_total": len(sub)}
            for metric in ["spec", "amp", "env"]:
                row[f"N_valid_{metric}"] = int(sub[f"r_{metric}"].notna().sum())
            row["N_missing_any"] = int((sub["n_metrics"] == 0).sum())
            row["Pct_missing_any"] = 100.0 * row["N_missing_any"] / max(1, len(sub))
            rows.append(row)
    return pd.DataFrame(rows)


def write_data_dictionary():
    rows = [
        ("TrialID", "Trial/row index within a call type", "integer"),
        ("CallType", "flat / mod / tv", "category"),
        ("Distance", "Propagation distance", "m"),
        ("r_spec", "Spectrogram correlation with 1 m reference", "correlation"),
        ("r_amp", "Amplitude-spectrum correlation with 1 m reference", "correlation"),
        ("r_env", "Amplitude-envelope correlation with 1 m reference", "correlation"),
        ("r_*_pos", "r_* clipped to [0,1] for fusion", "correlation"),
        ("n_metrics", "Number of available metrics for the trial-distance", "count"),
        ("F_main", "PCA-weighted linear fidelity, available metrics (main)", "index [0,1]"),
        ("F_equal", "Equal-weight linear fidelity", "index [0,1]"),
        ("F_entropy", "Entropy-weight linear fidelity", "index [0,1]"),
        ("F_geo", "Weighted geometric fidelity (short-board penalty)", "index [0,1]"),
        ("F_worst", "Fidelity with missing metrics imputed as 0 (lower bound)", "index [0,1]"),
        ("F_best", "Fidelity with missing metrics imputed as 1 (upper bound)", "index [0,1]"),
    ]
    pd.DataFrame(rows, columns=["Column", "Description", "Unit_Type"]).to_csv(
        os.path.join(OUT_DIR, "P2_data_dictionary.csv"), index=False, encoding="utf-8-sig")


# ==============================================================================
# 3. Degradation models
# ==============================================================================
def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _at_bound(popt, bounds, atol=1e-5):
    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    p = np.asarray(popt, float)
    return bool(np.any(np.isclose(p, lo, atol=atol)) or np.any(np.isclose(p, hi, atol=atol)))


def fit_logexp(d, y):
    """F(d) = F4 - b ln(d/4) - c (d-4)  [reparameterised at 4 m]."""
    def f(x, F4, b, c):
        x = np.asarray(x, float)
        return F4 - b * np.log(x / 4.0) - c * (x - 4.0)
    bounds = ([0, 0, 0], [1.5, 2.0, 0.05])
    try:
        popt, _ = curve_fit(f, d, y, p0=[0.85, 0.05, 0.002],
                            bounds=bounds, maxfev=20000)
    except Exception:
        return None
    return {"name": "Physical log-exp",
            "predict": lambda x: _clip01(f(np.asarray(x, float), *popt)),
            "params": popt, "n_params": 3, "bounds": bounds,
            "boundary": _at_bound(popt, bounds)}


def fit_powerexp(d, y):
    """F(d) = F4 (d/4)^(-b) exp(-c (d-4))."""
    def f(x, F4, b, c):
        x = np.asarray(x, float)
        return F4 * (x / 4.0) ** (-b) * np.exp(-c * (x - 4.0))
    bounds = ([0, 0, 0], [2.0, 2.0, 0.05])
    try:
        popt, _ = curve_fit(f, d, y, p0=[0.85, 0.05, 0.002],
                            bounds=bounds, maxfev=20000)
    except Exception:
        return None
    return {"name": "Physical power-exp",
            "predict": lambda x: _clip01(f(np.asarray(x, float), *popt)),
            "params": popt, "n_params": 3, "bounds": bounds,
            "boundary": _at_bound(popt, bounds)}


def fit_logistic(d, y):
    """F(d) = c + (a-c) / (1 + (d/d0)^b)."""
    def f(x, a, c, d0, b):
        x = np.asarray(x, float)
        return c + (a - c) / (1.0 + (x / d0) ** b)
    bounds = ([0, 0, 1, 0.1], [1.5, 0.6, 1000, 8.0])
    try:
        popt, _ = curve_fit(f, d, y, p0=[0.90, 0.10, 50.0, 1.5],
                            bounds=bounds, maxfev=20000)
    except Exception:
        return None
    return {"name": "Logistic",
            "predict": lambda x: _clip01(f(np.asarray(x, float), *popt)),
            "params": popt, "n_params": 4, "bounds": bounds,
            "boundary": _at_bound(popt, bounds)}


def fit_logistic_sigma(d, y, sigma):
    def f(x, a, c, d0, b):
        x = np.asarray(x, float)
        return c + (a - c) / (1.0 + (x / d0) ** b)
    bounds = ([0, 0, 1, 0.1], [1.5, 0.6, 1000, 8.0])
    try:
        popt, _ = curve_fit(f, d, y, p0=[0.90, 0.10, 50.0, 1.5], sigma=sigma,
                            absolute_sigma=False, bounds=bounds, maxfev=20000)
    except Exception:
        return None
    return {"name": "Logistic weighted",
            "predict": lambda x: _clip01(f(np.asarray(x, float), *popt)),
            "params": popt, "n_params": 4, "bounds": bounds,
            "boundary": _at_bound(popt, bounds)}


def fit_weibull(d, y):
    def f(x, lam, k):
        return np.exp(-(np.asarray(x, float) / lam) ** k)
    bounds = ([1, 0.1], [5000, 10])
    try:
        popt, _ = curve_fit(f, d, y, p0=[100.0, 1.0], bounds=bounds, maxfev=20000)
    except Exception:
        return None
    return {"name": "Weibull",
            "predict": lambda x: _clip01(f(np.asarray(x, float), *popt)),
            "params": popt, "n_params": 2, "bounds": bounds,
            "boundary": _at_bound(popt, bounds)}


def fit_gam(d, y, n_knots=5):
    X = np.log(np.asarray(d, float)).reshape(-1, 1)
    pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False,
                                     extrapolation="linear")),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20)))
    ])
    try:
        pipe.fit(X, y)
    except Exception:
        return None
    n_params = pipe.named_steps["spline"].transform(X).shape[1]
    return {"name": "GAM spline",
            "predict": lambda x: _clip01(pipe.predict(
                np.log(np.asarray(x, float)).reshape(-1, 1))),
            "params": pipe, "n_params": n_params, "bounds": None,
            "boundary": False}


def fit_isotonic(d, y):
    """Monotone decreasing isotonic envelope; used as the robust main range model."""
    try:
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(np.asarray(d, float), np.asarray(y, float))
    except Exception:
        return None
    n_params = max(1, len(getattr(iso, "X_thresholds_", [1])))
    return {"name": "Isotonic (monotone)",
            "predict": lambda x: _clip01(iso.predict(np.asarray(x, float))),
            "params": np.array([]), "n_params": n_params,
            "bounds": None, "boundary": False,
            "iso": iso}


def fit_isotonic_weighted(d, y, w):
    """Isotonic fit with sample weights (used in the weighted-fit sensitivity)."""
    try:
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(np.asarray(d, float), np.asarray(y, float),
                sample_weight=np.asarray(w, float))
    except Exception:
        return None
    return {"name": "Isotonic weighted",
            "predict": lambda x: _clip01(iso.predict(np.asarray(x, float))),
            "params": np.array([]), "n_params": 1,
            "bounds": None, "boundary": False, "iso": iso}


def fit_monotone_logistic(d, y):
    """
    Monotone decreasing logistic for P(Y=1|d):
        logit p = a + b log(d),  b <= 0.
    Fitted by penalised Bernoulli MLE; used for the common-trial probability
    curve so that R_tau(p) is smooth and has finite bootstrap uncertainty.
    """
    x = np.log(np.asarray(d, float))
    y = np.asarray(y, float)

    def nll(theta):
        a, b = theta
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        eps = 1e-9
        return float(-np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps))
                     + 1e-4 * (a * a + b * b))

    try:
        res = minimize(nll, x0=[1.0, -1.0], method="L-BFGS-B",
                       bounds=[(-20.0, 20.0), (-20.0, 0.0)])
    except Exception:
        return None
    if not res.success:
        return None
    a, b = res.x
    return {"a": float(a), "b": float(b),
            "predict": lambda dd: 1.0 / (1.0 + np.exp(-np.clip(
                a + b * np.log(np.asarray(dd, float)), -50.0, 50.0)))}


MODEL_FITTERS = {
    "Isotonic (monotone)": fit_isotonic,
    "Physical log-exp": fit_logexp,
    "Physical power-exp": fit_powerexp,
    "Logistic": fit_logistic,
    "Weibull": fit_weibull,
    "GAM spline": fit_gam,
}

PARAM_NAMES = {
    "Physical log-exp": ["F4", "b", "c"],
    "Physical power-exp": ["F4", "b", "c"],
    "Logistic": ["a", "c", "d0", "b"],
    "Weibull": ["lambda", "k"],
}
PARAM_DESC = {
    "F4": "fitted fidelity at 4 m (within the observed range)",
    "a": "upper asymptote (logistic)",
    "b": "log-distance loss / power exponent / logistic steepness",
    "c": "linear excess absorption / logistic floor",
    "d0": "half-decay distance (logistic)",
    "lambda": "scale distance (Weibull)",
    "k": "shape parameter (Weibull)",
}


def pick_main(models):
    for name in MAIN_MODEL_PREFERENCE:
        if name in models:
            return name
    return list(models.keys())[0]


def fit_models_for_ct(df_ct):
    d = df_ct["Distance"].values.astype(float)
    y = df_ct["F_main"].values.astype(float)
    trial = df_ct["TrialID"].values
    mask = np.isfinite(d) & np.isfinite(y)
    d, y, trial = d[mask], y[mask], trial[mask]
    models = {}
    for name, fitter in MODEL_FITTERS.items():
        m = fitter(d, y)
        if m is None:
            print(f"  [warn] {name} failed to converge")
        else:
            models[name] = m
    return models, d, y, trial


def cv_rmse(fitter, d, y, trial, n_splits=N_SPLITS, seed=SEED):
    unique = np.unique(trial)
    if len(unique) < n_splits:
        return np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errs = []
    for tr, te in kf.split(unique):
        train = np.isin(trial, unique[tr])
        test = ~train
        if train.sum() < 10 or test.sum() < 5:
            continue
        m = fitter(d[train], y[train])
        if m is None:
            continue
        yp = m["predict"](d[test])
        errs.append((y[test] - yp) ** 2)
    if not errs:
        return np.nan
    return float(np.sqrt(np.concatenate(errs).mean()))


def model_fit_metrics(models, d, y, trial):
    rows = []
    n = len(y)
    for name, m in models.items():
        yhat = m["predict"](d)
        resid = y - yhat
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        rmse = float(np.sqrt(sse / n))
        mae = float(np.mean(np.abs(resid)))
        r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
        cv = cv_rmse(MODEL_FITTERS[name], d, y, trial)
        rows.append({
            "Model": name, "n_points": n, "n_trials": int(len(np.unique(trial))),
            "n_params": int(m["n_params"]), "RMSE": rmse, "MAE": mae, "R2": r2,
            "CV_RMSE": cv, "Boundary": bool(m.get("boundary", False)),
            "Note": "CV-RMSE is the primary criterion; AIC/BIC omitted because "
                    "row-level n ignores trial clustering"
        })
    out = pd.DataFrame(rows)
    best = out["CV_RMSE"].min()
    out["PoorFit"] = (out["R2"] < 0.60) | (out["CV_RMSE"] > 1.5 * best)
    out["Best_CV"] = out["CV_RMSE"] == best
    return out


def compute_model_average(models, metrics):
    elig = metrics[(~metrics["Boundary"]) & (~metrics["PoorFit"]) &
                   metrics["CV_RMSE"].notna()]
    if len(elig) == 0:
        elig = metrics[metrics["CV_RMSE"].notna() & (metrics["R2"] >= 0.5)]
    if len(elig) == 0:
        names = list(models.keys())
        w = pd.Series(np.ones(len(names)) / len(names), index=names)
    else:
        inv = 1.0 / np.maximum(elig["CV_RMSE"].values, 1e-9) ** 2
        w = pd.Series(inv / inv.sum(), index=elig["Model"].values)
    curve = np.zeros_like(GRID)
    for name, weight in w.items():
        curve += weight * models[name]["predict"](GRID)
    return _clip01(curve), w, elig["Model"].tolist()


# ==============================================================================
# 4. Bootstrap with fusion-weight uncertainty
# ==============================================================================
def bootstrap_main(df_ct, fitter, n_boot=N_BOOT, seed=SEED):
    Xraw = df_ct[["r_spec_pos", "r_amp_pos", "r_env_pos"]].values.astype(float)
    d = df_ct["Distance"].values.astype(float)
    trial = df_ct["TrialID"].values
    unique = np.unique(trial)
    idx_by = {t: np.where(trial == t)[0] for t in unique}
    rng = np.random.default_rng(seed)
    curves, params, weights, n_fail = [], [], [], 0
    for _ in range(n_boot):
        samp = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([idx_by[t] for t in samp])
        Xb = Xraw[idx]
        wb, _, _ = pca_weights_from_matrix(Xb)
        Fb = fuse_rows(Xb, wb)
        mask = np.isfinite(d[idx]) & np.isfinite(Fb)
        if mask.sum() < 10:
            n_fail += 1
            continue
        m = fitter(d[idx][mask], Fb[mask])
        if m is None:
            n_fail += 1
            continue
        curves.append(m["predict"](GRID))
        try:
            params.append(np.asarray(m.get("params", []), float))
        except Exception:
            params.append(np.array([]))
        weights.append(wb)
    return (np.asarray(curves), np.asarray(params),
            np.asarray(weights), n_fail)


# ==============================================================================
# 5. Effective range, missingness bounds, probabilistic range, crossover
# ==============================================================================
def solve_range(curve, tau, interp="log"):
    curve = np.asarray(curve, float)
    if not np.any(np.isfinite(curve)):
        return np.nan, "fit_failed"
    above = curve >= tau
    if not np.any(above):
        return 0.0, "below_4m"
    if np.all(above):
        return float(GRID[-1]), "above_256m"
    idx = int(np.where(above)[0][-1])
    if idx >= len(GRID) - 1:
        return float(GRID[-1]), "above_256m"
    if interp == "log":
        x1, x2 = np.log(GRID[idx]), np.log(GRID[idx + 1])
    else:
        x1, x2 = GRID[idx], GRID[idx + 1]
    y1, y2 = curve[idx], curve[idx + 1]
    if y2 == y1:
        return float(GRID[idx]), "interpolated"
    t = (tau - y1) / (y2 - y1)
    x = x1 + t * (x2 - x1)
    return float(np.exp(x) if interp == "log" else x), "interpolated"


def solve_prob_range(ds, ps, level):
    ds = np.asarray(ds, float)
    ps = np.asarray(ps, float)
    above = ps >= level
    if not np.any(above):
        return 0.0, "below_4m"
    if np.all(above):
        return float(ds[-1]), "above_256m"
    idx = int(np.where(above)[0][-1])
    if idx >= len(ds) - 1:
        return float(ds[-1]), "above_256m"
    x1, x2 = np.log(ds[idx]), np.log(ds[idx + 1])
    y1, y2 = ps[idx], ps[idx + 1]
    if y2 == y1:
        return float(ds[idx]), "interpolated"
    t = (level - y1) / (y2 - y1)
    return float(np.exp(x1 + t * (x2 - x1))), "interpolated"


def effective_range_table(curves_main, boot_curves):
    rows = []
    for ct in CALL_ORDER:
        for level, tau in LEVELS.items():
            r, status = solve_range(curves_main[ct], tau)
            br = np.array([solve_range(c, tau)[0] for c in boot_curves[ct]], float) \
                if len(boot_curves[ct]) else np.array([])
            br = br[np.isfinite(br)]
            lo, hi = (np.percentile(br, [2.5, 97.5]) if len(br) >= 10 else (np.nan, np.nan))
            rows.append({"CallType": ct, "Level": level, "tau": tau,
                         "Range_m": r, "CI_Low": float(lo), "CI_High": float(hi),
                         "Status": status, "Censored": status != "interpolated",
                         "N_boot_valid": int(len(br))})
    out = pd.DataFrame(rows)
    out["Level"] = pd.Categorical(out["Level"], categories=list(LEVELS.keys()), ordered=True)
    return out


def model_averaged_range_table(models_by_ct, avg_curves, weights_by_ct, eligible_by_ct):
    rows = []
    for ct in CALL_ORDER:
        models = models_by_ct[ct]
        for tau in TAU_GRID:
            r_avg, status = solve_range(avg_curves[ct], tau)
            all_r = [solve_range(m["predict"](GRID), tau)[0] for m in models.values()]
            all_r = np.array(all_r, float)
            all_r = all_r[np.isfinite(all_r)]
            elig = [n for n in eligible_by_ct[ct] if n in models]
            elig_r = [solve_range(models[n]["predict"](GRID), tau)[0] for n in elig]
            elig_r = np.array(elig_r, float)
            elig_r = elig_r[np.isfinite(elig_r)]
            rows.append({
                "CallType": ct, "tau": tau, "Range_avg": r_avg, "Status": status,
                "Range_min_all": float(np.min(all_r)) if len(all_r) else np.nan,
                "Range_max_all": float(np.max(all_r)) if len(all_r) else np.nan,
                "Range_min_eligible": float(np.min(elig_r)) if len(elig_r) else np.nan,
                "Range_max_eligible": float(np.max(elig_r)) if len(elig_r) else np.nan,
            })
    return pd.DataFrame(rows)


def missingness_bounds_table(long, main_names):
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        d = sub["Distance"].values.astype(float)
        trial = sub["TrialID"].values
        for col, label in [("F_main", "complete_case"),
                           ("F_worst", "worst_case"),
                           ("F_best", "best_case")]:
            y = sub[col].values.astype(float)
            mask = np.isfinite(d) & np.isfinite(y)
            fitter = fit_isotonic
            m = fitter(d[mask], y[mask])
            if m is None:
                continue
            curve = m["predict"](GRID)
            yhat = m["predict"](d[mask])
            sse = float(np.sum((y[mask] - yhat) ** 2))
            sst = float(np.sum((y[mask] - y[mask].mean()) ** 2))
            r2 = 1.0 - sse / sst if sst > 0 else np.nan
            reliable = bool(np.isfinite(r2) and r2 >= 0.60)
            for level, tau in LEVELS.items():
                r, status = solve_range(curve, tau)
                rows.append({"CallType": ct, "Scenario": label, "Level": level,
                             "tau": tau, "Range_m": r, "Status": status,
                             "R2": r2, "Reliable": reliable,
                             "Method": "Isotonic (monotone)"})
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(os.path.join(OUT_DIR, "P2_missingness_bounds.csv"),
                   index=False, encoding="utf-8-sig")
        piv = out.pivot_table(index=["CallType", "Level", "tau"],
                              columns="Scenario", values="Range_m").reset_index()
        if "worst_case" in piv.columns and "best_case" in piv.columns:
            piv["Bound_width"] = piv["best_case"] - piv["worst_case"]
        piv.to_csv(os.path.join(OUT_DIR, "P2_missingness_bounds_wide.csv"),
                   index=False, encoding="utf-8-sig")
    return out


def _probabilistic_range_table_isotonic(long, n_boot=500, seed=SEED):
    """
    Monotone common-trial recognition probability.
    Only trials valid at all distances are used, and P(F>=tau|d) is estimated
    by isotonic regression (monotone decreasing), so the range is well defined
    even though the raw proportion curves are non-monotonic.
    """
    rows = []
    curves = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        piv = sub.pivot_table(index="TrialID", columns="Distance", values="F_main")
        common = piv.dropna()
        n_common = len(common)
        if n_common < 20:
            continue
        trials = common.index.values
        for tau in [0.50, 0.70]:
            X, Y = [], []
            for d in DISTANCES:
                v = common[d].values.astype(float)
                X.extend([float(d)] * len(v))
                Y.extend((v >= tau).astype(float).tolist())
            X = np.asarray(X, float)
            Y = np.asarray(Y, float)
            try:
                iso = IsotonicRegression(increasing=False, out_of_bounds="clip",
                                         y_min=0.0, y_max=1.0).fit(X, Y)
            except Exception:
                continue
            p_grid = iso.predict(GRID)
            p_dist = iso.predict(np.asarray(DISTANCES, float))
            for d, pd_ in zip(DISTANCES, p_dist):
                curves.append({"CallType": ct, "tau": tau, "Distance": d,
                               "P_recognize": float(pd_), "N_common": n_common})
            point = {pl: solve_prob_range(GRID, p_grid, pl)
                     for pl in [0.50, 0.75, 0.90]}
            rng = np.random.default_rng(seed)
            boot = {pl: [] for pl in [0.50, 0.75, 0.90]}
            for _ in range(n_boot):
                samp = rng.choice(trials, size=len(trials), replace=True)
                cb = common.loc[samp]
                Xb, Yb = [], []
                for d in DISTANCES:
                    v = cb[d].values.astype(float)
                    Xb.extend([float(d)] * len(v))
                    Yb.extend((v >= tau).astype(float).tolist())
                try:
                    ib = IsotonicRegression(increasing=False, out_of_bounds="clip",
                                            y_min=0.0, y_max=1.0).fit(
                        np.asarray(Xb, float), np.asarray(Yb, float))
                    pb = ib.predict(GRID)
                    for pl in boot:
                        boot[pl].append(solve_prob_range(GRID, pb, pl)[0])
                except Exception:
                    continue
            for pl in [0.50, 0.75, 0.90]:
                r, status = point[pl]
                arr = np.asarray(boot[pl], float)
                arr = arr[np.isfinite(arr)]
                lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) >= 10
                          else (np.nan, np.nan))
                rows.append({"CallType": ct, "tau": tau, "p_level": pl,
                             "Range_m": r, "Status": status,
                             "CI_Low": float(lo), "CI_High": float(hi),
                             "N_common": n_common, "N_boot_valid": int(len(arr)),
                             "Method": "Isotonic common-trial"})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def probabilistic_range_table(long, n_boot=500, seed=SEED):
    """
    Smooth monotone common-trial probability:
        logit P(F>=tau | d) = a + b log(d),  b <= 0.
    This replaces the isotonic step probability, which produced zero-width
    CIs (e.g. tv [192,192]) and extreme p=0.90 ranges.
    """
    rows = []
    curves = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        piv = sub.pivot_table(index="TrialID", columns="Distance", values="F_main")
        common = piv.dropna()
        n_common = len(common)
        if n_common < 20:
            continue
        trials = common.index.values
        for tau in [0.50, 0.70]:
            X, Y = [], []
            for d in DISTANCES:
                v = common[d].values.astype(float)
                X.extend([float(d)] * len(v))
                Y.extend((v >= tau).astype(float).tolist())
            X = np.asarray(X, float)
            Y = np.asarray(Y, float)
            m = fit_monotone_logistic(X, Y)
            if m is None:
                continue
            p_grid = m["predict"](GRID)
            p_dist = m["predict"](np.asarray(DISTANCES, float))
            p_emp = np.array([np.mean(common[d].values.astype(float) >= tau)
                              for d in DISTANCES], float)
            for d, pd_, pe in zip(DISTANCES, p_dist, p_emp):
                k = int(round(pe * n_common))
                ci_lo, ci_hi = wilson_ci(k, n_common)
                curves.append({"CallType": ct, "tau": tau, "Distance": float(d),
                               "P_recognize": float(pd_),
                               "P_empirical": float(pe),
                               "CI_Low": float(ci_lo), "CI_High": float(ci_hi),
                               "N_common": n_common})
            point = {pl: solve_prob_range(GRID, p_grid, pl)
                     for pl in [0.50, 0.75, 0.90]}
            rng = np.random.default_rng(seed)
            boot = {pl: [] for pl in [0.50, 0.75, 0.90]}
            for _ in range(n_boot):
                samp = rng.choice(trials, size=len(trials), replace=True)
                cb = common.loc[samp]
                Xb, Yb = [], []
                for d in DISTANCES:
                    v = cb[d].values.astype(float)
                    Xb.extend([float(d)] * len(v))
                    Yb.extend((v >= tau).astype(float).tolist())
                mb = fit_monotone_logistic(np.asarray(Xb, float),
                                           np.asarray(Yb, float))
                if mb is None:
                    continue
                pb = mb["predict"](GRID)
                for pl in boot:
                    boot[pl].append(solve_prob_range(GRID, pb, pl)[0])
            for pl in [0.50, 0.75, 0.90]:
                r, status = point[pl]
                arr = np.asarray(boot[pl], float)
                arr = arr[np.isfinite(arr)]
                lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) >= 10
                          else (np.nan, np.nan))
                above = p_emp >= pl
                changes = np.where(above[:-1] != above[1:])[0]
                if len(changes) == 0:
                    if np.all(above):
                        b_low = b_high = float(DISTANCES[-1])
                        b_status = "above_256m"
                    else:
                        b_low, b_high = 0.0, float(DISTANCES[0])
                        b_status = "below_4m"
                else:
                    b_low = float(DISTANCES[int(changes[0])])
                    b_high = float(DISTANCES[int(changes[-1]) + 1])
                    b_status = ("multiple_crossings" if len(changes) > 1
                                else "single_crossing")
                model_inside = bool(b_low <= r <= b_high)
                rows.append({"CallType": ct, "tau": tau, "p_level": pl,
                             "Range_m": r, "Status": status,
                             "CI_Low": float(lo), "CI_High": float(hi),
                             "Bracket_Low": b_low, "Bracket_High": b_high,
                             "Bracket_Status": b_status,
                             "N_Empirical_Crossings": int(len(changes)),
                             "Interval_Censored": bool(b_high > b_low),
                             "Model_Inside_Bracket": model_inside,
                             "Recommended": ("bracket" if not model_inside
                                             else "bracket (model inside)"),
                             "N_common": n_common, "N_boot_valid": int(len(arr)),
                             "Method": "Monotone logistic (common-trial)"})
    return pd.DataFrame(rows), pd.DataFrame(curves)


def crossover_from_curves(curve_tv, curve_mod, tau_grid):
    r_tv = np.array([solve_range(curve_tv, t)[0] for t in tau_grid], float)
    r_mod = np.array([solve_range(curve_mod, t)[0] for t in tau_grid], float)
    diff = r_tv - r_mod
    for i in range(len(tau_grid) - 1):
        if np.isfinite(diff[i]) and np.isfinite(diff[i + 1]) and diff[i] * diff[i + 1] <= 0 \
                and diff[i] != diff[i + 1]:
            t = (0 - diff[i]) / (diff[i + 1] - diff[i])
            tau = tau_grid[i] + t * (tau_grid[i + 1] - tau_grid[i])
            r = r_tv[i] + t * (r_tv[i + 1] - r_tv[i])
            return float(tau), float(r), "crossed"
    return np.nan, np.nan, "no_crossing"


def crossover_table(curves_main, boot_curves):
    tau_star, r_star, status = crossover_from_curves(
        curves_main["tv"], curves_main["mod"], TAU_GRID_FINE)
    ts, rs = [], []
    n = min(len(boot_curves["tv"]), len(boot_curves["mod"]))
    for i in range(n):
        t, r, s = crossover_from_curves(boot_curves["tv"][i],
                                        boot_curves["mod"][i], TAU_GRID_FINE)
        if np.isfinite(t):
            ts.append(t)
            rs.append(r)
    ts = np.array(ts)
    rs = np.array(rs)
    t_lo, t_hi = (np.percentile(ts, [2.5, 97.5]) if len(ts) >= 10 else (np.nan, np.nan))
    r_lo, r_hi = (np.percentile(rs, [2.5, 97.5]) if len(rs) >= 10 else (np.nan, np.nan))
    return pd.DataFrame([{
        "Model": "Main isotonic", "tau_star": tau_star, "Range_star": r_star,
        "tau_star_CI_Low": float(t_lo), "tau_star_CI_High": float(t_hi),
        "Range_star_CI_Low": float(r_lo), "Range_star_CI_High": float(r_hi),
        "Status": status, "N_boot_valid": int(len(ts))
    }])


def metric_specific_range_table(long):
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        d = sub["Distance"].values.astype(float)
        for col, label in [("r_spec", "Spectrogram"), ("r_amp", "Amplitude spectrum"),
                           ("r_env", "Envelope")]:
            y = sub[col].values.astype(float)
            mask = np.isfinite(d) & np.isfinite(y)
            m = fit_isotonic(d[mask], y[mask])
            if m is None:
                continue
            for tau in [0.50, 0.70]:
                r, status = solve_range(m["predict"](GRID), tau)
                rows.append({"CallType": ct, "Metric": label, "tau": tau,
                             "Range_m": r, "Status": status,
                             "Method": "Isotonic (monotone)",
                             "Interpretation": ("threshold not reached at 4 m"
                                                if status == "below_4m" else "")})
    return pd.DataFrame(rows)


def weighted_fit_table(long):
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        d = sub["Distance"].values.astype(float)
        y = sub["F_main"].values.astype(float)
        sd_by_d = sub.groupby("Distance")["F_main"].std()
        sigma = sub["Distance"].map(sd_by_d).fillna(sub["F_main"].std()).values
        mask = np.isfinite(d) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        w = 1.0 / np.maximum(sigma[mask], 1e-6) ** 2
        m = fit_isotonic_weighted(d[mask], y[mask], w)
        if m is None:
            continue
        for tau in [0.50, 0.70]:
            r, status = solve_range(m["predict"](GRID), tau)
            rows.append({"CallType": ct, "Fit": "Weighted isotonic", "tau": tau,
                         "Range_m": r, "Status": status})
        m0 = fit_isotonic(d[mask], y[mask])
        if m0 is not None:
            for tau in [0.50, 0.70]:
                r, status = solve_range(m0["predict"](GRID), tau)
                rows.append({"CallType": ct, "Fit": "Unweighted isotonic", "tau": tau,
                             "Range_m": r, "Status": status})
    return pd.DataFrame(rows)


def gam_knot_table(long):
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        d = sub["Distance"].values.astype(float)
        y = sub["F_main"].values.astype(float)
        mask = np.isfinite(d) & np.isfinite(y)
        for k in [4, 5, 6]:
            m = fit_gam(d[mask], y[mask], n_knots=k)
            if m is None:
                continue
            for tau in [0.50, 0.70]:
                r, status = solve_range(m["predict"](GRID), tau)
                rows.append({"CallType": ct, "n_knots": k, "tau": tau,
                             "Range_m": r, "Status": status})
    return pd.DataFrame(rows)


def model_band_table(models_by_ct):
    rows = []
    for ct in CALL_ORDER:
        for tau in TAU_GRID:
            rs = [solve_range(m["predict"](GRID), tau)[0]
                  for m in models_by_ct[ct].values()]
            rs = np.array(rs, float)
            rs = rs[np.isfinite(rs)]
            rows.append({"CallType": ct, "tau": tau,
                         "Range_min": float(np.min(rs)) if len(rs) else np.nan,
                         "Range_max": float(np.max(rs)) if len(rs) else np.nan})
    return pd.DataFrame(rows)


def fusion_sensitivity_table(long):
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        d = sub["Distance"].values.astype(float)
        for col, label in FUSIONS:
            y = sub[col].values.astype(float)
            mask = np.isfinite(d) & np.isfinite(y)
            m = fit_isotonic(d[mask], y[mask])
            if m is None:
                continue
            for tau in [0.50, 0.70]:
                r, status = solve_range(m["predict"](GRID), tau)
                rows.append({"CallType": ct, "Fusion": label, "tau": tau,
                             "Range_m": r, "Status": status})
    return pd.DataFrame(rows)


# ==============================================================================
# 6. Playback calibration (adjusted logistic)
# ==============================================================================
def playback_calibration(models_by_ct, main_names):
    path = glob.glob(os.path.join(DATA_DIR, "*Data02*"))
    if not path:
        print("[Playback calibration] Data02 not found; skipped.")
        return pd.DataFrame()
    df = pd.read_csv(path[0], na_values=["NA", "na", "NaN", "null", "", " "])
    df.columns = [str(c).strip() for c in df.columns]
    need = {"CallType", "Context", "Distance", "Npenguins", "ResponseYesNo"}
    if not need.issubset(df.columns):
        print("[Playback calibration] Data02 lacks required columns; skipped.")
        return pd.DataFrame()
    df["CallType"] = df["CallType"].astype(str).str.lower()
    df = df[df["CallType"].isin(CALL_ORDER)].copy()
    df["Context"] = pd.Categorical(df["Context"].astype(str),
                                   categories=["commuting", "after", "searching"])
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df["Npenguins"] = pd.to_numeric(df["Npenguins"], errors="coerce")
    df["ResponseYesNo"] = pd.to_numeric(df["ResponseYesNo"], errors="coerce")
    df = df.dropna(subset=["Distance", "Npenguins", "ResponseYesNo", "Context"])
    if len(df) < 20:
        print("[Playback calibration] too few rows; skipped.")
        return pd.DataFrame()

    # Model-derived fidelity at the playback distance
    F = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples()):
        ct = row.CallType
        name = main_names.get(ct)
        if name is None or ct not in models_by_ct:
            continue
        F[i] = float(models_by_ct[ct][name]["predict"]([row.Distance])[0])
    df["F_model"] = F
    df = df.dropna(subset=["F_model"])
    if len(df) < 20:
        return pd.DataFrame()

    def design(frame):
        X = pd.get_dummies(frame[["Context"]], drop_first=True).astype(float)
        X["F_model"] = frame["F_model"].values
        X["Npenguins"] = frame["Npenguins"].values
        return X

    X = design(df)
    y = df["ResponseYesNo"].values.astype(float)
    try:
        m = LogisticRegression(C=1e6, max_iter=5000).fit(X.values, y)
    except Exception:
        print("[Playback calibration] logistic fit failed; skipped.")
        return pd.DataFrame()

    f_grid = np.linspace(0, 1, 201)
    ctx_ref = X.iloc[0:1].copy()
    for c in ctx_ref.columns:
        if c.startswith("Context_"):
            ctx_ref[c] = 0.0
    n_med = float(df["Npenguins"].median())

    def threshold_from_model(model, fg):
        Z = pd.DataFrame(np.tile(ctx_ref.values, (len(fg), 1)), columns=X.columns)
        Z["F_model"] = fg
        Z["Npenguins"] = n_med
        p = model.predict_proba(Z.values)[:, 1]
        out = {}
        for level in [0.5, 0.7]:
            idx = np.where(p >= level)[0]
            if len(idx) == 0:
                out[level] = np.nan
            else:
                j = idx[0]
                if j == 0:
                    out[level] = float(fg[0])
                else:
                    t = (level - p[j - 1]) / (p[j] - p[j - 1])
                    out[level] = float(fg[j - 1] + t * (fg[j] - fg[j - 1]))
        return out

    point = threshold_from_model(m, f_grid)

    # Cluster bootstrap by expID
    boot = []
    if "expID" in df.columns:
        exp = df["expID"].astype(str).values
        uniq = np.unique(exp)
        rng = np.random.default_rng(SEED)
        for _ in range(500):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(exp == e)[0] for e in samp])
            if len(np.unique(y[idx])) < 2:
                continue
            try:
                mb = LogisticRegression(C=1e6, max_iter=5000).fit(X.values[idx], y[idx])
                out = threshold_from_model(mb, f_grid)
                boot.append([out[0.5], out[0.7]])
            except Exception:
                continue
    boot = np.asarray(boot, float)
    ci = (np.nanpercentile(boot, [2.5, 97.5], axis=0) if len(boot) >= 10
          else np.full((2, 2), np.nan))
    rows = [
        {"Quantity": "F at response probability 0.5", "Estimate": point[0.5],
         "CI_Low": float(ci[0, 0]), "CI_High": float(ci[1, 0]), "N": len(df)},
        {"Quantity": "F at response probability 0.7", "Estimate": point[0.7],
         "CI_Low": float(ci[0, 1]), "CI_High": float(ci[1, 1]), "N": len(df)},
    ]
    out = pd.DataFrame(rows)
    out["Note"] = ("Adjusted for Context and Npenguins; cluster bootstrap by expID. "
                   "Descriptive calibration; Problem 3 owns the formal behavioural model.")
    return out


def playback_descriptive(models_by_ct, main_names):
    """
    Descriptive playback check only. The earlier adjusted-logistic calibration
    was not identified (95% CIs spanned [0,1]), so we do not report threshold
    estimates; Problem 3 owns the formal behavioural model.
    """
    path = glob.glob(os.path.join(DATA_DIR, "*Data02*"))
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path[0], na_values=["NA", "na", "NaN", "null", "", " "])
    df.columns = [str(c).strip() for c in df.columns]
    need = {"CallType", "Distance", "Npenguins", "ResponseYesNo"}
    if not need.issubset(df.columns):
        return pd.DataFrame()
    df["CallType"] = df["CallType"].astype(str).str.lower()
    df = df[df["CallType"].isin(CALL_ORDER)].copy()
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    df["ResponseYesNo"] = pd.to_numeric(df["ResponseYesNo"], errors="coerce")
    df = df.dropna(subset=["Distance", "ResponseYesNo"])
    F = np.full(len(df), np.nan)
    for i, row in enumerate(df.itertuples()):
        ct = row.CallType
        name = main_names.get(ct)
        if name is None or ct not in models_by_ct:
            continue
        F[i] = float(models_by_ct[ct][name]["predict"]([row.Distance])[0])
    df["F_model"] = F
    df = df.dropna(subset=["F_model"])
    if len(df) < 20:
        return pd.DataFrame()

    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    rows = []
    for ct in ["pooled"] + CALL_ORDER:
        sub = df if ct == "pooled" else df[df["CallType"] == ct]
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = sub[(sub["F_model"] >= lo) & (sub["F_model"] < hi)]
            if len(b) == 0:
                continue
            rows.append({"Group": ct, "F_bin": f"{lo:.1f}-{hi:.1f}",
                         "F_mid": 0.5 * (lo + min(hi, 1.0)),
                         "N": len(b), "Response_rate": float(b["ResponseYesNo"].mean())})
    return pd.DataFrame(rows)


# ==============================================================================
# 7. Robustness and pairwise differences
# ==============================================================================
def robustness_table(avg_ranges, band, avg_curves, main_ranges):
    rows = []
    for ct in CALL_ORDER:
        a50 = avg_ranges[(avg_ranges["CallType"] == ct) & (avg_ranges["tau"] == 0.50)].iloc[0]
        a70 = avg_ranges[(avg_ranges["CallType"] == ct) & (avg_ranges["tau"] == 0.70)].iloc[0]
        m50 = main_ranges[(main_ranges["CallType"] == ct) & (main_ranges["Level"] == "Potential")].iloc[0]
        m70 = main_ranges[(main_ranges["CallType"] == ct) & (main_ranges["Level"] == "Reliable")].iloc[0]
        curve = avg_curves[ct]
        auc = float(np.trapz(curve, np.log(GRID)) / (np.log(256) - np.log(4)))
        rows.append({
            "CallType": ct,
            "d50_model_avg": a50["Range_avg"],
            "d50_min_all": a50["Range_min_all"], "d50_max_all": a50["Range_max_all"],
            "d70_model_avg": a70["Range_avg"],
            "d70_min_all": a70["Range_min_all"], "d70_max_all": a70["Range_max_all"],
            "d50_logistic": m50["Range_m"], "d70_logistic": m70["Range_m"],
            "AUC_log": auc,
            "Rank_d70": np.nan
        })
    out = pd.DataFrame(rows)
    out["Rank_d70"] = out["d70_model_avg"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("Rank_d70").reset_index(drop=True)


def pairwise_differences(boot_curves, taus=(0.50, 0.70)):
    pairs = [("tv", "flat"), ("tv", "mod"), ("mod", "flat")]
    rows = []
    for tau in taus:
        ranges = {ct: np.array([solve_range(c, tau)[0] for c in boot_curves[ct]], float)
                  for ct in CALL_ORDER}
        for a, b in pairs:
            da, db = ranges[a], ranges[b]
            n = min(len(da), len(db))
            if n == 0:
                continue
            diff = da[:n] - db[:n]
            diff = diff[np.isfinite(diff)]
            if len(diff) < 10:
                continue
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"Comparison": f"{a} - {b}", "tau": tau,
                         "Diff_m": float(diff.mean()), "CI_Low": float(lo),
                         "CI_High": float(hi), "P_diff_gt_0": float(np.mean(diff > 0)),
                         "N_boot": int(len(diff))})
    return pd.DataFrame(rows)


def monotonicity_diagnostic(long):
    """Raw mean curve vs its isotonic monotone envelope."""
    rows = []
    for ct in CALL_ORDER:
        sub = long[long["CallType"] == ct]
        means, sds, ns = [], [], []
        for d in DISTANCES:
            v = sub[sub["Distance"] == d]["F_main"].dropna().values.astype(float)
            means.append(float(v.mean()) if len(v) else np.nan)
            sds.append(float(v.std(ddof=1)) if len(v) > 1 else np.nan)
            ns.append(len(v))
        means = np.asarray(means, float)
        diffs = np.diff(means)
        viol = int(np.sum(diffs > 0))
        maxjump = float(np.max(np.maximum(diffs, 0))) if len(diffs) else np.nan
        try:
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                np.asarray(DISTANCES, float), means)
            fit = iso.predict(np.asarray(DISTANCES, float))
            rmse = float(np.sqrt(np.mean((means - fit) ** 2)))
        except Exception:
            fit = means
            rmse = np.nan
        rows.append({"CallType": ct, "N_violations": viol,
                     "Max_upward_jump": maxjump,
                     "RMSE_raw_vs_isotonic": rmse})
    return pd.DataFrame(rows)


def uncertainty_decomposition(eff, boot_curves, band, bounds, fs, curves_main, avg_ranges):
    """
    Tornado-style decomposition of d70 uncertainty at tau = 0.70.
    Model choice uses eligible models only; missingness uses Reliable bounds only.
    If a source is not identified, Width is NaN and Identified=False.
    """
    rows = []
    for ct in CALL_ORDER:
        main_row = eff[(eff["CallType"] == ct) & (eff["Level"] == "Reliable")].iloc[0]
        center = float(main_row["Range_m"])
        sources = {}

        # sampling (main isotonic bootstrap)
        sources["Sampling (bootstrap CI)"] = (float(main_row["CI_Low"]),
                                              float(main_row["CI_High"]))

        # model choice: eligible models only
        ar = avg_ranges[(avg_ranges["CallType"] == ct) & (avg_ranges["tau"] == 0.70)]
        if len(ar) and np.isfinite(ar.iloc[0]["Range_min_eligible"]) \
                and np.isfinite(ar.iloc[0]["Range_max_eligible"]):
            sources["Model choice (eligible)"] = (
                float(ar.iloc[0]["Range_min_eligible"]),
                float(ar.iloc[0]["Range_max_eligible"]))
        else:
            b = band[(band["CallType"] == ct) & (band["tau"] == 0.70)].iloc[0]
            sources["Model choice (eligible)"] = (float(b["Range_min"]),
                                                  float(b["Range_max"]))

        # missingness: reliable bounds only
        bs = bounds[(bounds["CallType"] == ct) & (bounds["Level"] == "Reliable")]
        lo = bs[bs["Scenario"] == "worst_case"]
        hi = bs[bs["Scenario"] == "best_case"]
        if len(lo) and len(hi) and bool(lo.iloc[0]["Reliable"]) and bool(hi.iloc[0]["Reliable"]):
            sources["Missingness (reliable worst-best)"] = (
                float(lo.iloc[0]["Range_m"]), float(hi.iloc[0]["Range_m"]))
        else:
            sources["Missingness (reliable worst-best)"] = (np.nan, np.nan)

        # fusion
        f = fs[(fs["CallType"] == ct) & (fs["tau"] == 0.70)]
        if len(f):
            sources["Fusion method (min-max)"] = (float(f["Range_m"].min()),
                                                  float(f["Range_m"].max()))

        # threshold +/- 0.05
        r65, _ = solve_range(curves_main[ct], 0.65)
        r75, _ = solve_range(curves_main[ct], 0.75)
        sources["Threshold ±0.05"] = (float(r75), float(r65))

        for name, (low, high) in sources.items():
            identified = bool(np.isfinite(low) and np.isfinite(high))
            rows.append({"CallType": ct, "Source": name,
                         "Low": low, "High": high,
                         "Width": abs(high - low) if identified else np.nan,
                         "Identified": identified, "Center": center})
    return pd.DataFrame(rows)


# ==============================================================================
# 8. Figures
# ==============================================================================
def plot_metric_decay(summary):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, metric, label in zip(axes[:3], ["r_spec", "r_amp", "r_env"],
                                 ["Spectrogram", "Amplitude spectrum", "Envelope"]):
        for ct in CALL_ORDER:
            s = summary[summary["CallType"] == ct].sort_values("Distance")
            ax.errorbar(s["Distance"], s[f"{metric}_mean"], yerr=s[f"{metric}_sd"],
                        marker="o", capsize=3, label=ct, color=PALETTE[ct], lw=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Correlation with 1 m reference")
        ax.set_title(f"{label} decay", fontweight="bold")
        if metric == "r_spec":
            ax.legend(title="Call type")
    ax = axes[3]
    for ct in CALL_ORDER:
        s = summary[summary["CallType"] == ct].sort_values("Distance")
        ax.errorbar(s["Distance"], s["F_main_mean"], yerr=s["F_main_sd"],
                    marker="o", capsize=3, label=ct, color=PALETTE[ct], lw=2)
    for tau in LEVELS.values():
        ax.axhline(tau, color="gray", ls="--", lw=0.9, alpha=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(DISTANCES)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Composite fidelity (PCA)")
    ax.set_title("Acoustic fidelity F(d)", fontweight="bold")
    plt.suptitle("Propagation Degradation of the Three Call Types",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig1_metric_decay.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_fits(models_by_ct, summary, main_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, ct in zip(axes, CALL_ORDER):
        s = summary[summary["CallType"] == ct].sort_values("Distance")
        ax.errorbar(s["Distance"], s["F_main_mean"], yerr=s["F_main_sd"],
                    fmt="o", capsize=3, color=PALETTE[ct], label="Observed mean ± SD")
        show = [(main_names[ct], "-", "#222222"), ("Physical log-exp", ":", "#7F7F7F"),
                ("GAM spline", "--", "#D62728")]
        for name, ls, col in show:
            m = models_by_ct[ct].get(name)
            if m is None:
                continue
            ax.plot(GRID, m["predict"](GRID), ls, lw=2, color=col, label=name)
        for tau in LEVELS.values():
            ax.axhline(tau, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance (m)")
        ax.set_title(ct, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Composite fidelity F(d)")
    plt.suptitle("Degradation Models: Main Logistic, Physical Comparator, GAM Benchmark",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig2_model_fits.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_effective_ranges(eff, avg_ranges):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    y = np.arange(len(eff))
    for ax, title in [(axes[0], "Main logistic: bootstrap 95% CI"),
                      (axes[1], "Model average: min–max across models")]:
        ax.set_title(title, fontweight="bold")
        ax.axvline(4, color="gray", ls=":", lw=1)
        ax.axvline(256, color="gray", ls=":", lw=1)
        ax.set_xlim(0, 280)
        ax.set_xlabel("Effective communication range (m)")
    for i, (_, r) in enumerate(eff.iterrows()):
        xerr = None
        if np.isfinite(r["CI_Low"]) and np.isfinite(r["CI_High"]):
            xerr = [[r["Range_m"] - r["CI_Low"]], [r["CI_High"] - r["Range_m"]]]
        axes[0].errorbar(r["Range_m"], y[i], xerr=xerr, fmt="o", capsize=4,
                         color=PALETTE[r["CallType"]], lw=1.4, ms=6)
        axes[0].text(r["Range_m"] + 4, y[i], f"{r['Range_m']:.0f}", va="center", fontsize=8.5)
    for i, ct in enumerate(CALL_ORDER):
        for j, tau in enumerate([0.50, 0.70]):
            row = avg_ranges[(avg_ranges["CallType"] == ct) & (avg_ranges["tau"] == tau)].iloc[0]
            pos = i * 2 + j
            lo = row["Range_min_all"]
            hi = row["Range_max_all"]
            axes[1].errorbar(row["Range_avg"], pos,
                             xerr=[[row["Range_avg"] - lo], [hi - row["Range_avg"]]],
                             fmt="o", capsize=5, color=PALETTE[ct], lw=1.4, ms=6)
            axes[1].text(row["Range_avg"] + 4, pos,
                         f"{row['Range_avg']:.0f} [{lo:.0f},{hi:.0f}]",
                         va="center", fontsize=8)
    labels = [f"{r['CallType']} | {r['Level']} (τ={r['tau']:.2f})" for _, r in eff.iterrows()]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[1].set_yticks(np.arange(len(CALL_ORDER) * 2))
    axes[1].set_yticklabels([f"{ct} | τ={tau:.2f}" for ct in CALL_ORDER
                             for tau in [0.50, 0.70]], fontsize=8.5)
    axes[1].invert_yaxis()
    plt.suptitle("Effective Range: Sampling Uncertainty vs Model Uncertainty",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig3_effective_range.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_threshold_sensitivity(ts, band, crossover):
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for ct in CALL_ORDER:
        s = ts[ts["CallType"] == ct].sort_values("tau")
        ax.plot(s["tau"], s["Range_m"], color=PALETTE[ct], lw=2, label=ct)
        b = band[band["CallType"] == ct].sort_values("tau")
        ax.fill_between(b["tau"], b["Range_min"], b["Range_max"],
                        color=PALETTE[ct], alpha=0.12)
    for tau in LEVELS.values():
        ax.axvline(tau, color="gray", ls="--", lw=0.9, alpha=0.8)
    if len(crossover) and np.isfinite(crossover.iloc[0]["tau_star"]):
        tstar = crossover.iloc[0]["tau_star"]
        rstar = crossover.iloc[0]["Range_star"]
        ax.axvline(tstar, color="black", ls="-.", lw=1.2)
        ax.scatter([tstar], [rstar], color="black", zorder=5, s=45)
        ax.annotate(f"crossover τ*={tstar:.2f}\n≈{rstar:.0f} m",
                    (tstar, rstar), textcoords="offset points", xytext=(8, 12),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Recognition threshold τ")
    ax.set_ylabel("Effective range (m)")
    ax.set_title("Threshold Sensitivity and tv–mod Crossover\n"
                 "Lines = main logistic; bands = min–max across models",
                 fontweight="bold")
    ax.legend(title="Call type")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig4_threshold_sensitivity.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_comparison(fit_metrics_all, weights_all):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4))
    sns.barplot(data=fit_metrics_all, x="CallType", y="CV_RMSE", hue="Model", ax=axes[0])
    axes[0].set_title("5-fold trial-level CV-RMSE (primary criterion)", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    w = weights_all.copy()
    if "Eligible" in w.columns:
        w = w[w["Eligible"]]
    w["Label"] = w["CallType"] + " | " + w["Model"]
    sns.barplot(data=w, x="Weight", y="Label", hue="CallType", dodge=False, ax=axes[1])
    axes[1].set_title("Model-averaging weights (CV-RMSE based)", fontweight="bold")
    axes[1].set_xlabel("Weight")
    axes[1].set_ylabel("")
    axes[1].legend_.remove() if axes[1].legend_ is not None else None
    plt.suptitle("Model Comparison and Model-Averaging Weights",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig5_model_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_bootstrap_bands(boot_curves, curves_main, summary, main_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, ct in zip(axes, CALL_ORDER):
        bc = boot_curves[ct]
        if len(bc):
            lo, hi = np.percentile(bc, [2.5, 97.5], axis=0)
            ax.fill_between(GRID, lo, hi, color=PALETTE[ct], alpha=0.18,
                            label="95% bootstrap band")
        ax.plot(GRID, curves_main[ct], color=PALETTE[ct], lw=2,
                label=f"Main fit ({main_names[ct]})")
        s = summary[summary["CallType"] == ct].sort_values("Distance")
        ax.scatter(s["Distance"], s["F_main_mean"], color=PALETTE[ct],
                   edgecolors="black", zorder=3, s=35, label="Observed mean")
        for tau in LEVELS.values():
            ax.axhline(tau, color="gray", ls="--", lw=0.8, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance (m)")
        ax.set_title(ct, fontweight="bold")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Composite fidelity F(d)")
    plt.suptitle("Trial Bootstrap with Fusion-Weight Uncertainty",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig6_bootstrap_bands.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_robustness(robust):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    x = np.arange(len(robust))
    w = 0.36
    for shift, col, label in [(-w / 2, "d50", "Potential (τ=0.50)"),
                              (w / 2, "d70", "Reliable (τ=0.70)")]:
        axes[0].bar(x + shift, robust[f"{col}_model_avg"], width=w,
                    color="#8DA0CB" if col == "d50" else "#FC8D62", label=label)
        axes[0].errorbar(x + shift, robust[f"{col}_model_avg"],
                         yerr=[robust[f"{col}_model_avg"] - robust[f"{col}_min_all"],
                               robust[f"{col}_max_all"] - robust[f"{col}_model_avg"]],
                         fmt="none", ecolor="black", capsize=3, lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(robust["CallType"])
    axes[0].set_ylabel("Effective range (m)")
    axes[0].set_title("Model-averaged range ± model spread", fontweight="bold")
    axes[0].legend(fontsize=8)
    for _, r in robust.iterrows():
        axes[1].scatter(r["d50_model_avg"], r["d70_model_avg"], s=140,
                        color=PALETTE[r["CallType"]], edgecolors="black", zorder=3)
        axes[1].annotate(r["CallType"], (r["d50_model_avg"], r["d70_model_avg"]),
                         textcoords="offset points", xytext=(7, 7), fontsize=10,
                         fontweight="bold")
    axes[1].set_xlabel("Model-averaged d50 (m)")
    axes[1].set_ylabel("Model-averaged d70 (m)")
    axes[1].set_title("Robustness plane", fontweight="bold")
    plt.suptitle("Long-Distance Robustness: Model-Averaged Ranges",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig7_robustness_ranking.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_fusion_sensitivity(fs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, tau in zip(axes, [0.50, 0.70]):
        s = fs[fs["tau"] == tau]
        sns.barplot(data=s, x="CallType", y="Range_m", hue="Fusion", ax=ax)
        ax.set_title(f"τ = {tau:.2f}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Effective range (m)")
    plt.suptitle("Fusion-Method Sensitivity of Effective Range",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig8_fusion_sensitivity.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_missingness(miss):
    piv = miss.pivot_table(index="CallType", columns="Distance",
                           values="Pct_missing_any")
    piv = piv.reindex(CALL_ORDER)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    sns.heatmap(piv, annot=True, fmt=".0f", cmap="Reds", vmin=0, vmax=40,
                cbar_kws={"label": "% trials missing"}, ax=ax)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("")
    ax.set_title("Informative Missingness by Call Type and Distance\n"
                 "(same trials are missing for all three metrics)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig9_missingness.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_missingness_bounds(bounds):
    if not len(bounds):
        return
    piv = bounds.pivot_table(index=["CallType", "Level", "tau"],
                             columns="Scenario", values="Range_m").reset_index()
    labels = [f"{r['CallType']} | {r['Level']} (τ={r['tau']:.2f})" for _, r in piv.iterrows()]
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for i, r in piv.iterrows():
        lo = r.get("worst_case", np.nan)
        hi = r.get("best_case", np.nan)
        mid = r.get("complete_case", np.nan)
        xerr = None
        if np.isfinite(lo) and np.isfinite(hi):
            xerr = [[mid - lo], [hi - mid]]
        ax.errorbar(mid, y[i], xerr=xerr, fmt="o", capsize=4, color="#4C72B0", lw=1.3)
        ax.text(mid + 3, y[i], f"{mid:.0f} [{lo:.0f},{hi:.0f}]", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 280)
    ax.axvline(4, color="gray", ls=":", lw=1)
    ax.axvline(256, color="gray", ls=":", lw=1)
    ax.set_xlabel("Effective range (m)")
    ax.set_title("Partial-Identification Bounds under Informative Missingness\n"
                 "complete case vs worst case (missing F=0) vs best case (missing F=1)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig10_missingness_bounds.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_recognition_probability(prob_curves, prob_ranges):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, ct in zip(axes, CALL_ORDER):
        for tau, ls, col in [(0.50, "-", "#4C72B0"), (0.70, "--", "#D62728")]:
            s = prob_curves[(prob_curves["CallType"] == ct) &
                            (prob_curves["tau"] == tau)].sort_values("Distance")
            if len(s) == 0:
                continue
            ax.plot(s["Distance"], s["P_recognize"], ls, lw=2, color=col,
                    label=f"Model τ={tau:.2f}")
            if "P_empirical" in s.columns:
                yerr = np.vstack([s["P_empirical"] - s["CI_Low"],
                                  s["CI_High"] - s["P_empirical"]])
                ax.errorbar(s["Distance"], s["P_empirical"], yerr=yerr,
                            fmt="o", mfc="white", mec=col, ecolor=col,
                            capsize=3, ms=5, lw=1.2, label=f"Empirical τ={tau:.2f}")
            br = prob_ranges[(prob_ranges["CallType"] == ct) &
                             (prob_ranges["tau"] == tau) &
                             (prob_ranges["p_level"] == 0.50)]
            if len(br) and np.isfinite(br.iloc[0]["Bracket_Low"]):
                ax.axvspan(br.iloc[0]["Bracket_Low"], br.iloc[0]["Bracket_High"],
                           color=col, alpha=0.06)
        for p in [0.50, 0.75, 0.90]:
            ax.axhline(p, color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance (m)")
        ax.set_title(ct, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
    axes[0].set_ylabel("P(fidelity ≥ τ) across trials")
    plt.suptitle("Empirical Proportion vs Monotone-Logistic Model\n"
                 "Shaded bands = empirical bracket at p=0.50 (interval censoring)",
                 fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig11_recognition_probability.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_metric_specific(ms):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, tau in zip(axes, [0.50, 0.70]):
        s = ms[ms["tau"] == tau]
        sns.barplot(data=s, x="CallType", y="Range_m", hue="Metric", ax=ax)
        ax.set_title(f"τ = {tau:.2f}", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Effective range (m)")
    plt.suptitle("Metric-Specific Effective Ranges: Which Channel Limits Communication?",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig12_metric_specific_ranges.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_playback_calibration(calib):
    if not len(calib):
        return
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    est = calib.set_index("Quantity")["Estimate"]
    ax.bar(["Response 0.5", "Response 0.7"],
           [est.get("F at response probability 0.5", np.nan),
            est.get("F at response probability 0.7", np.nan)],
           color=["#8DA0CB", "#FC8D62"])
    ax.axhline(0.50, color="gray", ls="--", lw=1)
    ax.axhline(0.70, color="gray", ls=":", lw=1)
    ax.set_ylabel("Model-derived fidelity F at playback distance")
    ax.set_title("Playback-Calibrated Fidelity Thresholds\n"
                 "Adjusted for Context and Npenguins; descriptive",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig13_playback_calibration.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_playback_descriptive(pb):
    if not len(pb):
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for grp in ["pooled"] + CALL_ORDER:
        s = pb[pb["Group"] == grp].sort_values("F_mid")
        if len(s) == 0:
            continue
        ax.plot(s["F_mid"], s["Response_rate"], marker="o", lw=2,
                color="#666666" if grp == "pooled" else PALETTE[grp], label=grp)
        for _, r in s.iterrows():
            ax.annotate(str(int(r["N"])), (r["F_mid"], r["Response_rate"]),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=7, ha="center")
    ax.set_xlabel("Model-derived fidelity F at playback distance")
    ax.set_ylabel("Proportion responding")
    ax.set_ylim(0, 1.05)
    ax.set_title("Descriptive Playback Check (not a calibrated threshold)\n"
                 "Numbers above points = N; context/group size confounded",
                 fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig13_playback_descriptive.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_monotonicity(long, mono):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), sharey=True)
    for ax, ct in zip(axes, CALL_ORDER):
        sub = long[long["CallType"] == ct]
        means, sds = [], []
        for d in DISTANCES:
            v = sub[sub["Distance"] == d]["F_main"].dropna().values.astype(float)
            means.append(float(v.mean()) if len(v) else np.nan)
            sds.append(float(v.std(ddof=1)) if len(v) > 1 else np.nan)
        means = np.asarray(means, float)
        sds = np.asarray(sds, float)
        ax.errorbar(DISTANCES, means, yerr=sds, fmt="o", capsize=3,
                    color=PALETTE[ct], label="Raw mean ± SD")
        try:
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                np.asarray(DISTANCES, float), means)
            ax.plot(DISTANCES, iso.predict(np.asarray(DISTANCES, float)),
                    color="black", lw=2, drawstyle="steps-post",
                    label="Isotonic envelope")
        except Exception:
            pass
        row = mono[mono["CallType"] == ct]
        if len(row):
            ax.set_title(f"{ct}\nviolations={int(row.iloc[0]['N_violations'])}, "
                         f"max jump={row.iloc[0]['Max_upward_jump']:.3f}",
                         fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(DISTANCES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Distance (m)")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Composite fidelity F(d)")
    plt.suptitle("Monotonicity Diagnostic: Raw Distance Trend vs Isotonic Envelope",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig14_monotonicity.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def plot_tornado(uncert):
    if not len(uncert):
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=False)
    for ax, ct in zip(axes, CALL_ORDER):
        s_all = uncert[uncert["CallType"] == ct]
        s = s_all[np.isfinite(s_all["Width"])].sort_values("Width")
        y = np.arange(len(s))
        center = s["Center"].iloc[0] if len(s) else 0
        ax.hlines(y, s["Low"], s["High"], color="#4C72B0", lw=8, alpha=0.55)
        ax.scatter(s["Low"], y, color="#4C72B0", s=20, zorder=3)
        ax.scatter(s["High"], y, color="#4C72B0", s=20, zorder=3)
        ax.axvline(center, color="black", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(s["Source"], fontsize=8.5)
        ax.set_xlabel("d70 (m)")
        n_unid = int((~s_all["Identified"]).sum())
        title = ct + (f"\n{n_unid} source(s) not identified" if n_unid else "")
        ax.set_title(title, fontweight="bold")
    plt.suptitle("Uncertainty Decomposition at τ=0.70: Which Source Dominates?",
                 fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "P2_fig15_tornado.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 9. Key results
# ==============================================================================
def write_key_results(long, weights, pca_var, fit_metrics_all, weights_all,
                      eff, avg_ranges, bounds, prob_ranges, cross, ms, fs,
                      metric_ranges, weighted, gamknot, calib, robust, pair,
                      boot_summary, main_names):
    lines = [
        "Problem 2 v2 -- key results",
        f"Environment: Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, sklearn {sklearn.__version__}, seaborn {sns.__version__}",
        f"Mode: {'FAST' if FAST else 'FULL'} (N_BOOT={N_BOOT})",
        "",
        "Data and missingness:",
    ]
    for ct in CALL_ORDER:
        s = long[long["CallType"] == ct]
        lines.append(f"  {ct}: {s['TrialID'].nunique()} trials; "
                     f"F_main valid={s['F_main'].notna().sum()}/{len(s)}")
    lines += ["  Missingness is trial-level and distance-dependent; see P2_missingness_audit.csv."]

    lines += ["", "Fusion (PCA, main):"]
    for i, lab in enumerate(["Spectrogram", "Amplitude spectrum", "Envelope"]):
        lines.append(f"  {lab}: weight={weights['pca'][i]:.3f}")
    lines.append(f"  PC1 explains {pca_var*100:.1f}% of standardised metric variance "
                 f"(near one-factor structure).")

    lines += ["", "Main model per call type (logistic preferred; fallback in parentheses):"]
    for ct in CALL_ORDER:
        s = fit_metrics_all[fit_metrics_all["CallType"] == ct]
        main = main_names[ct]
        row = s[s["Model"] == main].iloc[0]
        lines.append(f"  {ct}: {main}, CV-RMSE={row['CV_RMSE']:.3f}, "
                     f"R2={row['R2']:.3f}, boundary={bool(row['Boundary'])}")

    lines += ["", "Model-averaging weights (eligible models only):"]
    for ct in CALL_ORDER:
        w = weights_all[weights_all["CallType"] == ct]
        txt = ", ".join(f"{r['Model']}={r['Weight']:.2f}" for _, r in w.iterrows())
        lines.append(f"  {ct}: {txt}")

    lines += ["", "Main logistic effective range (bootstrap 95% CI):"]
    for _, r in eff.iterrows():
        ci = (f"[{r['CI_Low']:.1f}, {r['CI_High']:.1f}]"
              if np.isfinite(r["CI_Low"]) and np.isfinite(r["CI_High"]) else "[n/a]")
        lines.append(f"  {r['CallType']} | {r['Level']} (τ={r['tau']:.2f}): "
                     f"{r['Range_m']:.1f} m {ci} ({r['Status']})")

    lines += ["", "Model-averaged range and model spread:"]
    for ct in CALL_ORDER:
        for tau in [0.50, 0.70]:
            r = avg_ranges[(avg_ranges["CallType"] == ct) &
                           (avg_ranges["tau"] == tau)].iloc[0]
            lines.append(f"  {ct} τ={tau:.2f}: avg={r['Range_avg']:.1f} m; "
                         f"eligible [{r['Range_min_eligible']:.1f}, "
                         f"{r['Range_max_eligible']:.1f}]; "
                         f"all models [{r['Range_min_all']:.1f}, {r['Range_max_all']:.1f}]")
    lines.append("  Use the eligible interval for model-choice uncertainty; the all-model "
                 "interval includes boundary/poor-fit models and is an extreme scenario.")

    lines += ["", "Partial-identification bounds under informative missingness:"]
    if len(bounds):
        for _, r in bounds[bounds["Level"] == "Reliable"].iterrows():
            lines.append(f"  {r['CallType']} | {r['Scenario']} τ={r['tau']:.2f}: "
                         f"{r['Range_m']:.1f} m (R2={r['R2']:.3f})")
    else:
        lines.append("  skipped (fit failed).")

    lines += ["", "Probabilistic range R_tau(p):"]
    for _, r in prob_ranges.iterrows():
        lines.append(f"  {r['CallType']} | τ={r['tau']:.2f} | p={r['p_level']:.2f}: "
                     f"{r['Range_m']:.1f} m ({r['Status']})")

    lines += ["", "Threshold-dependent dominance (tv vs mod):"]
    for _, r in cross.iterrows():
        lines.append(f"  τ*={r['tau_star']:.3f} [{r['tau_star_CI_Low']:.3f}, "
                     f"{r['tau_star_CI_High']:.3f}], R*={r['Range_star']:.1f} m "
                     f"[{r['Range_star_CI_Low']:.1f}, {r['Range_star_CI_High']:.1f}]")
    lines.append("  Interpretation: tv dominates at low thresholds (potential recognition); "
                 "mod dominates at high thresholds (reliable recognition).")

    lines += ["", "Metric-specific ranges (bottleneck channel):"]
    for ct in CALL_ORDER:
        s = metric_ranges[(metric_ranges["CallType"] == ct) & (metric_ranges["tau"] == 0.70)]
        if len(s):
            lines.append(f"  {ct} τ=0.70: " + ", ".join(
                f"{r['Metric']}={r['Range_m']:.0f} m" for _, r in s.iterrows()))

    lines += ["", "Sensitivity checks:"]
    lines.append("  Weighted vs unweighted logistic (τ=0.70):")
    for ct in CALL_ORDER:
        s = weighted[(weighted["CallType"] == ct) & (weighted["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"{r['Fit']}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  GAM knots (τ=0.70):")
    for ct in CALL_ORDER:
        s = gamknot[(gamknot["CallType"] == ct) & (gamknot["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"k={int(r['n_knots'])}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  Fusion methods (τ=0.70):")
    for ct in CALL_ORDER:
        s = fs[(fs["CallType"] == ct) & (fs["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"{r['Fusion']}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  The tv high-threshold range is fusion-dependent; entropy/geometric are "
                 "more conservative because they weight the weak envelope channel more heavily.")

    lines += ["", "Playback calibration (adjusted logistic):"]
    if len(calib):
        for _, r in calib.iterrows():
            lines.append(f"  {r['Quantity']}: F={r['Estimate']:.3f} "
                         f"[{r['CI_Low']:.3f}, {r['CI_High']:.3f}] (N={int(r['N'])})")
        lines.append("  Compare with acoustic thresholds 0.50/0.60/0.70.")
    else:
        lines.append("  skipped (Data02 not available or fit failed).")

    lines += ["", "Robustness ranking by model-averaged d70:"]
    for _, r in robust.iterrows():
        lines.append(f"  {int(r['Rank_d70'])}. {r['CallType']}: "
                     f"d50={r['d50_model_avg']:.1f} [{r['d50_min_all']:.1f},"
                     f"{r['d50_max_all']:.1f}], "
                     f"d70={r['d70_model_avg']:.1f} [{r['d70_min_all']:.1f},"
                     f"{r['d70_max_all']:.1f}], AUC_log={r['AUC_log']:.3f}")

    lines += ["", "Pairwise differences (main logistic bootstrap, conditional on model):"]
    for _, r in pair.iterrows():
        lines.append(f"  τ={r['tau']:.2f} | {r['Comparison']}: {r['Diff_m']:.1f} m "
                     f"[{r['CI_Low']:.1f}, {r['CI_High']:.1f}], "
                     f"P(diff>0)={r['P_diff_gt_0']:.3f}")

    lines += [
        "",
        "Honest conclusions:",
        "  1. At the potential threshold (τ=0.50) the model-averaged ordering is "
        "reported above; flat is clearly lowest, tv/mod overlap in model uncertainty.",
        "  2. At the reliable threshold (τ=0.70) the tv/mod ordering is model-dependent; "
        "do not present a single winner without the model-spread interval.",
        "  3. The tv–mod range curves cross at τ* ≈ 0.59: tv is better for potential "
        "recognition, mod for reliable recognition.",
        "  4. Informative missingness can bias complete-case ranges upward; the "
        "worst/best-case bounds are the defensible uncertainty statement.",
        "  5. The mean-curve bootstrap CI is not an individual recognition interval; "
        "use the probabilistic range R_tau(p) for ecological claims.",
    ]
    with open(os.path.join(OUT_DIR, "P2_key_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ==============================================================================
# 10. Main
# ==============================================================================
def write_key_results_v2(long, weights, pca_var, fit_metrics_all, weights_all,
                         eff, avg_ranges, bounds, prob_ranges, cross, fs,
                         metric_ranges, weighted, gamknot, pbdesc, robust, pair,
                         boot_summary, main_names, mono, uncert):
    lines = [
        "Problem 2 v2.1 -- key results",
        f"Environment: Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, sklearn {sklearn.__version__}, seaborn {sns.__version__}",
        f"Mode: {'FAST' if FAST else 'FULL'} (N_BOOT={N_BOOT})",
        "",
        "Data, missingness and monotonicity:",
    ]
    for ct in CALL_ORDER:
        s = long[long["CallType"] == ct]
        m = mono[mono["CallType"] == ct].iloc[0]
        lines.append(f"  {ct}: {s['TrialID'].nunique()} trials; "
                     f"F_main valid={s['F_main'].notna().sum()}/{len(s)}; "
                     f"raw monotonicity violations={int(m['N_violations'])}, "
                     f"max upward jump={m['Max_upward_jump']:.3f}, "
                     f"RMSE(raw vs isotonic)={m['RMSE_raw_vs_isotonic']:.4f}")
    lines.append("  The raw distance trend is NOT monotone; all range estimates below "
                 "use the isotonic monotone envelope as the robust main model.")

    lines += ["", "Fusion (PCA, main):"]
    for i, lab in enumerate(["Spectrogram", "Amplitude spectrum", "Envelope"]):
        lines.append(f"  {lab}: weight={weights['pca'][i]:.3f}")
    lines.append(f"  PC1 explains {pca_var*100:.1f}% of standardised metric variance.")

    lines += ["", "Main model per call type (isotonic monotone envelope preferred):"]
    for ct in CALL_ORDER:
        s = fit_metrics_all[fit_metrics_all["CallType"] == ct]
        main = main_names[ct]
        row = s[s["Model"] == main].iloc[0]
        lines.append(f"  {ct}: {main}, CV-RMSE={row['CV_RMSE']:.3f}, "
                     f"R2={row['R2']:.3f}, boundary={bool(row['Boundary'])}")

    lines += ["", "Model-averaging weights (eligible models only; ineligible = 0):"]
    for ct in CALL_ORDER:
        w = weights_all[weights_all["CallType"] == ct]
        if len(w) == 0:
            continue
        elig = w[w["Eligible"]]
        if len(elig) == 0:
            lines.append(f"  {ct}: no model passed the screen; see model spread.")
        elif len(elig) == 1:
            lines.append(f"  {ct}: only {elig.iloc[0]['Model']} eligible; "
                         f"weighted average = that model; model spread is the robust statement.")
        else:
            txt = ", ".join(f"{r['Model']}={r['Weight']:.2f}" for _, r in elig.iterrows())
            lines.append(f"  {ct}: {txt}")

    lines += ["", "Main isotonic effective range (bootstrap 95% CI):"]
    for _, r in eff.iterrows():
        ci = (f"[{r['CI_Low']:.1f}, {r['CI_High']:.1f}]"
              if np.isfinite(r["CI_Low"]) and np.isfinite(r["CI_High"]) else "[n/a]")
        lines.append(f"  {r['CallType']} | {r['Level']} (τ={r['tau']:.2f}): "
                     f"{r['Range_m']:.1f} m {ci} ({r['Status']})")

    lines += ["", "Model-averaged range and model spread:"]
    for ct in CALL_ORDER:
        for tau in [0.50, 0.70]:
            r = avg_ranges[(avg_ranges["CallType"] == ct) &
                           (avg_ranges["tau"] == tau)].iloc[0]
            lines.append(f"  {ct} τ={tau:.2f}: avg={r['Range_avg']:.1f} m; "
                         f"eligible [{r['Range_min_eligible']:.1f}, "
                         f"{r['Range_max_eligible']:.1f}]; "
                         f"all models [{r['Range_min_all']:.1f}, {r['Range_max_all']:.1f}]")
    lines.append("  Use the eligible interval for model-choice uncertainty; the all-model "
                 "interval includes boundary/poor-fit models and is an extreme scenario.")

    lines += ["", "Partial-identification bounds under informative missingness "
                  "(isotonic; Reliable = R2>=0.60):"]
    if len(bounds):
        for _, r in bounds[bounds["Level"] == "Reliable"].iterrows():
            tag = "" if r["Reliable"] else "  [UNRELIABLE FIT]"
            lines.append(f"  {r['CallType']} | {r['Scenario']} τ={r['tau']:.2f}: "
                         f"{r['Range_m']:.1f} m (R2={r['R2']:.3f}){tag}")
        lines.append("  Do not treat an UNRELIABLE FIT as a credible bound; it is an "
                     "extreme-scenario extrapolation.")
        for ct in CALL_ORDER:
            s = bounds[(bounds["CallType"] == ct) & (bounds["Level"] == "Reliable")]
            w = s[s["Scenario"] == "worst_case"]
            b = s[s["Scenario"] == "best_case"]
            ok = (len(w) and len(b) and bool(w.iloc[0]["Reliable"])
                  and bool(b.iloc[0]["Reliable"]))
            if not ok:
                lines.append(f"  {ct}: missingness bounds NOT IDENTIFIED under MNAR "
                             "(no reliable worst/best fit); the complete-case value is "
                             "optimistic and should be labelled as such.")
    else:
        lines.append("  skipped (fit failed).")

    lines += ["", "Probabilistic range R_tau(p), common trials + monotone logistic:"]
    if len(prob_ranges):
        for _, r in prob_ranges.iterrows():
            ci = (f"[{r['CI_Low']:.1f}, {r['CI_High']:.1f}]"
                  if np.isfinite(r["CI_Low"]) and np.isfinite(r["CI_High"]) else "[n/a]")
            bracket = f"[{r['Bracket_Low']:.0f}, {r['Bracket_High']:.0f}]"
            flag = ""
            if not bool(r["Model_Inside_Bracket"]):
                flag += " [MODEL POINT OUTSIDE BRACKET]"
            if bool(r["Interval_Censored"]):
                flag += " [interval-censored]"
            lines.append(f"  {r['CallType']} | τ={r['tau']:.2f} | p={r['p_level']:.2f}: "
                         f"{r['Range_m']:.1f} m {ci}; bracket {bracket} "
                         f"({r['Bracket_Status']}; N_cross={int(r['N_Empirical_Crossings'])})"
                         f"{flag} (N_common={int(r['N_common'])}; "
                         f"recommended={r['Recommended']})")
        out_rows = prob_ranges[~prob_ranges["Model_Inside_Bracket"]]
        if len(out_rows):
            lines.append(f"  {len(out_rows)} row(s) have MODEL POINT OUTSIDE BRACKET; "
                         "quote the bracket only (the logistic is a smooth interpolation, "
                         "not an identified value).")
        lines.append("  The bracket is the nonparametric identification interval from the "
                     "empirical crossings. A zero-width CI means the model sees the same "
                     "all-above/all-below pattern in every bootstrap; it is NOT a precise "
                     "range. Use the bracket as the honest uncertainty statement.")
        lines.append("  The probability is a smooth monotone logistic on log-distance; "
                     "this reduces the isotonic step artifact. Where the data are "
                     "all-above/all-below between adjacent distances (e.g. tv at τ=0.50), "
                     "the model CI can still be zero-width; use the bracket, not the CI.")
        lines.append("  p=0.90 is an individual-level guarantee: if it is very short or 0, "
                     "fewer than 90% of trials reach tau even near the source. Report "
                     "p=0.50/0.75 as the main ecological range and p=0.90 as a strict bound.")
    else:
        lines.append("  skipped (common-trial set too small).")

    lines += ["", "Threshold-dependent dominance (isotonic main model):"]
    if len(cross):
        r = cross.iloc[0]
        if r["Status"] == "crossed" and np.isfinite(r["tau_star"]):
            lines.append(f"  τ*={r['tau_star']:.3f} [{r['tau_star_CI_Low']:.3f}, "
                         f"{r['tau_star_CI_High']:.3f}], R*={r['Range_star']:.1f} m "
                         f"[{r['Range_star_CI_Low']:.1f}, {r['Range_star_CI_High']:.1f}]")
            lines.append("  A crossing means the tv/mod ranking is threshold-dependent; "
                         "report both regimes rather than a single winner.")
            p70 = pair[(pair["Comparison"] == "tv - mod") & (pair["tau"] == 0.70)]
            if len(p70) and p70.iloc[0]["CI_Low"] <= 0 <= p70.iloc[0]["CI_High"]:
                lines.append("  At τ=0.70 the tv-mod difference CI includes 0; treat tv and "
                             "mod as indistinguishable at the reliable threshold even "
                             "though the fitted curves cross at τ*.")
        else:
            lines.append(f"  No crossing in the 0.40-0.80 range (status={r['Status']}); "
                         "the tv/mod ordering is not threshold-dependent in this interval.")
    else:
        lines.append("  skipped.")

    lines += ["", "Metric-specific ranges (τ=0.70):"]
    for ct in CALL_ORDER:
        s = metric_ranges[(metric_ranges["CallType"] == ct) &
                          (metric_ranges["tau"] == 0.70)]
        if len(s):
            parts = []
            for _, r in s.iterrows():
                if r["Status"] == "below_4m":
                    parts.append(f"{r['Metric']}=not reached at 4 m")
                else:
                    parts.append(f"{r['Metric']}={r['Range_m']:.0f} m")
            lines.append(f"  {ct}: " + ", ".join(parts))
    lines.append("  The envelope channel cannot reach 0.70 at 4 m. The composite 0.70 "
                 "range is therefore a compensatory average and can be SHORTER than the "
                 "spectrogram/amplitude ranges; this is a short-board effect, not a bug.")

    lines += ["", "Sensitivity checks:"]
    lines.append("  Weighted vs unweighted isotonic (τ=0.70):")
    for ct in CALL_ORDER:
        s = weighted[(weighted["CallType"] == ct) & (weighted["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"{r['Fit']}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  GAM knots (τ=0.70):")
    for ct in CALL_ORDER:
        s = gamknot[(gamknot["CallType"] == ct) & (gamknot["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"k={int(r['n_knots'])}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  Fusion methods (τ=0.70):")
    for ct in CALL_ORDER:
        s = fs[(fs["CallType"] == ct) & (fs["tau"] == 0.70)]
        if len(s):
            lines.append(f"    {ct}: " + ", ".join(
                f"{r['Fusion']}={r['Range_m']:.0f} m" for _, r in s.iterrows()))
    lines.append("  The tv high-threshold range is fusion-dependent; entropy/geometric are "
                 "more conservative because they weight the weak envelope channel more heavily.")

    lines += ["", "Playback check (descriptive only; no calibrated threshold):"]
    if len(pbdesc):
        for _, r in pbdesc[pbdesc["Group"] == "pooled"].iterrows():
            if int(r["N"]) >= 10:
                lines.append(f"  F bin {r['F_bin']}: response={r['Response_rate']:.2f} "
                             f"(N={int(r['N'])})")
        lines.append("  Context/group size are confounded; the earlier adjusted-logistic "
                     "threshold calibration was not identified, so Problem 3 owns the "
                     "formal behavioural model.")
    else:
        lines.append("  skipped (Data02 not available).")

    lines += ["", "Robustness at τ=0.70 (model-averaged d70 and AUC):"]
    for _, r in robust.iterrows():
        ar = avg_ranges[(avg_ranges["CallType"] == r["CallType"]) &
                        (avg_ranges["tau"] == 0.70)].iloc[0]
        lines.append(f"  {r['CallType']}: d70={r['d70_model_avg']:.1f} m; "
                     f"eligible [{ar['Range_min_eligible']:.1f}, "
                     f"{ar['Range_max_eligible']:.1f}]; "
                     f"all [{r['d70_min_all']:.1f}, {r['d70_max_all']:.1f}], "
                     f"AUC_log={r['AUC_log']:.3f}")
    lines.append("  The d70 and AUC orderings can disagree; do not present a single "
                 "rank when the model intervals overlap.")

    lines += ["", "Pairwise differences (main isotonic bootstrap; conditional on model):"]
    for _, r in pair.iterrows():
        lines.append(f"  τ={r['tau']:.2f} | {r['Comparison']}: {r['Diff_m']:.1f} m "
                     f"[{r['CI_Low']:.1f}, {r['CI_High']:.1f}], "
                     f"P(diff>0)={r['P_diff_gt_0']:.3f}")

    lines += ["", "Uncertainty decomposition at τ=0.70 (sorted by width):"]
    if len(uncert):
        for ct in CALL_ORDER:
            s = uncert[uncert["CallType"] == ct].sort_values(
                "Width", ascending=False, na_position="last")
            if len(s) == 0:
                continue
            lines.append(f"  {ct}:")
            for _, r in s.iterrows():
                if not bool(r["Identified"]):
                    lines.append(f"    {r['Source']}: NOT IDENTIFIED")
                else:
                    lines.append(f"    {r['Source']}: [{r['Low']:.1f}, {r['High']:.1f}] "
                                 f"(width {r['Width']:.1f} m)")
    lines.append("  Sampling CIs are narrow; model choice / threshold / fusion / missingness "
                 "dominate. This is why a single point estimate with a ±3 m CI is misleading.")
    tv_thr = uncert[(uncert["CallType"] == "tv") &
                    (uncert["Source"] == "Threshold ±0.05")]
    if len(tv_thr) and np.isfinite(tv_thr.iloc[0]["Width"]):
        lines.append(f"  Threshold sensitivity is large: tv ±0.05 spans "
                     f"{tv_thr.iloc[0]['Width']:.1f} m; report the range-threshold curve, "
                     "not a single d70.")

    def _avg(ct, tau):
        return avg_ranges[(avg_ranges["CallType"] == ct) &
                          (avg_ranges["tau"] == tau)].iloc[0]

    tv50, md50 = _avg("tv", 0.50), _avg("mod", 0.50)
    tv70, md70 = _avg("tv", 0.70), _avg("mod", 0.70)
    overlap50 = not (tv50["Range_min_eligible"] > md50["Range_max_eligible"]
                     or md50["Range_min_eligible"] > tv50["Range_max_eligible"])
    overlap70 = not (tv70["Range_min_eligible"] > md70["Range_max_eligible"]
                     or md70["Range_min_eligible"] > tv70["Range_max_eligible"])

    lines += [
        "",
        "Honest conclusions:",
        "  1. The raw fidelity curves are non-monotonic; the isotonic envelope is the "
        "defensible monotone representation. The violations are reported above.",
        f"  2. At τ=0.50: tv {tv50['Range_avg']:.1f} m "
        f"(eligible [{tv50['Range_min_eligible']:.1f}, {tv50['Range_max_eligible']:.1f}]) "
        f"vs mod {md50['Range_avg']:.1f} m "
        f"(eligible [{md50['Range_min_eligible']:.1f}, {md50['Range_max_eligible']:.1f}]); "
        f"{'intervals overlap' if overlap50 else 'tv is above mod'}. "
        f"At τ=0.70: tv {tv70['Range_avg']:.1f} m "
        f"(eligible [{tv70['Range_min_eligible']:.1f}, {tv70['Range_max_eligible']:.1f}]) "
        f"vs mod {md70['Range_avg']:.1f} m "
        f"(eligible [{md70['Range_min_eligible']:.1f}, {md70['Range_max_eligible']:.1f}]); "
        f"{'intervals overlap' if overlap70 else 'tv is above mod'}. "
        "Flat is clearly lowest at both thresholds; do not rank tv/mod at τ=0.70.",
        "  3. The fitted isotonic curves cross at τ* (reported above). At τ=0.70 the "
        "tv-mod difference CI includes 0 unless stated otherwise, so treat tv and mod "
        "as indistinguishable at the reliable threshold.",
        "  4. Informative missingness can bias complete-case ranges upward; only bounds "
        "with Reliable=TRUE should be quoted as credible. For a call type with no reliable "
        "worst/best fit, the range is NOT IDENTIFIED under MNAR and the complete-case "
        "value is optimistic (tv is the clear example).",
        "  5. The mean-curve bootstrap CI is not an individual recognition interval; the "
        "common-trial bracket (with the monotone logistic as a smooth interpolation) is "
        "the appropriate probabilistic statement.",
        "  6. The envelope channel cannot reach 0.70 at 4 m; the composite 0.70 range is "
        "a compensatory average, not a channel-wise bottleneck.",
        "  7. p=0.90 is an individual-level guarantee. A short or zero p=0.90 range means "
        "fewer than 90% of trials reach tau even at 4 m; report p=0.50/0.75 as the main "
        "ecological range and p=0.90 as a strict bound.",
    ]
    with open(os.path.join(OUT_DIR, "P2_key_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    archive_deprecated_outputs()
    check_data()
    write_data_manifest()

    print("\n" + "=" * 78)
    print("STEP 2: LOAD, AUDIT MISSINGNESS, FUSE, SUMMARISE")
    print("=" * 78)
    long, wide_by_ct, audit = load_propagation_data()
    audit.to_csv(os.path.join(OUT_DIR, "P2_data_audit.csv"),
                 index=False, encoding="utf-8-sig")
    miss = missingness_audit(long)
    miss.to_csv(os.path.join(OUT_DIR, "P2_missingness_audit.csv"),
                index=False, encoding="utf-8-sig")
    print(miss.to_string(index=False))

    weights, pca_var, weight_table, corr_table = compute_fusion_weights(long)
    long = apply_fusion(long, weights)
    long.to_csv(os.path.join(OUT_DIR, "P2_analysis_dataset.csv"),
                index=False, encoding="utf-8-sig")
    weight_table.to_csv(os.path.join(OUT_DIR, "P2_fusion_weights.csv"),
                        index=False, encoding="utf-8-sig")
    corr_table.to_csv(os.path.join(OUT_DIR, "P2_metric_correlation.csv"),
                      index=False, encoding="utf-8-sig")
    write_data_dictionary()
    summary = summarize_fidelity(long)
    summary.to_csv(os.path.join(OUT_DIR, "P2_fidelity_summary.csv"),
                   index=False, encoding="utf-8-sig")
    plot_metric_decay(summary)
    plot_missingness(miss)

    print("\n" + "=" * 78)
    print("STEP 3: DEGRADATION MODELS AND MODEL AVERAGING")
    print("=" * 78)
    models_by_ct, main_names, metrics_by_ct = {}, {}, {}
    curves_main, avg_curves, weights_by_ct, eligible_by_ct = {}, {}, {}, {}
    boot_curves, boot_params = {}, {}
    boot_rows = []
    fit_rows = []
    weights_all_rows = []

    for ct in CALL_ORDER:
        print(f"\n[{ct}]")
        df_ct = long[long["CallType"] == ct]
        models, d, y, trial = fit_models_for_ct(df_ct)
        if not models:
            raise RuntimeError(f"No model converged for {ct}.")
        main_name = pick_main(models)
        main_names[ct] = main_name
        models_by_ct[ct] = models
        mfit = model_fit_metrics(models, d, y, trial)
        mfit.insert(0, "CallType", ct)
        metrics_by_ct[ct] = mfit
        fit_rows.append(mfit)
        print(mfit[["Model", "CV_RMSE", "R2", "Boundary", "PoorFit"]]
              .round(3).to_string(index=False))

        avg_curve, w, elig = compute_model_average(models, mfit)
        avg_curves[ct] = avg_curve
        weights_by_ct[ct] = w
        eligible_by_ct[ct] = elig
        for name in models:
            weights_all_rows.append({"CallType": ct, "Model": name,
                                     "Weight": float(w.get(name, 0.0)),
                                     "Eligible": bool(name in elig)})

        curves_main[ct] = models[main_name]["predict"](GRID)
        bc, bp, bw, n_fail = bootstrap_main(df_ct, MODEL_FITTERS[main_name])
        boot_curves[ct] = bc
        boot_params[ct] = bp
        boot_rows.append({"CallType": ct, "MainModel": main_name,
                          "N_requested": N_BOOT, "N_valid": len(bc),
                          "N_failed": n_fail})
        print(f"  main={main_name}; bootstrap={len(bc)}/{N_BOOT} valid")

    fit_metrics_all = pd.concat(fit_rows, ignore_index=True)
    fit_metrics_all.to_csv(os.path.join(OUT_DIR, "P2_model_fit_metrics.csv"),
                           index=False, encoding="utf-8-sig")
    weights_all = pd.DataFrame(weights_all_rows)
    weights_all.to_csv(os.path.join(OUT_DIR, "P2_model_averaging.csv"),
                       index=False, encoding="utf-8-sig")
    boot_summary = pd.DataFrame(boot_rows)
    boot_summary.to_csv(os.path.join(OUT_DIR, "P2_bootstrap_summary.csv"),
                        index=False, encoding="utf-8-sig")

    # Parameter table: main model with bootstrap CI; physical comparator point estimates
    param_rows = []
    for ct in CALL_ORDER:
        for name in ["Logistic", "Physical log-exp", "Physical power-exp", "Weibull"]:
            m = models_by_ct[ct].get(name)
            if m is None:
                continue
            p = np.asarray(m["params"], float)
            for i, pn in enumerate(PARAM_NAMES[name]):
                est = float(p[i])
                lo = hi = np.nan
                if name == main_names[ct] and len(boot_params[ct]):
                    bp = boot_params[ct][:, i]
                    bp = bp[np.isfinite(bp)]
                    if len(bp) >= 10:
                        lo, hi = np.percentile(bp, [2.5, 97.5])
                param_rows.append({"CallType": ct, "Model": name, "Parameter": pn,
                                   "Estimate": est, "CI_Low": float(lo),
                                   "CI_High": float(hi),
                                   "Boundary": bool(m.get("boundary", False)),
                                   "Description": PARAM_DESC.get(pn, "")})
    pd.DataFrame(param_rows).to_csv(
        os.path.join(OUT_DIR, "P2_model_parameters.csv"),
        index=False, encoding="utf-8-sig")

    plot_model_fits(models_by_ct, summary, main_names)
    plot_bootstrap_bands(boot_curves, curves_main, summary, main_names)

    print("\n" + "=" * 78)
    print("STEP 4: EFFECTIVE RANGE, MISSINGNESS BOUNDS, PROBABILISTIC RANGE")
    print("=" * 78)
    eff = effective_range_table(curves_main, boot_curves)
    eff.to_csv(os.path.join(OUT_DIR, "P2_effective_range.csv"),
               index=False, encoding="utf-8-sig")
    print(eff[["CallType", "Level", "tau", "Range_m", "CI_Low", "CI_High", "Status"]]
          .round(2).to_string(index=False))

    avg_ranges = model_averaged_range_table(models_by_ct, avg_curves,
                                            weights_by_ct, eligible_by_ct)
    avg_ranges.to_csv(os.path.join(OUT_DIR, "P2_model_averaged_range.csv"),
                      index=False, encoding="utf-8-sig")

    band = model_band_table(models_by_ct)
    band.to_csv(os.path.join(OUT_DIR, "P2_model_band.csv"),
                index=False, encoding="utf-8-sig")

    bounds = missingness_bounds_table(long, main_names)
    print("\n[Missingness partial-identification bounds]")
    print(bounds.round(2).to_string(index=False))

    prob_ranges, prob_curves = probabilistic_range_table(long)
    prob_ranges.to_csv(os.path.join(OUT_DIR, "P2_probabilistic_range.csv"),
                       index=False, encoding="utf-8-sig")
    prob_curves.to_csv(os.path.join(OUT_DIR, "P2_recognition_probability.csv"),
                       index=False, encoding="utf-8-sig")
    print("\n[Probabilistic range R_tau(p)]")
    print(prob_ranges.round(2).to_string(index=False))

    cross = crossover_table(curves_main, boot_curves)
    cross.to_csv(os.path.join(OUT_DIR, "P2_crossover.csv"),
                 index=False, encoding="utf-8-sig")
    print("\n[Threshold crossover tv vs mod]")
    print(cross.round(3).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 5: METRIC-SPECIFIC AND SENSITIVITY ANALYSES")
    print("=" * 78)
    metric_ranges = metric_specific_range_table(long)
    metric_ranges.to_csv(os.path.join(OUT_DIR, "P2_metric_specific_ranges.csv"),
                         index=False, encoding="utf-8-sig")
    weighted = weighted_fit_table(long)
    weighted.to_csv(os.path.join(OUT_DIR, "P2_weighted_fit_sensitivity.csv"),
                    index=False, encoding="utf-8-sig")
    gamknot = gam_knot_table(long)
    gamknot.to_csv(os.path.join(OUT_DIR, "P2_gam_knot_sensitivity.csv"),
                   index=False, encoding="utf-8-sig")
    fs = fusion_sensitivity_table(long)
    fs.to_csv(os.path.join(OUT_DIR, "P2_fusion_sensitivity.csv"),
              index=False, encoding="utf-8-sig")
    for tab, label in [(metric_ranges, "metric-specific"),
                       (weighted, "weighted fit"),
                       (gamknot, "GAM knots"),
                       (fs, "fusion")]:
        print(f"\n[{label} sensitivity]")
        print(tab.round(2).to_string(index=False))

    print("\n" + "=" * 78)
    print("STEP 6: ROBUSTNESS, PAIRWISE DIFFERENCES, PLAYBACK CALIBRATION")
    print("=" * 78)
    robust = robustness_table(avg_ranges, band, avg_curves, eff)
    robust.to_csv(os.path.join(OUT_DIR, "P2_robustness_ranking.csv"),
                  index=False, encoding="utf-8-sig")
    print(robust.round(2).to_string(index=False))

    pair = pairwise_differences(boot_curves)
    pair.to_csv(os.path.join(OUT_DIR, "P2_pairwise_range_differences.csv"),
                index=False, encoding="utf-8-sig")
    print(pair.round(3).to_string(index=False))

    pbdesc = playback_descriptive(models_by_ct, main_names)
    if len(pbdesc):
        pbdesc.to_csv(os.path.join(OUT_DIR, "P2_playback_descriptive.csv"),
                      index=False, encoding="utf-8-sig")
        print(pbdesc.round(3).to_string(index=False))

    mono = monotonicity_diagnostic(long)
    mono.to_csv(os.path.join(OUT_DIR, "P2_monotonicity_diagnostic.csv"),
                index=False, encoding="utf-8-sig")
    uncert = uncertainty_decomposition(eff, boot_curves, band, bounds, fs,
                                      curves_main, avg_ranges)
    uncert.to_csv(os.path.join(OUT_DIR, "P2_uncertainty_decomposition.csv"),
                  index=False, encoding="utf-8-sig")

    # Figures for the new analyses
    plot_effective_ranges(eff, avg_ranges)
    plot_threshold_sensitivity(ts_table(curves_main, main_names), band, cross)
    plot_model_comparison(fit_metrics_all, weights_all)
    plot_robustness(robust)
    if len(fs):
        plot_fusion_sensitivity(fs)
    plot_missingness_bounds(bounds)
    plot_recognition_probability(prob_curves, prob_ranges)
    if len(metric_ranges):
        plot_metric_specific(metric_ranges)
    plot_playback_descriptive(pbdesc)
    plot_monotonicity(long, mono)
    plot_tornado(uncert)

    write_key_results_v2(long, weights, pca_var, fit_metrics_all, weights_all,
                         eff, avg_ranges, bounds, prob_ranges, cross, fs,
                         metric_ranges, weighted, gamknot, pbdesc, robust, pair,
                         boot_summary, main_names, mono, uncert)

    with open(os.path.join(OUT_DIR, "P2_run_environment.txt"), "w", encoding="utf-8") as f:
        f.write(f"Python {sys.version}\n")
        f.write(f"numpy {np.__version__}\npandas {pd.__version__}\n")
        f.write(f"scikit-learn {sklearn.__version__}\nseaborn {sns.__version__}\n")
        f.write(f"FAST={FAST}; N_BOOT={N_BOOT}\n")

    print("\n" + "=" * 78)
    print("ANSWER SUMMARY")
    print("=" * 78)
    print("Robustness at τ=0.70 (model-averaged d70 and model spread):")
    for _, r in robust.iterrows():
        ar = avg_ranges[(avg_ranges["CallType"] == r["CallType"]) &
                        (avg_ranges["tau"] == 0.70)].iloc[0]
        print(f"  {r['CallType']:>4s}: d50={r['d50_model_avg']:6.1f} m, "
              f"d70={r['d70_model_avg']:6.1f} m; "
              f"eligible [{ar['Range_min_eligible']:.1f},"
              f"{ar['Range_max_eligible']:.1f}], "
              f"all [{r['d70_min_all']:.1f},{r['d70_max_all']:.1f}], "
              f"AUC_log={r['AUC_log']:.3f}")
    if len(cross):
        r = cross.iloc[0]
        if r["Status"] == "crossed" and np.isfinite(r["tau_star"]):
            print(f"  Crossover tv–mod: τ*={r['tau_star']:.3f}, "
                  f"R*={r['Range_star']:.1f} m")
        else:
            print(f"  Crossover tv–mod: {r['Status']}")
    print("  Raw monotonicity violations: " + ", ".join(
        f"{r['CallType']}={int(r['N_violations'])}" for _, r in mono.iterrows()))
    print("  Note: d70 and AUC can rank tv/mod differently; use the model intervals, "
          "not a single rank.")
    print(f"\nAll outputs saved to ./{os.path.relpath(OUT_DIR, BASE_DIR)}/")


def ts_table(curves_main, main_names):
    """Main-model threshold sensitivity table (logistic by default)."""
    rows = []
    for ct in CALL_ORDER:
        for tau in TAU_GRID:
            r, status = solve_range(curves_main[ct], tau)
            rows.append({"CallType": ct, "tau": float(tau),
                         "Range_m": r, "Status": status})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_with_log(main)
