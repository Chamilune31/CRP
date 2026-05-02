"""
Data processing & ML model module for the CRP Forecast Anomaly Dashboard.

Loads the synthetic dataset and computes:
  1. Confidence-interval anomaly detection (tunable level)
  2. Isolation-Forest anomaly detection
  3. Logistic-Regression classifier with vectorised engineered features
  4. Forecast-revision metrics & per-item suspicion rankings

Heavy results are cached to disk under ./cache so the Flask app starts fast.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "crp_synthetic_dataset.csv"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "processed.pkl"


# --------------------------------------------------------------------------- #
#  Loading & basic preparation
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["SalesDate"] = pd.to_datetime(df["SalesDate"], errors="coerce")
    df["ExecutionDate"] = pd.to_datetime(df["ExecutionDate"], errors="coerce")
    df["SnapshotMonth"] = df["SnapshotMonth"].astype(str)
    df["ForecastMonth"] = df["ForecastMonth"].astype(str)
    df["Country"] = df["SubAreaCode"].str[-2:]
    return df


# --------------------------------------------------------------------------- #
#  1. Confidence-interval anomaly detection
# --------------------------------------------------------------------------- #
def confidence_interval_anomalies(df: pd.DataFrame, level: float = 0.44):
    """Flag rows whose `DifferenceGrossSales` falls outside a `level`-CI built
    from the population of non-zero differences. Returns
    (flag_array, metrics_dict).
    """
    sample = df.loc[df["DifferenceGrossSales"] != 0,
                    "DifferenceGrossSales"].astype(float)
    n = len(sample)
    mean = float(sample.mean()) if n else 0.0
    sd = float(sample.std(ddof=1)) if n > 1 else 0.0
    z = stats.norm.ppf(0.5 + level / 2.0)
    lower = mean - z * sd
    upper = mean + z * sd

    flags = np.zeros(len(df), dtype=int)
    diff = df["DifferenceGrossSales"].astype(float).values
    flags[(diff != 0) & ((diff < lower) | (diff > upper))] = 1

    actual = df["IsInconsistentGrossSales"].astype(int).values
    metrics = _classification_metrics(actual, flags)
    metrics.update({"lower_bound": float(lower), "upper_bound": float(upper),
                    "level": float(level), "sample_n": int(n),
                    "mean": mean, "sd": sd})
    return flags, metrics


# --------------------------------------------------------------------------- #
#  2. Isolation Forest
# --------------------------------------------------------------------------- #
def isolation_forest_anomalies(df: pd.DataFrame, contamination: float = 0.05,
                               random_state: int = 42):
    feats = df[["DifferenceGrossSales", "GrossSalesQuantitySwitched",
                "DifferenceSalesOut", "SalesOutQuantitySwitched"
                ]].astype(float).fillna(0).values

    mask_inconsistent = df["DifferenceGrossSales"].values != 0
    flags = np.zeros(len(df), dtype=int)
    scores = np.zeros(len(df), dtype=float)

    if mask_inconsistent.sum() > 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(feats[mask_inconsistent])
        iso = IsolationForest(contamination=contamination,
                              random_state=random_state, n_jobs=-1)
        iso.fit(X)
        sub_flags = (iso.predict(X) == -1).astype(int)
        sub_scores = -iso.score_samples(X)
        flags[mask_inconsistent] = sub_flags
        scores[mask_inconsistent] = sub_scores

    actual = df["IsInconsistentGrossSales"].astype(int).values
    metrics = _classification_metrics(actual, flags)
    metrics.update({"contamination": float(contamination),
                    "n_scored": int(mask_inconsistent.sum())})
    return flags, scores, metrics


# --------------------------------------------------------------------------- #
#  3. Logistic Regression
# --------------------------------------------------------------------------- #
def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight, fully-vectorised feature engineering.

    Computes log-scaled magnitudes, signed-log values, absolute differences,
    and a gross/sell-out ratio.  We deliberately AVOID per-item rolling/lag
    on 75 K item groups: it overwhelms startup with little benefit on this
    synthetic 500 K-row sample.
    """
    out = df.copy()
    for col in ["DifferenceGrossSales", "DifferenceSalesOut"]:
        out[f"{col}_abs"] = out[col].abs().astype(float)
        out[f"{col}_log"] = (np.sign(out[col])
                             * np.log1p(out[col].abs())).astype(float)
    out["gross_to_so_ratio"] = (
        out["GrossSalesQuantitySwitched"]
        / out["SalesOutQuantitySwitched"].replace(0, np.nan))
    out = out.fillna(0)
    return out


def logistic_regression_model(df: pd.DataFrame, sample_n: int = 150_000):
    feats_df = build_engineered_features(df).reset_index(drop=True)
    feature_cols = [c for c in feats_df.columns
                    if c.endswith(("_abs", "_log"))]
    feature_cols += ["gross_to_so_ratio",
                     "DifferenceGrossSales", "DifferenceSalesOut",
                     "GrossSalesQuantitySwitched",
                     "SalesOutQuantitySwitched"]

    X = feats_df[feature_cols].astype(float).values
    y = feats_df["IsInconsistentGrossSales"].astype(int).values

    sort_idx = feats_df["SalesDate"].argsort().values
    split = int(0.8 * len(sort_idx))
    train_idx = sort_idx[:split]
    test_idx = sort_idx[split:]

    if sample_n is not None and len(train_idx) > sample_n:
        rng = np.random.default_rng(42)
        train_idx = rng.choice(train_idx, size=sample_n, replace=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    X_full = scaler.transform(X)

    model = LogisticRegression(class_weight="balanced", solver="lbfgs",
                               max_iter=100, n_jobs=-1)
    model.fit(X_train, y[train_idx])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = _classification_metrics(y[test_idx], y_pred)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y[test_idx], y_proba))
    except ValueError:
        metrics["roc_auc"] = None
    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["features"] = feature_cols
    metrics["coefficients"] = {
        f: float(c) for f, c in zip(feature_cols, model.coef_[0])}

    full_pred = model.predict(X_full).astype(int)
    full_proba = model.predict_proba(X_full)[:, 1].astype(float)
    return full_pred, full_proba, metrics


# --------------------------------------------------------------------------- #
#  Metrics helper
# --------------------------------------------------------------------------- #
def _classification_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# --------------------------------------------------------------------------- #
#  Per-item suspicion rankings
# --------------------------------------------------------------------------- #
def build_rankings(df: pd.DataFrame, ci_flags, if_flags, lr_flags,
                   if_scores, lr_proba) -> pd.DataFrame:
    work = df.copy()
    work["ci_flag"] = ci_flags
    work["if_flag"] = if_flags
    work["lr_flag"] = lr_flags
    work["if_score"] = if_scores
    work["lr_proba"] = lr_proba

    work["abs_revision"] = work["ForecastRevisions"].abs()
    work["abs_diff_gross"] = work["DifferenceGrossSales"].abs()
    grp = work.groupby("ItemCode", sort=False)
    rk = grp.agg(
        rows=("ItemCode", "size"),
        n_ci=("ci_flag", "sum"),
        n_if=("if_flag", "sum"),
        n_lr=("lr_flag", "sum"),
        n_inconsistent=("IsInconsistentGrossSales", "sum"),
        n_revised=("RevisionFlag", "sum"),
        avg_abs_revision=("abs_revision", "mean"),
        avg_revision_ratio=("RevisionRatioAbsolute", "mean"),
        max_diff_gross=("abs_diff_gross", "max"),
        avg_if_score=("if_score", "mean"),
        avg_lr_proba=("lr_proba", "mean"),
    ).reset_index()

    rk["pct_anomaly_ci"] = (rk["n_ci"] / rk["rows"]).round(4)
    rk["pct_anomaly_if"] = (rk["n_if"] / rk["rows"]).round(4)
    rk["pct_anomaly_lr"] = (rk["n_lr"] / rk["rows"]).round(4)
    rk["pct_revised"] = (rk["n_revised"] / rk["rows"]).round(4)

    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0
    rk["suspicion_score"] = (
        0.30 * _norm(rk["pct_anomaly_if"])
        + 0.20 * _norm(rk["pct_anomaly_ci"])
        + 0.20 * _norm(rk["pct_anomaly_lr"])
        + 0.15 * _norm(rk["avg_abs_revision"].fillna(0))
        + 0.15 * _norm(rk["max_diff_gross"])
    ).round(4)
    rk = rk.sort_values("suspicion_score", ascending=False).reset_index(drop=True)
    return rk


# --------------------------------------------------------------------------- #
#  KPI summary
# --------------------------------------------------------------------------- #
def compute_kpis(df: pd.DataFrame, ci_flags, if_flags, lr_flags) -> dict:
    n = len(df)
    revised_items = df.loc[df["RevisionFlag"] == 1, "ItemCode"].nunique()
    total_items = df["ItemCode"].nunique()
    inc = df["IsInconsistentGrossSales"] == 1
    ok = df["IsInconsistentGrossSales"] == 0

    kpis = {
        "rows": int(n),
        "items": int(total_items),
        "subareas": int(df["SubAreaCode"].nunique()),
        "snapshots": int(df["SnapshotMonth"].nunique()),
        "date_min": str(df["SalesDate"].min().date()),
        "date_max": str(df["SalesDate"].max().date()),
        "pct_revised_rows": round(100.0 * (df["RevisionFlag"] == 1).mean(), 2),
        "pct_revised_items": round(100.0 * revised_items / max(total_items, 1), 2),
        "pct_inconsistent_gross": round(100.0 * inc.mean(), 2),
        "pct_inconsistent_so": round(100.0 * (df["IsInconsistentSalesOut"] == 1).mean(), 2),
        "pct_anomaly_ci": round(100.0 * float(np.mean(ci_flags)), 2),
        "pct_anomaly_if": round(100.0 * float(np.mean(if_flags)), 2),
        "pct_anomaly_lr": round(100.0 * float(np.mean(lr_flags)), 2),
        "avg_abs_revision_inconsistent": (
            round(float(np.nanmean(np.abs(df.loc[inc, "ForecastRevisions"]))), 2)
            if inc.any() else 0.0),
        "avg_abs_revision_consistent": (
            round(float(np.nanmean(np.abs(df.loc[ok, "ForecastRevisions"]))), 2)
            if ok.any() else 0.0),
    }
    return kpis


# --------------------------------------------------------------------------- #
#  End-to-end pipeline + caching
# --------------------------------------------------------------------------- #
def run_pipeline(force: bool = False) -> dict:
    if CACHE_FILE.exists() and not force:
        try:
            with open(CACHE_FILE, "rb") as fh:
                cached = pickle.load(fh)
            print("[cache] loaded pre-computed pipeline; loading df…", flush=True)
            cached["df"] = load_dataset()
            return cached
        except Exception as exc:
            print(f"[cache] failed to load ({exc}); recomputing", flush=True)

    print("[pipeline] loading dataset")
    df = load_dataset()
    print("[pipeline] CI detection")
    ci_flags, ci_metrics = confidence_interval_anomalies(df, level=0.44)
    print("[pipeline] Isolation-Forest detection")
    if_flags, if_scores, if_metrics = isolation_forest_anomalies(df)
    print("[pipeline] Logistic-Regression model")
    lr_flags, lr_proba, lr_metrics = logistic_regression_model(df)
    print("[pipeline] rankings")
    rankings = build_rankings(df, ci_flags, if_flags, lr_flags,
                              if_scores, lr_proba)
    kpis = compute_kpis(df, ci_flags, if_flags, lr_flags)

    # Cache only the lightweight pieces; df is reloaded on demand.
    cache_bundle = {
        "ci_flags": ci_flags, "ci_metrics": ci_metrics,
        "if_flags": if_flags, "if_scores": if_scores, "if_metrics": if_metrics,
        "lr_flags": lr_flags, "lr_proba": lr_proba, "lr_metrics": lr_metrics,
        "rankings": rankings,
        "kpis": kpis,
    }
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(cache_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print("[pipeline] cached ->", CACHE_FILE, flush=True)
    bundle = dict(cache_bundle)
    bundle["df"] = df
    return bundle


if __name__ == "__main__":
    out = run_pipeline(force=True)
    print("KPIs:", out["kpis"])
    print("CI:", out["ci_metrics"])
    print("IF:", out["if_metrics"])
    print("LR keys:", list(out["lr_metrics"].keys()))
    print(out["rankings"].head())
