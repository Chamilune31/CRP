"""
Flask application — CRP Forecast Anomaly Dashboard
===================================================

Pages
  /                  Overview dashboard (KPIs, trends, alerts)
  /anomalies         Three-section anomaly detection page (CI, IF, LR)
  /forecast          Forecast vs Actual comparison + snapshot overlay
  /rankings          Suspicious-item rankings with filters
  /item/<code>       Per-item drill-down

JSON APIs (used by front-end Plotly charts)
  /api/overview
  /api/anomaly_scatter?model=ci|if|lr
  /api/forecast_actual?item=...&subarea=...
  /api/snapshot_overlap?item=...&subarea=...
  /api/rankings_table
  /api/item/<code>
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import data_processing as dp

# --------------------------------------------------------------------------- #
#  App initialisation — pre-compute everything once
# --------------------------------------------------------------------------- #
app = Flask(__name__)

print("[app] booting pipeline…", flush=True)
PIPE = dp.run_pipeline(force=False)
DF = PIPE["df"]
DF["__row_idx"] = np.arange(len(DF))
DF["ci_flag"] = PIPE["ci_flags"]
DF["if_flag"] = PIPE["if_flags"]
DF["lr_flag"] = PIPE["lr_flags"]
DF["if_score"] = PIPE["if_scores"]
DF["lr_proba"] = PIPE["lr_proba"]
RANKINGS = PIPE["rankings"]
KPIS = PIPE["kpis"]
print(f"[app] ready — {len(DF):,} rows, {DF['ItemCode'].nunique():,} items", flush=True)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _to_jsonable(obj):
    """Make a Python object JSON-serialisable (handles numpy / pandas types)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.ndarray,)):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def _filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply optional global filters from query-string."""
    item = request.args.get("item")
    sub = request.args.get("subarea")
    country = request.args.get("country")
    snap = request.args.get("snapshot")
    if item:
        df = df[df["ItemCode"] == item]
    if sub:
        df = df[df["SubAreaCode"] == sub]
    if country:
        df = df[df["Country"] == country]
    if snap:
        df = df[df["SnapshotMonth"] == snap]
    return df


def _sample_indices(df: pd.DataFrame, max_points: int = 4000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=42)


# --------------------------------------------------------------------------- #
#  Page routes
# --------------------------------------------------------------------------- #
@app.route("/")
def page_overview():
    return render_template(
        "dashboard.html",
        kpis=KPIS,
        ci_metrics=PIPE["ci_metrics"],
        if_metrics=PIPE["if_metrics"],
        lr_metrics=PIPE["lr_metrics"],
        countries=sorted(DF["Country"].unique().tolist()),
        snapshots=sorted(DF["SnapshotMonth"].unique().tolist()),
    )


@app.route("/anomalies")
def page_anomalies():
    return render_template(
        "anomalies.html",
        kpis=KPIS,
        ci_metrics=PIPE["ci_metrics"],
        if_metrics=PIPE["if_metrics"],
        lr_metrics=PIPE["lr_metrics"],
        coef_top=_top_coefficients(PIPE["lr_metrics"].get("coefficients", {})),
    )


@app.route("/forecast")
def page_forecast():
    items = RANKINGS["ItemCode"].head(50).tolist()
    return render_template("forecast.html", items=items, kpis=KPIS,
                           subareas=sorted(DF["SubAreaCode"].unique().tolist()))


@app.route("/rankings")
def page_rankings():
    return render_template("rankings.html", kpis=KPIS)


@app.route("/item/<code>")
def page_item(code):
    if code not in RANKINGS["ItemCode"].values:
        return f"Item {code} not found.", 404
    rk_row = RANKINGS[RANKINGS["ItemCode"] == code].iloc[0].to_dict()
    return render_template("item_detail.html", code=code,
                           rk=_to_jsonable(rk_row), kpis=KPIS)


# --------------------------------------------------------------------------- #
#  JSON APIs
# --------------------------------------------------------------------------- #
@app.route("/api/overview")
def api_overview():
    sub = _filter(DF)
    # Time-series of inconsistency rate by month
    by_month = (sub.assign(m=sub["SalesDate"].dt.to_period("M").astype(str))
                  .groupby("m")
                  .agg(rows=("ItemCode", "size"),
                       inc=("IsInconsistentGrossSales", "sum"),
                       rev=("RevisionFlag", "sum"),
                       ci=("ci_flag", "sum"),
                       iso=("if_flag", "sum"),
                       lr=("lr_flag", "sum"))
                  .reset_index())
    by_month["inc_pct"] = (100 * by_month["inc"] / by_month["rows"]).round(2)
    by_month["rev_pct"] = (100 * by_month["rev"] / by_month["rows"]).round(2)
    by_month["ci_pct"] = (100 * by_month["ci"] / by_month["rows"]).round(2)
    by_month["if_pct"] = (100 * by_month["iso"] / by_month["rows"]).round(2)
    by_month["lr_pct"] = (100 * by_month["lr"] / by_month["rows"]).round(2)

    # Distribution of revisions (non-null)
    rev = sub["ForecastRevisions"].dropna()
    bins = np.linspace(rev.quantile(0.01), rev.quantile(0.99), 30) if len(rev) else []
    if len(bins):
        hist, edges = np.histogram(rev.clip(bins[0], bins[-1]), bins=bins)
    else:
        hist, edges = [], []

    # Top-10 most-revised items
    top_rev = (RANKINGS.sort_values("avg_abs_revision", ascending=False)
                       .head(10)[["ItemCode", "avg_abs_revision",
                                  "n_revised", "rows"]]
                       .to_dict("records"))
    # Top-10 by suspicion
    top_sus = (RANKINGS.head(10)[["ItemCode", "suspicion_score",
                                   "pct_anomaly_if", "pct_anomaly_ci",
                                   "pct_anomaly_lr"]]
               .to_dict("records"))

    # Country-level inconsistency rate
    by_country = (sub.groupby("Country")
                   .agg(rows=("ItemCode", "size"),
                        inc=("IsInconsistentGrossSales", "sum"),
                        rev=("RevisionFlag", "sum"))
                   .assign(inc_pct=lambda d: (100 * d["inc"] / d["rows"]).round(2),
                           rev_pct=lambda d: (100 * d["rev"] / d["rows"]).round(2))
                   .reset_index()
                   .sort_values("inc_pct", ascending=False))

    return jsonify(_to_jsonable({
        "by_month": by_month.to_dict("list"),
        "rev_hist": {"counts": list(hist),
                     "edges": list(edges)},
        "top_rev": top_rev,
        "top_sus": top_sus,
        "by_country": by_country.to_dict("list"),
    }))


@app.route("/api/anomaly_scatter")
def api_anomaly_scatter():
    """Return points for normal vs anomaly scatter for a chosen model."""
    model = request.args.get("model", "if")
    sub = _filter(DF)
    sub = sub[sub["DifferenceGrossSales"] != 0]
    flag_col = {"ci": "ci_flag", "if": "if_flag", "lr": "lr_flag"}.get(model, "if_flag")

    norm = sub[sub[flag_col] == 0]
    anom = sub[sub[flag_col] == 1]
    norm = _sample_indices(norm, 3000)
    anom = _sample_indices(anom, 3000)

    payload = {
        "normal": {
            "x": norm["DifferenceGrossSales"].astype(float).tolist(),
            "y": norm["GrossSalesQuantitySwitched"].astype(float).tolist(),
            "item": norm["ItemCode"].tolist(),
            "date": norm["SalesDate"].dt.strftime("%Y-%m-%d").tolist(),
        },
        "anomaly": {
            "x": anom["DifferenceGrossSales"].astype(float).tolist(),
            "y": anom["GrossSalesQuantitySwitched"].astype(float).tolist(),
            "item": anom["ItemCode"].tolist(),
            "date": anom["SalesDate"].dt.strftime("%Y-%m-%d").tolist(),
        },
        "ci_bounds": [PIPE["ci_metrics"]["lower_bound"],
                      PIPE["ci_metrics"]["upper_bound"]],
        "model": model,
    }
    return jsonify(_to_jsonable(payload))


@app.route("/api/forecast_actual")
def api_forecast_actual():
    """Time-series for actual sales vs forecast for an (item, subarea)."""
    item = request.args.get("item", "")
    sub = request.args.get("subarea", "")
    sel = DF[(DF["ItemCode"] == item)]
    if sub:
        sel = sel[sel["SubAreaCode"] == sub]
    if not len(sel):
        return jsonify({"error": "no data"})

    # Actual sales aggregated by date
    actual = (sel.groupby("SalesDate")
                 .agg(actual=("GrossSalesQuantitySwitched", "sum"))
                 .reset_index().sort_values("SalesDate"))

    # Forecast aggregated by ForecastMonth (mid-month)
    fc = (sel.dropna(subset=["ForecastSalesInQuantity"])
            .groupby("ForecastMonth")
            .agg(forecast=("ForecastSalesInQuantity", "mean"))
            .reset_index())
    fc["ForecastMonth_dt"] = pd.to_datetime(fc["ForecastMonth"] + "-15",
                                            errors="coerce")
    fc = fc.dropna(subset=["ForecastMonth_dt"]).sort_values("ForecastMonth_dt")

    # Confidence band on forecast: ±1 SD of revisions
    sd = sel["ForecastRevisions"].dropna().std() or 0
    fc["upper"] = fc["forecast"] + sd
    fc["lower"] = fc["forecast"] - sd

    return jsonify(_to_jsonable({
        "actual_x": actual["SalesDate"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_y": actual["actual"].astype(float).tolist(),
        "fc_x": fc["ForecastMonth_dt"].dt.strftime("%Y-%m-%d").tolist(),
        "fc_y": fc["forecast"].astype(float).tolist(),
        "fc_upper": fc["upper"].astype(float).tolist(),
        "fc_lower": fc["lower"].astype(float).tolist(),
        "sd": float(sd),
    }))


@app.route("/api/snapshot_overlap")
def api_snapshot_overlap():
    """Two snapshots' forecasts for the same (item, subarea) on one chart."""
    item = request.args.get("item", "")
    sub = request.args.get("subarea", "")
    sel = DF[DF["ItemCode"] == item]
    if sub:
        sel = sel[sel["SubAreaCode"] == sub]
    if not len(sel):
        return jsonify({"error": "no data"})

    snaps = sorted(sel["SnapshotMonth"].unique().tolist())
    if len(snaps) < 2:
        return jsonify({"error": "fewer than 2 snapshots", "snaps": snaps})

    s1, s2 = snaps[0], snaps[1]
    series = []
    for s in (s1, s2):
        sub_s = (sel[sel["SnapshotMonth"] == s]
                 .dropna(subset=["ForecastSalesInQuantity"])
                 .groupby("ForecastMonth")
                 .agg(qty=("ForecastSalesInQuantity", "mean"))
                 .reset_index())
        sub_s["dt"] = pd.to_datetime(sub_s["ForecastMonth"] + "-15", errors="coerce")
        sub_s = sub_s.dropna(subset=["dt"]).sort_values("dt")
        series.append({
            "snapshot": s,
            "x": sub_s["dt"].dt.strftime("%Y-%m-%d").tolist(),
            "y": sub_s["qty"].astype(float).tolist(),
        })

    return jsonify(_to_jsonable({"series": series, "all_snaps": snaps}))


@app.route("/api/rankings_table")
def api_rankings_table():
    n = int(request.args.get("limit", 200))
    minrows = int(request.args.get("minrows", 0))
    rk = RANKINGS
    if minrows > 0:
        rk = rk[rk["rows"] >= minrows]
    cols = ["ItemCode", "rows", "n_inconsistent", "n_revised",
            "pct_anomaly_ci", "pct_anomaly_if", "pct_anomaly_lr",
            "pct_revised", "avg_abs_revision", "avg_revision_ratio",
            "max_diff_gross", "suspicion_score"]
    rows = rk[cols].head(n).to_dict("records")
    return jsonify(_to_jsonable(rows))


@app.route("/api/item/<code>")
def api_item(code):
    sel = DF[DF["ItemCode"] == code]
    if not len(sel):
        return jsonify({"error": "not found"})
    sub = (sel.groupby("SalesDate")
              .agg(actual=("GrossSalesQuantitySwitched", "sum"),
                   so=("SalesOutQuantitySwitched", "sum"),
                   diff_gross=("DifferenceGrossSales", "sum"),
                   inc=("IsInconsistentGrossSales", "max"),
                   ci=("ci_flag", "max"),
                   ifl=("if_flag", "max"),
                   lr=("lr_flag", "max"),
                   if_score=("if_score", "max"),
                   lr_proba=("lr_proba", "max"))
              .reset_index().sort_values("SalesDate"))
    fc = (sel.dropna(subset=["ForecastSalesInQuantity"])
            .groupby("ForecastMonth")
            .agg(forecast=("ForecastSalesInQuantity", "mean"),
                 revisions=("ForecastRevisions",
                            lambda s: float(np.nanmean(np.abs(s)) or 0)))
            .reset_index())
    fc["dt"] = pd.to_datetime(fc["ForecastMonth"] + "-15", errors="coerce")
    fc = fc.dropna(subset=["dt"]).sort_values("dt")
    rk_row = RANKINGS[RANKINGS["ItemCode"] == code].iloc[0].to_dict()

    # Snapshots (multiple) — used for overlap chart
    snaps_payload = []
    for s in sorted(sel["SnapshotMonth"].unique().tolist()):
        ss = (sel[sel["SnapshotMonth"] == s]
              .dropna(subset=["ForecastSalesInQuantity"])
              .groupby("ForecastMonth")
              .agg(qty=("ForecastSalesInQuantity", "mean")).reset_index())
        ss["dt"] = pd.to_datetime(ss["ForecastMonth"] + "-15", errors="coerce")
        ss = ss.dropna(subset=["dt"]).sort_values("dt")
        snaps_payload.append({
            "snapshot": s,
            "x": ss["dt"].dt.strftime("%Y-%m-%d").tolist(),
            "y": ss["qty"].astype(float).tolist(),
        })

    return jsonify(_to_jsonable({
        "item": code,
        "actual_x": sub["SalesDate"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_y": sub["actual"].astype(float).tolist(),
        "so_y": sub["so"].astype(float).tolist(),
        "diff_y": sub["diff_gross"].astype(float).tolist(),
        "ci_flag": sub["ci"].astype(int).tolist(),
        "if_flag": sub["ifl"].astype(int).tolist(),
        "lr_flag": sub["lr"].astype(int).tolist(),
        "fc_x": fc["dt"].dt.strftime("%Y-%m-%d").tolist(),
        "fc_y": fc["forecast"].astype(float).tolist(),
        "fc_revisions": fc["revisions"].astype(float).tolist(),
        "snapshots": snaps_payload,
        "ranking": rk_row,
        "rows": len(sel),
    }))


# --------------------------------------------------------------------------- #
def _top_coefficients(coefs: dict, k: int = 8):
    return sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
