# CRP Forecast Anomaly Dashboard

Flask + Plotly dashboard for **spotting anomalies in forecasts before
validating** the next planning cycle. Built around the
`crp_synthetic_dataset.csv` file in this folder.

## What's inside

| File / folder         | Purpose                                                                |
|-----------------------|------------------------------------------------------------------------|
| `app.py`              | Flask routes (HTML pages + JSON APIs powering the Plotly charts)       |
| `data_processing.py`  | Data loading, three anomaly detectors, ranking & KPI computation       |
| `templates/`          | Jinja2 templates (Bootstrap 5)                                         |
| `static/css/style.css`| Dashboard styling                                                       |
| `static/js/dashboard.js` | Plotly defaults, fetch helpers, formatters                          |
| `cache/processed.pkl` | Cached detector results so the app starts instantly                    |
| `crp_synthetic_dataset.csv` | Source dataset                                                    |

## Running

```bash
pip install -r requirements.txt
python app.py
# → open http://127.0.0.1:5000
```

The first run computes the detectors (~30 s); subsequent runs load
results from `cache/processed.pkl` in under 2 s.

## Detectors

1. **Confidence-Interval (CI)** — flags rows whose
   `DifferenceGrossSales` falls outside µ ± z·σ at the chosen level
   (default 44 %, the value the reference study used to reach ≈ 80 % recall).
2. **Isolation Forest** — unsupervised ensemble; flags the most
   "easily-isolated" 5 % of inconsistent rows in (DifferenceGrossSales,
   GrossSalesQuantitySwitched, DifferenceSalesOut, SalesOutQuantitySwitched).
3. **Logistic Regression** — supervised classifier predicting
   `IsInconsistentGrossSales` from engineered features
   (abs / signed-log volumes, gross-to-sell-out ratio + raw quantities)
   with an 80/20 time-aware split.

A composite **suspicion score** ranks items combining the three detectors
plus revision magnitude. See the *Suspicious Items* page.

## Pages

* **Overview** — KPI tiles, monthly trend, revision distribution,
  country breakdown, top-10 most-revised and top-10 suspicious items.
* **Anomaly Detection** — explanation, accuracy / confusion matrix and an
  interactive scatter for each of the three detectors.
* **Forecast vs Actual** — actual vs forecast time-series with a ±1 σ
  revision band, plus snapshot 1 vs 2 overlap.
* **Suspicious Items** — sortable, searchable ranking with drill-down.
* **Item detail** — per-item KPIs, anomaly markers on the time-series,
  per-day Δ Gross Sales, and full snapshot overlay.

## Data conventions

The CSV is consumed as-is; no manual cleaning required. Country is
derived from the last two letters of `SubAreaCode` for the country-level
breakdown chart.
