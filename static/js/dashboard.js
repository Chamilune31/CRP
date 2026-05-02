/* Shared front-end utilities — refined Plotly theme + helpers */

window.CRP = (function () {
  const layoutDefaults = {
    margin: { t: 24, r: 16, b: 56, l: 64 },
    autosize: true,
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: {
      family: '"Inter", -apple-system, "Segoe UI", Roboto, sans-serif',
      size: 12,
      color: "#0f172a",
    },
    xaxis: {
      gridcolor: "#eef0f3",
      zerolinecolor: "#e2e8f0",
      tickcolor: "#cbd5e1",
      linecolor: "#e2e8f0",
      tickfont: { color: "#475569", size: 11 },
      titlefont: { color: "#475569", size: 12 },
    },
    yaxis: {
      gridcolor: "#eef0f3",
      zerolinecolor: "#e2e8f0",
      tickcolor: "#cbd5e1",
      linecolor: "#e2e8f0",
      tickfont: { color: "#475569", size: 11 },
      titlefont: { color: "#475569", size: 12 },
    },
    legend: {
      orientation: "h",
      y: -0.22,
      font: { size: 11, color: "#475569" },
      bgcolor: "rgba(0,0,0,0)",
    },
    hoverlabel: {
      bgcolor: "#0f172a",
      bordercolor: "#0f172a",
      font: { color: "#fff", family: '"Inter", sans-serif', size: 12 },
    },
    colorway: ["#2563eb", "#0891b2", "#d97706", "#16a34a", "#7c3aed",
               "#dc2626", "#0d9488", "#db2777"],
  };

  const config = {
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    responsive: true,
    toImageButtonOptions: { format: "png", filename: "crp_chart", scale: 2 },
  };

  function plot(elementId, traces, layoutOverrides = {}) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const layout = Object.assign({}, layoutDefaults, layoutOverrides, {
      xaxis: Object.assign({}, layoutDefaults.xaxis, layoutOverrides.xaxis || {}),
      yaxis: Object.assign({}, layoutDefaults.yaxis, layoutOverrides.yaxis || {}),
    });
    return Plotly.newPlot(el, traces, layout, config);
  }

  function fmt(n, digits = 0) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(digits + 1) + "M";
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(digits + 1) + "K";
    return Number(n).toFixed(digits);
  }

  function pct(n, digits = 1) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits) + "%";
  }

  async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    return await r.json();
  }

  return { plot, fmt, pct, fetchJSON };
})();
