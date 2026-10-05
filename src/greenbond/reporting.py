"""Machine-readable results and an English report with exportable figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COLORS = ["#64748b", "#b45309", "#0f766e", "#22a084", "#7c3aed", "#2563eb", "#dc2626"]


def evaluate(actual, predictions):
    naive_mse = mean_squared_error(actual, predictions["Naive (last value)"])
    rows = []
    for model, predicted in predictions.items():
        mse = mean_squared_error(actual, predicted)
        rows.append({"model": model, "mae": mean_absolute_error(actual, predicted),
                     "rmse": np.sqrt(mse), "mse": mse, "r2": r2_score(actual, predicted),
                     "mse_skill_vs_naive": 1 - mse / naive_mse if naive_mse else np.nan})
    return pd.DataFrame(rows)


def write_report(output: Path, frame, y, test_start, predictions, metadata, importance):
    dates, actual = y.index[test_start:], y.iloc[test_start:].to_numpy()
    metrics = evaluate(actual, predictions)
    metrics.to_csv(output / "metrics.csv", index=False)
    prediction_frame = pd.DataFrame({"actual": actual, **predictions}, index=dates)
    prediction_frame.to_csv(output / "predictions.csv", index_label="date")
    (output / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (output / "metrics.json").write_text(metrics.to_json(orient="records", indent=2), encoding="utf-8")
    figures = output / "figures"
    figures.mkdir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    qualifier = "Synthetic demonstration" if metadata["source"] == "synthetic" else "Local licensed-data experiment"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(frame.index, frame["green_bond_index"], color="#0f766e", lw=1.4)
    ax.axvline(pd.Timestamp(metadata["splits"]["validation_start"]), color="#b45309", ls="--", label="Validation starts")
    ax.axvline(dates[0], color="#7c3aed", ls="--", label="Test starts")
    ax.set(title=f"{qualifier} | Chronological split", ylabel="Index level", xlabel="Date")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "index_history.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(dates, actual, color="#111827", lw=2, label="Observed")
    for (model, predicted), color in zip(predictions.items(), COLORS):
        axes[0].plot(dates, predicted, label=model, lw=1, alpha=.8, color=color)
        axes[1].plot(dates, actual - predicted, lw=.8, alpha=.65, color=color)
    axes[0].set(title=f"{qualifier} | Rolling one-step forecasts", ylabel="Index level")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].axhline(0, color="#111827", lw=.7)
    axes[1].set(ylabel="Forecast error", xlabel="Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figures / "forecasts.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(metrics["model"], metrics["rmse"], color=COLORS[:len(metrics)])
    ax.invert_yaxis()
    ax.set(title=f"{qualifier} | Test error", xlabel="RMSE (index points; lower is better)")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=160)
    plt.close(fig)
    ranked = pd.Series(importance).sort_values(ascending=False)
    ranked.rename("importance").to_csv(output / "feature_importance.csv", index_label="feature")
    fig, ax = plt.subplots(figsize=(9, 5))
    ranked.head(12).sort_values().plot.barh(ax=ax, color="#0f766e")
    ax.set(title="Random Forest | Feature importance", xlabel="Impurity-based importance", ylabel="")
    fig.tight_layout()
    fig.savefig(figures / "feature_importance.png", dpi=160)
    plt.close(fig)
    from .dashboard import write_dashboard
    write_dashboard(output)
    table = ["| Model | MAE | RMSE | R2 | MSE skill vs naive |",
             "| :--- | ---: | ---: | ---: | ---: |"]
    for row in metrics.to_dict("records"):
        table.append(f"| {row['model']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | {row['mse_skill_vs_naive']:.2%} |")
    report = f"""# Forecasting the Green Bond Index: experiment report

[Open the interactive dashboard](dashboard.html) · [Tableau / Power BI data](forecast_facts.csv)

**Data:** {qualifier}. **Horizon:** next observed index date. **Test observations:** {len(actual)}.

All models forecast the same dates in original index units. Predictors are shifted by one observation.
Parameters and preprocessing use training data; LSTM early stopping and search use validation only.
ARIMA state updates after each forecast, without parameter refitting. Prior test observations become
available for subsequent one-step forecasts. This is not a fixed-origin multistep forecast.

{chr(10).join(table)}

Positive skill means lower MSE than last-value persistence; negative skill means worse.
High R2 on persistent index levels does not establish economically useful forecasting skill.

![Test forecasts](figures/forecasts.png)
![Test error](figures/model_comparison.png)
![Feature importance](figures/feature_importance.png)

## Reproducibility

See [run metadata](run.json), [metrics](metrics.csv), and [dated predictions](predictions.csv).
Feature importance describes the fitted forest and is not causal evidence. Synthetic results
are a software demonstration, not evidence of performance on the S&P Green Bond Index.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    return metrics
