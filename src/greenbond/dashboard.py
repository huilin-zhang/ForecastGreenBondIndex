"""Offline BI-style dashboard and long-form tables from a completed experiment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

BASELINE = "Naive (last value)"


def forecast_facts(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per forecast date and model, suitable for BI filters and aggregation."""
    required = {"date", "actual", BASELINE}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError("Forecasts require date, actual, and the last-value baseline.")
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise").dt.strftime("%Y-%m-%d")
    if predictions["date"].isna().any():
        raise ValueError("Forecast dates must not be missing.")
    if predictions["date"].duplicated().any():
        raise ValueError("Forecast dates must be unique.")
    numeric = predictions.drop(columns="date").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Forecast values must be finite.")
    long = predictions.melt(id_vars=["date", "actual"], var_name="model", value_name="prediction")
    lookup = predictions.set_index("date")[BASELINE]
    long["naive_prediction"] = long["date"].map(lookup)
    long["error"] = long["actual"] - long["prediction"]
    long["absolute_error"] = long["error"].abs()
    long["squared_error"] = long["error"] ** 2
    long["naive_squared_error"] = (long["actual"] - long["naive_prediction"]) ** 2
    return long


def write_dashboard(directory: Path) -> Path:
    """Regenerate presentation artifacts only; never retrain or choose model parameters."""
    directory = Path(directory)
    predictions = pd.read_csv(directory / "predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise").dt.strftime("%Y-%m-%d")
    predictions = predictions.sort_values("date")
    facts = forecast_facts(predictions)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    importance = pd.read_csv(directory / "feature_importance.csv")
    payload = {
        "dates": predictions["date"].tolist(),
        "actual": predictions["actual"].tolist(),
        "models": {c: predictions[c].tolist() for c in predictions.columns if c not in {"date", "actual"}},
        "metadata": metadata,
        "importance": importance.to_dict("records"),
        "importance_model": "Random Forest (random search)" if metadata.get("random_forest") else "Random Forest",
    }
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("templates").joinpath("dashboard.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", serialized).replace("__PLOTLY__", get_plotlyjs())
    destination = directory / "dashboard.html"
    destination.write_text(html, encoding="utf-8")
    facts.to_csv(directory / "forecast_facts.csv", index=False)
    return destination
