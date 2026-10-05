"""A single reproducible experiment, with no test-set model selection."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import time

from .data import quality_audit
from .features import make_features, split_points
from .models import random_forest, rolling_arima
from .reporting import write_report


def run(frame, output: Path, *, source="local", seed=42, lookback=10,
        rf_trials=8, neural=False, epochs=20, lstm_trials=4, macro_lag_days=None):
    # A fresh directory avoids mixing results or resuming unrelated tuner trials.
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a fresh directory.")
    if lookback < 1 or epochs < 1 or rf_trials < 0 or lstm_trials < 0:
        raise ValueError("Lookback/epochs must be positive; search budgets nonnegative.")
    start = time.perf_counter()
    X, y = make_features(frame)
    # Shared warm-up for every model, including the optional LSTM sequence context.
    if lookback > 1:
        X, y = X.iloc[lookback - 1:], y.iloc[lookback - 1:]
    train_end, test_start = split_points(len(y))
    if train_end < lookback + 20:
        raise ValueError("Lookback too large for the training sample.")
    output.mkdir(parents=True)
    predictions = {"Naive (last value)": X.iloc[test_start:]["index_lag_1"].to_numpy()}
    arima, arima_details = rolling_arima(y, train_end, test_start)
    predictions["ARIMA"] = arima
    forest, rf_details, importance = random_forest(X, y, train_end, test_start, seed, rf_trials)
    predictions.update(forest)
    lstm_details = None
    if neural:
        from .models import lstm_forecasts
        neural_predictions, lstm_details = lstm_forecasts(
            X, y, train_end, test_start, output, seed, lookback, epochs, lstm_trials)
        predictions.update(neural_predictions)
    packages = ["numpy", "pandas", "scikit-learn", "statsmodels", "matplotlib"]
    if neural:
        packages += ["tensorflow", "keras-tuner"]
    metadata = {
        "source": source, "seed": seed, "horizon": "one observation ahead",
        "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in packages},
        "input_sha256": hashlib.sha256(frame.to_csv().encode()).hexdigest(),
        "quality": quality_audit(frame), "feature_count": X.shape[1],
        "macro_lag_days": macro_lag_days,
        "splits": {"train_start": str(y.index[0].date()),
                   "train_end": str(y.index[train_end - 1].date()),
                   "validation_start": str(y.index[train_end].date()),
                   "validation_end": str(y.index[test_start - 1].date()),
                   "test_start": str(y.index[test_start].date()),
                   "test_end": str(y.index[-1].date()),
                   "train_rows": train_end, "validation_rows": test_start - train_end,
                   "test_rows": len(y) - test_start},
        "arima": arima_details, "random_forest": rf_details, "lstm": lstm_details,
        "model_seconds": round(time.perf_counter() - start, 3),
    }
    return write_report(output, frame, y, test_start, predictions, metadata, importance)
