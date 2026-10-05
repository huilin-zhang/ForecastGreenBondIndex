"""Features for forecasting level y[t] using information through t-1."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET


def make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if len(frame) < 60:
        raise ValueError("At least 60 ordered observations are required.")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Dates must be sorted and unique.")
    y = frame[TARGET]
    # Shift all predictors: same-day market closes are unavailable at t-1.
    X = frame.drop(columns=[TARGET]).ffill().shift(1).copy()
    for lag in (1, 5, 10, 20):
        X[f"index_lag_{lag}"] = y.shift(lag)
    past = y.shift(1)
    X["index_mean_5"] = past.rolling(5).mean()
    X["index_mean_20"] = past.rolling(20).mean()
    X["index_std_20"] = past.rolling(20).std()
    X["index_change_1"] = y.diff().shift(1)
    X["index_return_5"] = y.pct_change(5, fill_method=None).shift(1)
    # Discard only the deterministic feature warm-up, not rows missing predictors.
    return X.iloc[20:].replace([np.inf, -np.inf], np.nan), y.iloc[20:]


class QuantileClipper(TransformerMixin, BaseEstimator):
    """Clip predictor outliers using thresholds learned only on training rows."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        self.bounds_ = frame.quantile([self.lower, self.upper]).to_numpy()
        # All-missing columns remain NaN and are handled by the imputer.
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), *self.bounds_)


def preprocessing(scale=False):
    steps = [("clip", QuantileClipper()),
             ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def split_points(n: int, train_fraction=.6, validation_fraction=.2):
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    if train_end < 30 or validation_end - train_end < 10 or n - validation_end < 10:
        raise ValueError("Not enough observations for 60/20/20 chronological splits.")
    return train_end, validation_end


def sequences(X, lookback):
    if lookback < 1 or len(X) < lookback:
        raise ValueError("Sequence lookback must be positive and fit the dataset.")
    return np.stack([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))])
