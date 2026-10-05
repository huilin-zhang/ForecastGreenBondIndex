import numpy as np
import pandas as pd
import pytest

from greenbond.data import TARGET, align_predictors, normalize, quality_audit, synthetic_data
from greenbond.features import make_features, preprocessing, sequences, split_points
from greenbond.reporting import evaluate


def test_future_observations_cannot_change_prior_features():
    frame = synthetic_data(180)
    original, _ = make_features(frame)
    changed = frame.copy()
    cutoff = frame.index[110]
    changed.loc[cutoff:, :] *= 1000
    revised, _ = make_features(changed)
    pd.testing.assert_frame_equal(original.loc[:cutoff], revised.loc[:cutoff])


def test_rolling_features_exclude_current_target():
    frame = synthetic_data(180)
    X, y = make_features(frame)
    t = X.index[30]
    position = frame.index.get_loc(t)
    assert X.loc[t, "index_lag_1"] == frame[TARGET].iloc[position - 1]
    assert X.loc[t, "index_mean_5"] == pytest.approx(frame[TARGET].iloc[position - 5:position].mean())
    assert y.loc[t] == frame.loc[t, TARGET]


def test_preprocessing_fits_training_only_and_keeps_missing_columns():
    train = pd.DataFrame({"value": [1., 2., 3., np.nan], "empty": [np.nan] * 4})
    test = pd.DataFrame({"value": [1e9, np.nan], "empty": [99., np.nan]})
    prep = preprocessing(scale=True).fit(train)
    bounds = prep.named_steps["clip"].bounds_.copy()
    statistics = prep.named_steps["impute"].statistics_.copy()
    transformed = prep.transform(test)
    np.testing.assert_equal(prep.named_steps["clip"].bounds_, bounds)
    np.testing.assert_equal(prep.named_steps["impute"].statistics_, statistics)
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()
    assert statistics[0] == 2


def test_macro_observation_not_available_until_assumed_release():
    dates = pd.to_datetime(["2020-02-10", "2020-03-16", "2020-03-17"])
    target = pd.DataFrame({TARGET: [130., 131., 132.]}, index=dates)
    daily = pd.DataFrame({"market": [5.]}, index=pd.to_datetime(["2020-02-01"]))
    monthly = pd.DataFrame({"macro": [10.]}, index=pd.to_datetime(["2020-01-31"]))
    aligned = align_predictors(target, daily, monthly, macro_lag_days=45)
    assert pd.isna(aligned.loc["2020-02-10", "macro"])
    assert aligned.loc["2020-03-16", "macro"] == 10
    assert aligned.loc["2020-02-10", "market"] == 5


def test_alignment_does_not_backfill_future_value():
    target = pd.DataFrame({TARGET: [130., 131.]}, index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    future = pd.DataFrame({"market": [9.]}, index=pd.to_datetime(["2020-01-02"]))
    aligned = align_predictors(target, future)
    assert pd.isna(aligned.iloc[0]["market"])
    assert aligned.iloc[1]["market"] == 9


def test_exact_duplicates_removed_but_conflicting_dates_rejected():
    frame = pd.DataFrame({"date": ["2020-01-01"] * 2, TARGET: [130, 130]})
    assert len(normalize(frame, target=True)) == 1
    frame.loc[1, TARGET] = 131
    with pytest.raises(ValueError, match="Conflicting"):
        normalize(frame, target=True)


@pytest.mark.parametrize("value", [-1, 0])
def test_invalid_index_levels_rejected(value):
    with pytest.raises(ValueError, match="positive"):
        normalize(pd.DataFrame({"date": ["2020-01-01"], TARGET: [value]}), target=True)


def test_sql_quality_audit():
    frame = synthetic_data(180)
    audit = quality_audit(frame)
    assert audit["rows"] == 180
    assert audit["duplicate_dates"] == 0
    assert audit["missing_by_column"]["treasury_yield"] == 5


def test_split_and_sequences_include_first_held_out_forecast():
    X = np.arange(100)[:, None]
    train_end, test_start = split_points(100)
    windows = sequences(X, 10)
    first_test = windows[test_start - 9]
    np.testing.assert_array_equal(first_test[:, 0], np.arange(test_start - 9, test_start + 1))
    assert train_end == 60 and test_start == 80
    assert len(windows[test_start - 9:]) == 20


def test_skill_uses_same_naive_test_errors():
    actual = np.array([100., 102., 101.])
    naive = np.array([99., 100., 102.])
    predictions = {"Naive (last value)": naive, "perfect": actual}
    scores = evaluate(actual, predictions).set_index("model")
    assert scores.loc["Naive (last value)", "mse_skill_vs_naive"] == 0
    assert scores.loc["perfect", "mse_skill_vs_naive"] == 1
