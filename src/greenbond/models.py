"""Comparable rolling one-step forecasts, with tuning restricted to past data."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA

from .features import preprocessing, sequences


def random_forest(X, y, train_end, test_start, seed=42, trials=8):
    # Learn changes rather than levels so forecasts can extend beyond training levels.
    changes = y.to_numpy() - X["index_lag_1"].to_numpy()
    model = Pipeline([("preprocess", preprocessing()),
                      ("rf", RandomForestRegressor(n_estimators=160, min_samples_leaf=5,
                                                    random_state=seed, n_jobs=1))])
    baseline = model.fit(X.iloc[:train_end], changes[:train_end])
    offset = X.iloc[test_start:]["index_lag_1"].to_numpy()
    predictions = {"Random Forest": baseline.predict(X.iloc[test_start:]) + offset}
    details = {}
    if trials:
        search = RandomizedSearchCV(
            model,
            {"rf__n_estimators": [80, 160, 240], "rf__max_depth": [4, 8, None],
             "rf__min_samples_leaf": [3, 5, 10], "rf__max_features": [0.5, 1.0]},
            n_iter=trials, cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_squared_error", random_state=seed, n_jobs=1,
            error_score="raise")
        search.fit(X.iloc[:train_end], changes[:train_end])
        predictions["Random Forest (random search)"] = search.predict(X.iloc[test_start:]) + offset
        details = {"best_params": search.best_params_,
                   "cv_mse_changes": float(-search.best_score_), "trials": trials}
        baseline = search.best_estimator_
    importance = dict(zip(X.columns, baseline.named_steps["rf"].feature_importances_.tolist()))
    return predictions, details, importance


def rolling_arima(y, train_end, test_start):
    # Select order using training AIC; no validation or test observations in fitting.
    values = np.asarray(y, dtype=float)
    candidates = []
    for order in [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]:
        try:
            fitted = ARIMA(values[:train_end], order=order).fit()
            if fitted.mle_retvals.get("converged", True) and np.isfinite(fitted.aic):
                candidates.append((fitted.aic, order, fitted))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not candidates:
        raise ValueError("No ARIMA candidate converged on the training data.")
    _, order, state = min(candidates, key=lambda item: item[0])
    state = state.append(values[train_end:test_start], refit=False)
    predictions = []
    for actual in values[test_start:]:
        predictions.append(float(state.forecast(1)[0]))
        # Update state only after issuing the forecast; keep fitted parameters fixed.
        state = state.extend([actual])
    return np.asarray(predictions), {"order": list(order), "selection": "training AIC"}


def lstm_forecasts(X, y, train_end, test_start, output, seed=42,
                   lookback=10, epochs=20, trials=4):
    try:
        import tensorflow as tf
        import keras_tuner as kt
    except ImportError as exc:
        raise RuntimeError('Install the neural extra: pip install -e ".[neural]"') from exc
    from sklearn.preprocessing import StandardScaler

    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    prep = preprocessing(scale=True).fit(X.iloc[:train_end])
    inputs = sequences(prep.transform(X).astype("float32"), lookback)
    offset = lookback - 1
    ntrain, nvalidation = train_end - offset, test_start - offset
    changes = y.to_numpy() - X["index_lag_1"].to_numpy()
    target_scaler = StandardScaler().fit(changes[:train_end, None])
    labels = target_scaler.transform(changes[offset:, None]).astype("float32")
    train = (inputs[:ntrain], labels[:ntrain])
    validation = (inputs[ntrain:nvalidation], labels[ntrain:nvalidation])
    test = inputs[nvalidation:]

    def build(hp):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=inputs.shape[1:]),
            tf.keras.layers.LSTM(hp.Choice("units", [16, 32, 64])),
            tf.keras.layers.Dropout(hp.Choice("dropout", [0.0, 0.1, 0.2])),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [0.0003, 0.001, 0.003])), loss="mse")
        return model

    def stopping():
        return [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                restore_best_weights=True)]

    defaults = kt.HyperParameters()
    defaults.Fixed("units", 32)
    defaults.Fixed("dropout", 0.1)
    defaults.Fixed("learning_rate", 0.001)
    baseline = build(defaults)
    history = baseline.fit(*train, validation_data=validation, epochs=epochs,
                           batch_size=32, shuffle=False, callbacks=stopping(), verbose=0)
    models = {"LSTM": baseline}
    details = {"baseline_epochs": len(history.history["loss"]), "lookback": lookback}
    for name, tuner_type in [("random", kt.RandomSearch), ("bayesian", kt.BayesianOptimization)]:
        if not trials:
            break
        tf.keras.backend.clear_session()
        tuner = tuner_type(build, objective="val_loss", max_trials=trials, seed=seed,
                           directory=str(output / "tuning"), project_name=name,
                           overwrite=False)
        tuner.search(*train, validation_data=validation, epochs=epochs, batch_size=32,
                     shuffle=False, callbacks=stopping(), verbose=0)
        models[f"LSTM ({name} search)"] = tuner.get_best_models(1)[0]
        details[name] = {"best_params": tuner.get_best_hyperparameters(1)[0].values,
                         "best_validation_loss": float(tuner.oracle.get_best_trials(1)[0].score),
                         "trials": trials}
    predictions = {}
    for name, model in models.items():
        deltas = target_scaler.inverse_transform(model.predict(test, verbose=0)).ravel()
        predictions[name] = X.iloc[test_start:]["index_lag_1"].to_numpy() + deltas
    return predictions, details
