# Forecast design and evaluation

## Forecast question

Predict the S&P Green Bond Index level at the next observed index date using information
available through the previous observed date. A horizon is one observation, not one calendar
day. Holidays and gaps follow the index calendar. Cross-market close times may require
additional per-series availability lags in a production experiment.

## Data and features

The pipeline validates numeric observations and dates, removes exact duplicate rows, and
rejects conflicting dates. It preserves index targets and audits missingness using SQLite.
Source data are aligned by forward fill; neither backward fill nor linear interpolation is
used. Monthly inputs need genuine availability dates or an explicit release-lag assumption.
The supplied legacy loader assumes 45 calendar days after the monthly observation date.
This does not eliminate revision bias in current-vintage macroeconomic data.

Predictors are shifted by one observation. Index lags (1, 5, 10, 20), rolling means (5, 20),
rolling standard deviation (20), prior change and prior five-observation return use past
values only. The 20-observation feature warm-up and a shared sequence warm-up are removed
before splitting. Missing predictors are retained for training-fitted median imputation.
Predictor outliers are clipped at training 1st/99th percentiles; market moves in the target
are never removed. All-missing predictor columns are retained with zero imputation.

## Splits and selection

After common warm-up, split observations chronologically into 60% training, 20% validation,
and 20% test. Do not shuffle. All final fitted preprocessors and model parameters use
training rows. Random Forest random search uses three expanding `TimeSeriesSplit` folds
within training; each fold independently fits clipping and imputation. LSTM search and
early stopping use only the subsequent validation block. No model is selected using test
metrics. Final models are not refitted on validation, keeping the fitting window consistent.

## Models

| Model | Training target | Selection | Test-time information |
| :--- | :--- | :--- | :--- |
| Last-value persistence | None | None | Previous index level |
| ARIMA | Index level | Four nonseasonal orders, training AIC | State extended after each observed level |
| Random Forest | Next change | Baseline plus random search on expanding training folds | Lagged features and previous level |
| LSTM | Standardized next change | Baseline, random and Bayesian search | Overlapping historical feature windows |

The forest and LSTM forecast changes and add them to the previous observed level. This
allows forecasts to move beyond index levels seen in training. LSTM feature and target
scalers are fitted once on training and reused. Sequences include the last feature row
available for a forecast date and carry context across split boundaries. Every model is
scored on the same test dates, including the first test observation.

ARIMA parameters stay fixed after training; validation observations initialize its state
for testing. Each actual test level becomes available only after its own forecast. The
forest and LSTM likewise use prior realized test observations in later lagged features.
This is rolling one-step evaluation, not a fixed-origin multistep projection.

## Metrics and interpretation

MAE and RMSE are measured in original index points. MSE and R2 are also exported.
MSE skill is `1 - model_MSE / naive_MSE`: positive values beat last-value persistence.
A high R2 on persistent index levels can coexist with poor forecasting skill. These metrics
do not measure investment returns, transaction costs, or statistically significant superiority.
Search does not guarantee improvement; baseline and tuned models are both reported.
Impurity feature importance can favor correlated or high-variance predictors and is not causal.

The synthetic dataset intentionally contains a lagged signal; its results demonstrate
execution and evaluation only. Generalization claims need licensed, release-aware data,
several temporal windows, repeated seeds, and uncertainty estimates. This repository
currently provides one chronological holdout per run.

## References

- [LSEG Data Library setup](https://developers.lseg.com/en/api-catalog/lseg-data-platform/lseg-data-library-for-python/quick-start/getting-started-with-python)
- [LSEG historical data access](https://developers.lseg.com/en/api-catalog/lseg-data-platform/lseg-data-library-for-python/tutorials/access-tutorials/tutorial-4-get_history)
- [scikit-learn preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [statsmodels ARIMA state updates](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMAResults.append.html)
- [Keras Tuner Bayesian optimization](https://keras.io/keras_tuner/api/tuners/bayesian/)
