# Original notebook review

`greenbond.ipynb` is preserved as the original exploratory artifact, including the user's
existing local edit. Its saved outputs are historical and should not be treated as validated
benchmarks. Use the package CLI or `notebooks/01_quickstart.ipynb` for the revised workflow.

Issues addressed in the package:

1. The original scaler was fitted separately to training, validation, and test features and
   targets. Separate fits make units inconsistent and expose held-out statistics. The new
   neural workflow reuses training-fitted feature and target transforms.
2. Same-day target rolling statistics and an auxiliary target z-score were included among
   predictors. New features explicitly stop before the forecasted observation.
3. Linear interpolation could use future predictor values. New alignment uses forward fill
   followed by training-fitted median imputation for remaining missing values.
4. Joining macro series by observation month could expose data before publication. The new
   legacy adapter applies an explicit availability lag and documents remaining revision bias.
5. Ordinary regression cross-validation could train on future data. Forest tuning now uses
   expanding chronological folds; neural tuning uses the chronological validation block.
6. Original ARIMA forecasts were fixed-origin multistep, while ML models had access to recent
   realized test observations. New ARIMA state updates provide comparable one-step forecasts.
7. LSTM windows dropped early test dates while other models scored them. The new workflow
   shares test dates and carries historical context across validation and test boundaries.
8. API downloads repeatedly fetched auxiliary series inside the per-instrument loop. The new
   extractor caches each request, records field fallbacks and failures, and closes its session.

## Claims from the project description

The description mentions 20% less data-processing time and 10% better prediction accuracy.
No matched timing benchmark or reproducible accuracy calculation was found supporting
these percentages. The original saved table also does not support a general tuning gain.
The README therefore describes implemented capabilities and publishes newly generated,
clearly identified synthetic demo results. To substantiate a claim, retain the baseline,
same input dataset, evaluation dates, metric definition, search budget, repeated runs,
and the calculation of relative improvement. Never substitute synthetic results for a
licensed-data forecasting result.
