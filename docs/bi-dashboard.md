# Interactive dashboard and BI integration

## Open the report

Open `examples/synthetic-demo/dashboard.html` in a web browser. The dashboard bundles
its chart runtime and experiment data, so it also works offline. On GitHub, download
the HTML file and open the downloaded copy; the repository file viewer displays source.
The README includes an actual browser screenshot for immediate preview.

Every new `greenbond demo`, `greenbond run`, or `greenbond legacy` experiment now produces
`dashboard.html` and `forecast_facts.csv` alongside the original report. Upgrade an existing
experiment without training again:

```bash
greenbond dashboard --report-dir examples/synthetic-demo
greenbond dashboard --report-dir reports/your-experiment
```

This command regenerates the dashboard and BI export from saved predictions, metadata,
and feature importance. Original forecasts and model metrics are preserved.

## Report interactions

- Model selection updates the KPI cards, forecasts, residuals, distribution, and rolling error.
- Date filters recompute metrics from the selected observations, including comparison bars
  and the scorecard. Quick windows select the last 30 or 90 observations, not calendar days.
- Selecting a model scorecard row focuses that model. Hover and zoom reveal individual values.
- Export filtered CSV downloads the selected model and date range with row-level errors.
- Print / Save PDF uses the browser print dialog and a dedicated print layout.
- Random Forest feature importance remains the fitted experiment's fixed importance;
  changing model or date filters does not retrain or change its values.

The scorecard is a descriptive view of test results. Filtering the display does not alter
model parameters or constitute a new validation experiment. Metadata always describe the
full original experiment; KPI cards describe the selected test period. R2 is undefined for
constant/single-observation windows, and skill is undefined when baseline MSE is zero.

## Data tables for Tableau and Power BI

| Artifact | Grain | Purpose |
| :--- | :--- | :--- |
| `forecast_facts.csv` | One date and model per row | Filtered forecasts, residuals, and dynamic metrics |
| `metrics.csv` | One model per full experiment | Fixed full-test performance summary |
| `feature_importance.csv` | One forest feature per row | Fixed importance chart |
| `run.json` | One experiment | Source, splits, parameters, and provenance |

`forecast_facts.csv` contains `date`, `model`, `actual`, `prediction`, `naive_prediction`,
`error`, `absolute_error`, `squared_error`, and `naive_squared_error`.

Import the fact CSV in either BI tool. Set `date` to Date, `model` to Text, and numeric fields
to Decimal. Use a single-model selector and a date-range selector for headline KPI cards.
For each date, average index values rather than summing across model rows. Compute filtered
error metrics from the facts; the preaggregated `metrics.csv` does not recalculate under
arbitrary date filters. The importance table is separate and has no date/model relationship.

## Power BI measure examples

Name the imported table `forecast_facts`. Add these measures, and format Skill as a percentage:

```dax
MAE = AVERAGE(forecast_facts[absolute_error])

MSE = AVERAGE(forecast_facts[squared_error])

RMSE = SQRT([MSE])

Naive MSE = AVERAGE(forecast_facts[naive_squared_error])

Skill = IF([Naive MSE] > 0, 1 - DIVIDE([MSE], [Naive MSE]), BLANK())

Observations = DISTINCTCOUNT(forecast_facts[date])
```

Use card visuals for RMSE, MAE, Skill, and Observations; line charts for observed/predicted
levels and residuals; a bar chart with model and RMSE for comparison; and a matrix with
model and the measures for the scorecard. See Microsoft's [DAX basics](https://learn.microsoft.com/en-us/power-bi/transform-model/desktop-quickstart-learn-dax-basics)
and [safe division guidance](https://learn.microsoft.com/en-us/dax/best-practices/dax-divide-function-operator).

## Tableau calculated field examples

```text
MAE: AVG([absolute_error])
MSE: AVG([squared_error])
RMSE: SQRT(AVG([squared_error]))
Skill:
IF AVG([naive_squared_error]) > 0 THEN
    1 - AVG([squared_error]) / AVG([naive_squared_error])
END
Observations: COUNTD([date])
```

Arrange KPI worksheets across the top, observed/predicted lines and a comparison bar chart
in the middle, and residuals and model scores below. Apply the model/date filters to the
relevant forecast worksheets. Keep the fixed feature-importance sheet outside those filters.
See Tableau's [calculated fields](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_create.htm)
and [aggregate calculations](https://help.tableau.com/current/pro/desktop/en-us/calculations_calculatedfields_aggregate_create.htm).

These measure templates have not been executed in a native Tableau or Power BI workbook.
The Python fact-table tests and browser checks verify the equivalent error calculations.

## Sharing

The committed dashboard uses synthetic data. A real-data dashboard embeds actual observations
and forecasts, so retain it in ignored `reports/` unless sharing is permitted by the data
provider. An offline file includes its data even if a chart is currently filtered.
For a hosted public preview, deploy only the synthetic file with the intended hosting service.
