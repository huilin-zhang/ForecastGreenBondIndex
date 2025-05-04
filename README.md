# Forecast Green Bond Index

## Project Description
This project aims to forecast the Green Bond Index using machine learning techniques. Green bonds are fixed-income securities designed to finance projects with positive environmental benefits. A Green Bond Index is designed to reflect the performance of the green bond market.

## Data Sources
1. **S&P Green Bond Index**: Data manually downloaded from [S&P Global](https://www.spglobal.com/spdji/en/indices/sustainability/sp-green-bond-index/#overview).
2. **LSEG Data Library**: Data accessed via API from the London Stock Exchange Group (LSEG).

## Methodology
- **Machine Learning**: Various machine learning models will be employed to forecast the Green Bond Index based on historical data and other relevant financial indicators. The specific methods include:
- **ARIMA (AutoRegressive Integrated Moving Average)**: A statistical time series forecasting method that models the dependencies between an observation and its lagged values to predict future trends.
- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) suitable for sequence prediction problems.
- **Random Forest**: This model captures non-linear relationships and interactions between variables.
