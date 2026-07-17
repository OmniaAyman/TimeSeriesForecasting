# Time Series Forecasting Experiments: Sales Data

This project contains a Python script for experimenting with time series forecasting models on a simulated sales dataset. The primary implementation uses the `darts` library to perform Exponential Smoothing, while providing a framework to easily expand into machine learning and deep learning approaches using `skforecast` and `neuralforecast`.

## Features

- **Automated Data Loading**: Uses `kagglehub` to directly download and load the "Simulated Sales Data with Timeseries Features" dataset into a Pandas DataFrame.
- **Time Series Modeling**: Implements an `ExponentialSmoothing` model (with additive trend and multiplicative seasonality) using the `darts` library.
- **Model Evaluation**: Validates the model using a standard train/test split (holdout validation) and calculates key performance metrics:
  - **MAPE** (Mean Absolute Percentage Error)
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
- **Results Comparison**: Generates a detailed comparison table showing Actual vs. Predicted values along with absolute and percentage errors.
- **Visualization (Optional)**: Includes functions to plot the training data, actual holdout data, and forecasted values.

## Prerequisites

Ensure you have Python 3.8+ installed. The following libraries are required to run the active code:

```bash
pip install kagglehub[pandas-datasets] python-dotenv darts pandas matplotlib
```

*Optional dependencies (for the experimental/commented-out functions):*
```bash
pip install skforecast xgboost statsforecast neuralforecast
```

## Setup & Configuration

1. **Environment Variables**: The script utilizes `python-dotenv` to load environment variables from a `.env` file. 
2. **Kaggle Authentication**: Since the script uses `kagglehub` to fetch datasets, ensure you have your Kaggle API token configured (e.g., `~/.kaggle/kaggle.json`).

## Usage

To run the forecasting experiment, execute the main script:

```bash
python main.py
```
*(Note: Replace `main.py` with the actual name of your script file if different).*

Upon execution, the script will:
1. Fetch and load the latest version of the sales dataset.
2. Print the first 5 records of the dataset for validation.
3. Convert the dataframe into a `darts` `TimeSeries` object.
4. Train the `ExponentialSmoothing` model on the dataset, reserving the last 6 time steps as a holdout set.
5. Predict the sales for the next 6 time steps.
6. Print a side-by-side comparison table of actual vs. predicted values and display the evaluation metrics (MAPE, RMSE, MAE).

## Code Structure

- `predict_sales_data_darts(df, forecast_horizon=6)`: The primary active function. It formats the dataframe, trains the exponential smoothing model, generates predictions, and calculates evaluation metrics against the holdout set.
- `validate_model(series, forecast_horizon=6)`: A utility function designed to train, predict, evaluate, and plot the actual vs. forecasted data using matplotlib.
- `predict_sales_data_skforecast()` / `predict_sales_data_neuralforecast()`: Placeholder functions for future experimentation with machine learning regressors (e.g., XGBoost wrapped for time series) and deep learning models (e.g., NBEATS).

## Future Work & Enhancements

- **Model Expansion**: Complete the implementations for `skforecast` (XGBoost) and `neuralforecast` (NBEATS, LSTM) to enable multi-model comparisons.
- **Hyperparameter Tuning**: Introduce grid search or automated optimization for model parameters.
- **Advanced Forecasting**: Explore the inclusion of exogenous variables (covariates) to improve prediction accuracy.
README.md
Displaying README.md.# Time Series Forecasting Experiments: Sales Data

This project contains a Python script for experimenting with time series forecasting models on a simulated sales dataset. The primary implementation uses the `darts` library to perform Exponential Smoothing, while providing a framework to easily expand into machine learning and deep learning approaches using `skforecast` and `neuralforecast`.

## Features

- **Automated Data Loading**: Uses `kagglehub` to directly download and load the "Simulated Sales Data with Timeseries Features" dataset into a Pandas DataFrame.
- **Time Series Modeling**: Implements an `ExponentialSmoothing` model (with additive trend and multiplicative seasonality) using the `darts` library.
- **Model Evaluation**: Validates the model using a standard train/test split (holdout validation) and calculates key performance metrics:
  - **MAPE** (Mean Absolute Percentage Error)
  - **RMSE** (Root Mean Squared Error)
  - **MAE** (Mean Absolute Error)
- **Results Comparison**: Generates a detailed comparison table showing Actual vs. Predicted values along with absolute and percentage errors.
- **Visualization (Optional)**: Includes functions to plot the training data, actual holdout data, and forecasted values.

## Prerequisites

Ensure you have Python 3.8+ installed. The following libraries are required to run the active code:

```bash
pip install kagglehub[pandas-datasets] python-dotenv darts pandas matplotlib
```

*Optional dependencies (for the experimental/commented-out functions):*
```bash
pip install skforecast xgboost statsforecast neuralforecast
```

## Setup & Configuration

1. **Environment Variables**: The script utilizes `python-dotenv` to load environment variables from a `.env` file. 
2. **Kaggle Authentication**: Since the script uses `kagglehub` to fetch datasets, ensure you have your Kaggle API token configured (e.g., `~/.kaggle/kaggle.json`).

## Usage

To run the forecasting experiment, execute the main script:

```bash
python main.py
```
*(Note: Replace `main.py` with the actual name of your script file if different).*

Upon execution, the script will:
1. Fetch and load the latest version of the sales dataset.
2. Print the first 5 records of the dataset for validation.
3. Convert the dataframe into a `darts` `TimeSeries` object.
4. Train the `ExponentialSmoothing` model on the dataset, reserving the last 6 time steps as a holdout set.
5. Predict the sales for the next 6 time steps.
6. Print a side-by-side comparison table of actual vs. predicted values and display the evaluation metrics (MAPE, RMSE, MAE).

## Code Structure

- `predict_sales_data_darts(df, forecast_horizon=6)`: The primary active function. It formats the dataframe, trains the exponential smoothing model, generates predictions, and calculates evaluation metrics against the holdout set.
- `validate_model(series, forecast_horizon=6)`: A utility function designed to train, predict, evaluate, and plot the actual vs. forecasted data using matplotlib.
- `predict_sales_data_skforecast()` / `predict_sales_data_neuralforecast()`: Placeholder functions for future experimentation with machine learning regressors (e.g., XGBoost wrapped for time series) and deep learning models (e.g., NBEATS).

## Future Work & Enhancements

- **Model Expansion**: Complete the implementations for `skforecast` (XGBoost) and `neuralforecast` (NBEATS, LSTM) to enable multi-model comparisons.
- **Hyperparameter Tuning**: Introduce grid search or automated optimization for model parameters.
- **Advanced Forecasting**: Explore the inclusion of exogenous variables (covariates) to improve prediction accuracy.
