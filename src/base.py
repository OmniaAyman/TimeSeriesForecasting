# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
from dotenv import load_dotenv
from darts.models import ExponentialSmoothing, ARIMA, Prophet
from darts.utils.utils import ModelMode
from darts.utils.utils import SeasonalityMode
from darts import TimeSeries
import matplotlib.pyplot as plt
from darts.metrics import mape, rmse, mae
import pandas as pd


# from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from xgboost import XGBRegressor
# # Super fast statistical models (AutoARIMA handles optimization natively)
# from statsforecast.models import AutoARIMA, HoltWinters
# # Cutting edge Neural network forecasting models
# from neuralforecast.models import NBEATS, NHITS, LSTM

load_dotenv()  # Load environment variables from .env file
# Set the path to the file you'd like to load
file_path = "sales.csv"

# Load the latest version
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "sudipmanchare/simulated-sales-data-with-timeseries-features",
    file_path,
    # Provide any additional arguments like
    # sql_query or pandas_kwargs. See the
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())


def predict_sales_data_darts():
    series = TimeSeries.from_dataframe(df, value_cols="Sales")
    model = ExponentialSmoothing(
        trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE
    )
    model.fit(series)
    pred = model.predict(6)
    print(pred.values())
    return pred


def predict_sales_data_skforecast():
    # You wrap a machine learning model to make it handle time series step-by-step
    forecaster = ForecasterAutoreg(
        regressor=XGBRegressor(random_state=123),
        lags=24,  # Use the past 24 time steps to predict the next ones
    )
    return forecaster


def predict_sales_data_neuralforecast():
    return NBEATS()


def validate_model(series, forecast_horizon=6):
    train, test = series[:-forecast_horizon], series[-forecast_horizon:]

    model = ExponentialSmoothing(
        trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE
    )
    model.fit(train)
    pred = model.predict(forecast_horizon)

    print("MAPE:", mape(test, pred))  # % error, lower is better
    print("RMSE:", rmse(test, pred))  # same units as Sales
    print("MAE:", mae(test, pred))  # same units as Sales, less sensitive to outliers

    train.plot(label="train")
    test.plot(label="actual")
    pred.plot(label="forecast")
    plt.legend()
    plt.show()
    return pred, test


def predict_sales_data_darts(df, forecast_horizon=6):
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    series = TimeSeries.from_dataframe(df, time_col="Date", value_cols="Sales")

    # Hold back the last `forecast_horizon` points as ground truth
    train = series[:-forecast_horizon]
    actual = series[-forecast_horizon:]

    model = ExponentialSmoothing(
        trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.MULTIPLICATIVE
    )
    model.fit(train)
    pred = model.predict(forecast_horizon)

    # Build a comparison table
    comparison = pd.DataFrame(
        {
            "Date": actual.time_index,
            "Actual": actual.values().flatten(),
            "Predicted": pred.values().flatten(),
        }
    )
    comparison["Error"] = comparison["Actual"] - comparison["Predicted"]
    comparison["% Error"] = (comparison["Error"] / comparison["Actual"]) * 100

    print(comparison.to_string(index=False))

    print("\nMAPE:", mape(actual, pred))
    print("RMSE:", rmse(actual, pred))
    print("MAE:", mae(actual, pred))

    return comparison


if __name__ == "__main__":
    # print("predict_sales_data_darts():", predict_sales_data_darts())
    # validate_model(TimeSeries.from_dataframe(df, value_cols='Sales'))
    predict_sales_data_darts(df)

    # print("predict_sales_data_skforecast():", predict_sales_data_skforecast())
    # print("predict_sales_data_neuralforecast():", predict_sales_data_neuralforecast())
