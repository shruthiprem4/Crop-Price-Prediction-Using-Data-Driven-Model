import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import matplotlib.dates as mdates
from functools import lru_cache
import io
import base64

from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

def load_and_preprocess_data():
    df = pd.read_csv("data1.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], format='%d-%m-%Y', errors='coerce')
    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df.dropna(subset=['arrival_date', 'modal_price'], inplace=True)
    df = df.groupby(['arrival_date', 'commodity', 'state'], as_index=False).agg({'modal_price': 'mean'})
    return df.sort_values('arrival_date')

def split_data(series, train_ratio=0.8):
    train_size = int(len(series) * train_ratio)
    return series.iloc[:train_size], series.iloc[train_size:]

def evaluate_model(y_actual, y_pred):
    y_actual, y_pred = np.array(y_actual), np.array(y_pred)
    min_length = min(len(y_actual), len(y_pred))
    y_actual, y_pred = y_actual[:min_length], y_pred[:min_length]
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    return mae, mse, rmse, mape

@lru_cache(maxsize=None)
def train_arima(train_series_tuple, steps_ahead):
    train_series = pd.Series(train_series_tuple)
    model = ARIMA(train_series, order=(2, 1, 2)).fit()
    return model.forecast(steps=steps_ahead)

@lru_cache(maxsize=None)
def train_auto_arima(train_series_tuple, steps_ahead):
    train_series = pd.Series(train_series_tuple)
    model = auto_arima(train_series, seasonal=False, trace=False)
    return model.predict(n_periods=steps_ahead)

@lru_cache(maxsize=None)
def train_sarima(train_series_tuple, steps_ahead):
    train_series = pd.Series(train_series_tuple)
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
    return model.forecast(steps=steps_ahead)

@tf.function()
def lstm_predict(model, input_tensor):
    return model(input_tensor, training=False)

def train_lstm(train_series, steps_ahead):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(6, len(scaled_data)):
        X_train.append(scaled_data[i-6:i])
        y_train.append(scaled_data[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    model_path = "lstm_model.keras"

    try:
        model = load_model(model_path)
    except Exception:
        model = Sequential([
            Bidirectional(LSTM(100, activation='relu', return_sequences=True, input_shape=(6, 1))),
            Dropout(0.3),
            LSTM(100, activation='relu'),
            Dropout(0.3),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
        model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0)
        model.save(model_path)

    last_inputs = scaled_data[-6:]
    lstm_predictions = []
    for _ in range(steps_ahead):
        pred = lstm_predict(model, last_inputs.reshape(1, 6, 1)).numpy()[0][0]
        lstm_predictions.append(pred)
        last_inputs = np.append(last_inputs[1:], pred).reshape(6, 1)

    return scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()

def train_hybrid_model(train_series, steps_ahead):
    train_tuple = tuple(train_series.values)
    arima_preds = train_arima(train_tuple, steps_ahead)
    lstm_preds = train_lstm(train_series, steps_ahead)

    arima_mape = evaluate_model(train_series[-steps_ahead:], arima_preds[:steps_ahead])[3]
    lstm_mape = evaluate_model(train_series[-steps_ahead:], lstm_preds[:steps_ahead])[3]

    total_mape = arima_mape + lstm_mape
    arima_weight = 1 - (arima_mape / total_mape)
    lstm_weight = 1 - (lstm_mape / total_mape)

    return arima_weight * np.array(arima_preds) + lstm_weight * np.array(lstm_preds)

def predict_crop_price(df, commodity, state, steps_ahead, time_unit):
    # Filter the dataset for the given commodity and state
    df_filtered = df[(df['commodity'].str.lower() == commodity.lower()) &
                     (df['state'].str.lower() == state.lower())]

    # Check if data is available
    if df_filtered.empty:
        print("No data available.")
        return None, None, None, None, None

    # Set the index as 'arrival_date' and extract 'modal_price'
    df_filtered.set_index('arrival_date', inplace=True)
    price_series = df_filtered['modal_price']

    # Check if there is enough data for training
    if len(price_series) < 10:
        print("Not enough data.")
        return None, None, None, None, None

    # Get last date in dataset
    last_dataset_date = price_series.index[-1]
    current_date = datetime.now().date()
    current_date = pd.to_datetime(current_date)

    # Calculate gap between last dataset date and current date
    if time_unit == "month":
        # Approximate number of months between dates
        gap_size = ((current_date.year - last_dataset_date.year) * 12 +
                    current_date.month - last_dataset_date.month)
    else:  # time_unit == "week"
        # Approximate number of weeks between dates
        gap_size = (current_date - last_dataset_date).days // 7

    # Ensure gap_size is at least 1
    gap_size = max(1, gap_size)

    # Split data into training and testing
    train, test = split_data(price_series)
    train_tuple = tuple(train.values)

    # Store original steps for plotting
    original_steps = steps_ahead

    # Convert steps_ahead based on the time unit (week or month)
    if time_unit == "month":
        modeling_steps = steps_ahead * 4  # Convert months to weeks for modeling
    elif time_unit == "week":
        modeling_steps = steps_ahead  # No change
    else:
        raise ValueError("Invalid time unit. Choose either 'week' or 'month'.")

    # Add the gap size to the modeling steps to predict from last dataset date to current date + steps_ahead
    total_modeling_steps = modeling_steps + gap_size

    # Make sure we have at least one step to predict
    total_modeling_steps = max(1, total_modeling_steps)

    # Train different models for crop price prediction with extended prediction horizon
    try:
        models = {
            "ARIMA": train_arima(train_tuple, total_modeling_steps),
            "Auto-ARIMA": train_auto_arima(train_tuple, total_modeling_steps),
            "SARIMA": train_sarima(train_tuple, total_modeling_steps),
            "LSTM": train_lstm(train, total_modeling_steps),
            "Hybrid": train_hybrid_model(train, total_modeling_steps)
        }
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None, None, None, None

    # Adjust first prediction point to match last known value
    for model_name in models:
        if models[model_name] is not None and len(models[model_name]) > 0:
            if isinstance(models[model_name], pd.Series):
                models[model_name].iloc[0] = price_series.iloc[-1]
            else:
                models[model_name][0] = price_series.iloc[-1]

    # Ensure the test dataset has enough values
    if len(test) < modeling_steps:
        print(f"Warning: Test data has only {len(test)} points, but {modeling_steps} are required!")
        test_size = len(test)
    else:
        test_size = modeling_steps

    # Compute model evaluation metrics safely
    metrics = {}
    for model_name, predictions in models.items():
        if predictions is not None and len(predictions) > 0:
            if isinstance(predictions, pd.Series):
                predictions = predictions.values
            predictions = predictions[:test_size]  # Truncate to valid length
            test_values = test[:test_size].values
            try:
                metrics[model_name] = evaluate_model(test_values, predictions)
            except Exception as e:
                print(f"Warning: Could not evaluate {model_name}: {e}")
        else:
            print(f"Warning: {model_name} did not generate predictions.")

    # Find the best model (based on the lowest MAPE)
    best_model = min(metrics, key=lambda k: metrics[k][3]) if metrics else None

    return models, best_model, price_series, original_steps, gap_size

def plot_results_for_web(models, best_model, price_series, steps_ahead, time_unit, gap_size=0):
    """Modified plot_results function that returns base64 encoded plot images for web display"""
    # Last date in the dataset
    last_dataset_date = price_series.index[-1]

    # Current date from system
    current_date = datetime.now().date()
    current_date = pd.to_datetime(current_date)

    # Create future dates from last dataset date
    future_dates_from_dataset = []
    if time_unit == 'week':
        for i in range(1, gap_size + steps_ahead + 1):
            future_dates_from_dataset.append(last_dataset_date + pd.DateOffset(weeks=i))
    else:  # month
        for i in range(1, gap_size + steps_ahead + 1):
            future_dates_from_dataset.append(last_dataset_date + pd.DateOffset(months=i))

    # Create future dates from current date
    future_dates_from_current = []
    if time_unit == 'week':
        for i in range(1, steps_ahead + 1):
            future_dates_from_current.append(current_date + pd.DateOffset(weeks=i))
    else:  # month
        for i in range(1, steps_ahead + 1):
            future_dates_from_current.append(current_date + pd.DateOffset(months=i))

    # Plot 1: All model predictions
    plt.figure(figsize=(12, 6))

    # Plot each model's predictions without padding
    for model_name, predictions in models.items():
        if predictions is None or len(predictions) == 0:
            continue

        if isinstance(predictions, pd.Series):
            values = predictions.values / 100  # Convert to per kg
        else:
            values = np.array(predictions) / 100  # Convert to per kg

        # Determine how many valid predictions we have
        if time_unit == 'month' and len(values) > steps_ahead:
            # For monthly predictions from weekly data, take appropriate values
            values_to_plot = []
            month_ends = []

            # Get actual month-end values for more accurate monthly predictions
            current_month = current_date.month
            current_year = current_date.year

            for i in range(steps_ahead):
                next_month = current_month + i
                year_offset = (next_month - 1) // 12
                month_num = ((next_month - 1) % 12) + 1

                month_end = pd.Timestamp(current_year + year_offset, month_num,
                                         pd.Timestamp(current_year + year_offset, month_num, 1).days_in_month)
                month_ends.append(month_end)

            # Find the closest weekly prediction to each month end
            weekly_dates = [current_date + pd.DateOffset(weeks=i) for i in range(len(values))]

            for month_end in month_ends:
                # Find closest weekly prediction to this month end
                closest_idx = min(range(len(weekly_dates)),
                                 key=lambda i: abs((weekly_dates[i] - month_end).total_seconds()))

                # Only use the value if it's within the available predictions
                if closest_idx < len(values):
                    values_to_plot.append(values[closest_idx])
                else:
                    break
        else:
            # For weekly predictions, use directly but limit to valid predictions
            values_to_plot = values[:min(steps_ahead, len(values))]

        # Only plot available predictions (no padding)
        valid_dates = future_dates_from_dataset[:min(len(values_to_plot), len(future_dates_from_dataset))]
        plt.plot(valid_dates, values_to_plot, marker='o', label=model_name)

    plt.legend()
    plt.title(f"Price Predictions by All Models ({time_unit.capitalize()}ly for {steps_ahead} {time_unit}s)")
    plt.xlabel("Future Dates")
    plt.ylabel("Price per kg (₹)")

    # Format x-axis dates based on time unit
    if time_unit == 'week':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    else:  # month
        if steps_ahead <= 12:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month

    plt.gcf().autofmt_xdate()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot to a buffer and convert to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot1_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Plot 2: Actual vs Best Model Prediction
    plt.figure(figsize=(12, 6))

    # Get historical data
    if time_unit == 'week':
        start_date = price_series.index[-1] - pd.DateOffset(months=10)
    else:  # month
        start_date = price_series.index[-1] - pd.DateOffset(years=3)

    historical_prices = price_series[price_series.index >= start_date]

    # Plot historical prices
    plt.plot(historical_prices.index, historical_prices / 100, marker='o', label="Actual Prices", color='blue')

    # Plot best model predictions
    if best_model and models[best_model] is not None and len(models[best_model]) > 0:
        best_predictions = models[best_model]
        if isinstance(best_predictions, pd.Series):
            best_values = best_predictions.values / 100
        else:
            best_values = np.array(best_predictions) / 100

        # Determine appropriate values to plot
        valid_length = min(gap_size + steps_ahead, len(best_values))
        best_model_values = best_values[:valid_length]
        valid_dates = future_dates_from_dataset[:min(len(best_model_values), len(future_dates_from_dataset))]

        # Plot predictions with dashed line
        plt.plot(valid_dates, best_model_values, marker='o', linestyle='dashed',
                 label=f"Predicted Prices ({best_model})", color='red')

        # Highlight the current date with a vertical line
        plt.axvline(x=current_date, color='green', linestyle='--', alpha=0.7, label="Current Date")

        # Highlight where predictions from current date begin
        current_date_idx = None
        for i, date in enumerate(valid_dates):
            if date >= current_date:
                current_date_idx = i
                break

        if current_date_idx is not None and current_date_idx < len(best_model_values):
            plt.plot([valid_dates[current_date_idx]], [best_model_values[current_date_idx]],
                     marker='*', markersize=10, color='green', label="Prediction From Current Date")

    plt.legend()
    plt.title(f"Actual vs. Predicted Prices ({best_model})")
    plt.xlabel("Date")
    plt.ylabel("Price per kg (₹)")

    # Format x-axis dates
    if time_unit == 'week':
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    else:  # month
        if steps_ahead <= 12:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3rd month

    plt.gcf().autofmt_xdate()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot to a buffer and convert to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot2_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return plot1_b64, plot2_b64
