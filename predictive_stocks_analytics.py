# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title of the Web App
st.title("Stock Price Prediction with ARIMA and LSTM")

# Sidebar for user input
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL")
prediction_method = st.sidebar.selectbox("Choose Prediction Method", ["ARIMA", "LSTM"])

# Function to fetch stock data from Yahoo Finance
@st.cache
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="5y")  # Last 5 years of data
    data = data[['Close']]  # We are only interested in 'Close' prices
    return data

# Fetch the stock data
data = load_stock_data(ticker)

# Show the data to the user
st.subheader(f"Stock Data for {ticker}")
st.write(data.tail())

# Normalize data for LSTM (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['Close']].values)

# Plot the stock data
st.subheader("Stock Price Visualization")
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label=f'{ticker} Close Price')
plt.title(f'{ticker} Stock Price')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot()

# ARIMA Model
def arima_forecast(data):
    st.subheader("ARIMA Model Prediction")

    # ARIMA requires data to be stationary. We will skip the stationarity check for simplicity.
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)  # Forecast next 30 days

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data, label="Actual Data")
    plt.plot(pd.date_range(data.index[-1], periods=30, freq='B'), forecast, label="Forecast", color='red')
    plt.title("ARIMA Stock Price Prediction")
    plt.legend()
    st.pyplot()

    # Return forecast
    return forecast

# LSTM Model
def lstm_forecast(data):
    st.subheader("LSTM Model Prediction")

    # Prepare the data for LSTM (create sequences)
    def create_lstm_data(data, window_size=60):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Create sequences and split into train/test
    X, y = create_lstm_data(scaled_data)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]

    # Reshape the data for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))  # Predict the next value

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    # Predicting and reversing the scaling
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label="True Price")
    plt.plot(data.index[-len(y_test):], predicted_price, label="Predicted Price", color='red')
    plt.title(f'LSTM Stock Price Prediction for {ticker}')
    plt.legend()
    st.pyplot()

    # Return predicted prices
    return predicted_price

# Prediction Based on User Choice
if prediction_method == "ARIMA":
    forecast = arima_forecast(data['Close'])
elif prediction_method == "LSTM":
    forecast = lstm_forecast(data['Close'])

# Model Evaluation Metrics (useful for both methods)
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

# If the method is ARIMA, forecast is already returned; for LSTM, use predicted_price
if prediction_method == "ARIMA":
    st.write("ARIMA Model Evaluation Metrics")
    mae, rmse = calculate_metrics(data['Close'].tail(30).values, forecast)
elif prediction_method == "LSTM":
    st.write("LSTM Model Evaluation Metrics")
    mae, rmse = calculate_metrics(data['Close'].tail(len(forecast)).values, forecast)

st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#pip install --upgrade pip
#pip install --upgrade typing_extensions
#python setup.py build_ext
#pip install numpy==1.19.2
#pip install pandas
#pip install yfinance
#pip install matplotlib
#pip install seaborn
#pip install sklearn
#pip install statsmodels
#pip install tensorflow-cpu
#pip install streamlit==0.88.0

#streamlit run predictive_stocks_analytics.py

