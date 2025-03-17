import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    # Ensure both are Series
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0]  # select first column (or set your target column)
        
    # Handle numpy array case for y_pred
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)
    elif isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0]  # just in case y_pred is a dataframe too

    # Reset index to make sure both Series align
    y_true = y_true.reset_index(drop=True)
    y_pred = y_pred.reset_index(drop=True)

    # Mask to avoid division by zero
    mask = y_true != 0

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return mae, rmse, mape


st.title("📊 Time Series Forecasting Dashboard")

#Data Retrieval
st.header("Data Retrieval")
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL):", "AAPL")
if st.button("Fetch Data"):
    try:
        data = yf.download(symbol, period="5y", interval="1d")
        if not data.empty:
            data.dropna(inplace=True)
            data.drop_duplicates(inplace=True)
            data.index = pd.to_datetime(data.index)
            data.index = data.index.tz_localize(None)
            st.session_state['data'] = data  # Save to session state
            st.success(f"Data for {symbol} fetched successfully.")
            st.write(data.tail())
        else:
            st.warning("No data found for this symbol.")
    except Exception as e:
        st.error(f"Error fetching data: {e}")

#Data Cleaning and Preprocessing
if 'data' in st.session_state:
    data = st.session_state['data']

    # Preprocessing Section
    with st.expander("🔄 Data Cleaning and Preprocessing", expanded=False):
        st.write(f"Cleaned Data Shape: {data.shape}")
        st.write(data.head())

    # EDA Section
    with st.expander("📊 Exploratory Data Analysis", expanded=False):
        st.subheader("Line Plot of Close Prices")
        st.line_chart(data['Close'])

        st.subheader("Heatmap (Monthly Average Prices)")
        data['Month'] = data.index.month
        data['Year'] = data.index.year
        pivot_table = data.pivot_table(values='Close', index='Month', columns='Year', aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Autocorrelation Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(data['Close'].dropna(), lags=50, ax=ax)
        ax.set_title("Autocorrelation Function (ACF)")
        st.pyplot(fig)
        plt.clf()

        st.subheader("ADF Test (Stationarity Check)")
        adf_result = adfuller(data['Close'])
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"p-value: {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            st.success("The data is stationary (p < 0.05)")
        else:
            st.warning("The data is non-stationary (p >= 0.05)")

    # Modeling Section
    st.header("⚙ Model Building and Evaluation")
    model_type = st.selectbox("Choose Model", ["ARIMA", "SARIMA", "Random Forest", "XGBoost","LSTM"])
    split_ratio = st.slider("Train/Test Split", 0.7, 0.95, 0.8)
    split_index = int(len(data) * split_ratio)
    train, test = data['Close'][:split_index], data['Close'][split_index:]
    forecast = None
    mae = rmse = mape = None

    # Model Training
    if model_type == "ARIMA":
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        st.success("ARIMA model trained")

    elif model_type == "SARIMA":
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        st.success("SARIMA model trained")

    elif model_type == "Random Forest":
        df_feat = data.copy()
        df_feat['lag1'] = df_feat['Close'].shift(1)
        df_feat['lag7'] = df_feat['Close'].shift(7)
        df_feat.dropna(inplace=True)
        X = df_feat[['lag1', 'lag7']]
        y = df_feat['Close']
        X_train, X_test = X.iloc[:split_index-7], X.iloc[split_index-7:]
        y_train, y_test = y.iloc[:split_index-7], y.iloc[split_index-7:]
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        st.success("Random Forest model trained")

    elif model_type == "XGBoost":
        df_feat = data.copy()
        df_feat['lag1'] = df_feat['Close'].shift(1)
        df_feat['lag7'] = df_feat['Close'].shift(7)
        df_feat.dropna(inplace=True)
        X = df_feat[['lag1', 'lag7']]
        y = df_feat['Close']
        X_train, X_test = X.iloc[:split_index-7], X.iloc[split_index-7:]
        y_train, y_test = y.iloc[:split_index-7], y.iloc[split_index-7:]
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        forecast = model.predict(X_test)
        st.success("XGBoost model trained")

    elif model_type == "LSTM":
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[['Close']])

        X_lstm, y_lstm = [], []
        for i in range(60, len(scaled)):
            X_lstm.append(scaled[i-60:i, 0])
            y_lstm.append(scaled[i, 0])

        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
        X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

        split_lstm = int(split_ratio * len(X_lstm))
        X_train, X_test = X_lstm[:split_lstm], X_lstm[split_lstm:]
        y_train, y_test = y_lstm[:split_lstm], y_lstm[split_lstm:]

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        pred = model.predict(X_test)
        forecast = scaler.inverse_transform(pred).flatten()
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        mae_lstm = mean_absolute_error(y_test_inv, forecast)
        rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, forecast))
        mape_lstm = np.mean(np.abs((y_test_inv - forecast) / y_test_inv)) * 100
        st.success("LSTM model trained")

    # Metrics
    if forecast is not None:
        if model_type in ["ARIMA", "SARIMA"]:
            mae, rmse, mape = calculate_metrics(test, forecast)
        elif model_type == 'LSTM':
            mae, rmse, mape = mae_lstm, rmse_lstm, mape_lstm
        else:
            mae, rmse, mape = calculate_metrics(y_test, forecast)

        st.subheader("Evaluation Metrics")
        st.write(f"*MAE:* {mae:.2f}")
        st.write(f"*RMSE:* {rmse:.2f}")
        st.write(f"*MAPE:* {mape:.2f}%")


    # Forecast Visualization
    st.header("Forecasting Visualization")
    plt.figure(figsize=(14, 6))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')

    if len(forecast) != len(test):
        # Auto-adjust the index slice to match the forecast length
        adjusted_index = test.index[-len(forecast):]
    else:
        adjusted_index = test.index

    plt.plot(adjusted_index, forecast, label=f"{model_type} Forecast")
    plt.legend()
    st.pyplot(plt.gcf())
