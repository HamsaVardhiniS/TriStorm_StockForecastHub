# StockForecastHub

# Overview

StockForecastHub is a time-series forecasting project that retrieves historical stock price data via Yahoo Finance (yfinance), preprocesses it, and applies predictive models like ARIMA, SARIMA, XGBoost, and LSTM. The results are visualized through interactive dashboards for better insights.

# Features

1. Data Retrieval – Fetches up to 5 years of stock price data via API
2. Data Preprocessing – Cleans and formats timestamps, handles missing values
3. Exploratory Data Analysis (EDA) – Visualizes trends with interactive plots
4. Forecasting Models – Implements ARIMA, SARIMA, Random Forest, XGBoost, and LSTM
5. Performance Evaluation – Compares models using RMSE, MAE, and MAPE metrics
6. Web App Deployment – Provides an interactive UI using Streamlit

# Installation and Usage

#Install dependencies

pip install -r requirements.txt

#Running the Web App

streamlit run app.py


