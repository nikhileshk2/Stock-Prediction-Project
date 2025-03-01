Stock Price Prediction with Hybrid LSTM Model

This project uses a Hybrid LSTM (Long Short-Term Memory) model to predict the daily closing prices of the AAPL (Apple Inc.) stock, integrating technical indicators and macroeconomic data. The code fetches historical stock data, computes technical indicators, prepares the dataset, trains the model, and visualizes the results.


Overview

Data Source: Daily AAPL stock data from Yahoo Finance (March 2, 2022 - January 1, 2025).

Technical Indicators: Includes Moving Averages, RSI, MACD, Bollinger Bands, ATR, etc., calculated using talib.

Macroeconomic Data: Quarterly data (e.g., GDP Growth, Inflation) from a CSV file (commented out in this version).

Model: Hybrid LSTM built with Keras, trained to predict closing prices.

Evaluation: R² score and Mean Absolute Error (MAE) for model performance.


Steps

1) Data Retrieval: Downloads AAPL stock data using finance.

2) Feature Engineering:

Computes technical indicators with a custom calculate_technical_indicators function.

Calculates daily returns and target variable (next day's returns).

Combines with external AAPL data from a CSV file.

3) Data Preprocessing:

Scales features and target using MinMaxScaler.

Creates 20-day sequences for LSTM input.

Splits data into 80% training and 20% testing sets.

4) Model Training:
   
Builds a Hybrid LSTM model with 2 layers and 50 units per layer.

Trains the model for 50 epochs with a batch size of 32.

5) Evaluation & Visualization:
   
Computes test loss, MAE, and R² score.

Plots actual vs. predicted closing prices using matplotlib.


Requirements

Python 3.x

Libraries: yfinance, pandas, numpy, talib, sklearn, tensorflow, matplotlib


Files

technicalIndicators.py: Custom function to calculate technical indicators.

HybridLSTMmodel.py / LSTMModel.py: Defines the Hybrid LSTM model architecture.

AAPL Data (2).csv: External dataset with additional AAPL features.

(Commented) macrodata.csv: Quarterly macroeconomic data (not used in this run).


Usage

Install dependencies: pip install -r requirements.txt.

Run the script: python stock_prediction.py.

Output:

Prints data samples, model summary, and performance metrics.

Saves integrated data to nifty_macro_technical_integrated_2015_2025.csv.

Displays a plot of actual vs. predicted prices.


Notes

The macroeconomic data integration is currently commented out but can be enabled by uncommenting Steps 3-6.

Adjust start_date, end_date, or seq_length as needed for different timeframes or prediction horizons.
