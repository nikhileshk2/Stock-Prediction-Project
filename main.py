import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from technicalIndicators import calculate_technical_indicators
from HybridLSTMmodel import HybridLSTM
from LSTMModel import build_hybrid_lstm_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Step 1: Pull daily NIFTY 50 data from Yahoo Finance
ticker = "AAPL"
start_date = "2022-03-02"
end_date = "2025-01-01"  # Current date: March 1, 2025

nifty_data = yf.download(ticker, start=start_date, end=end_date)
nifty_data = nifty_data[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
print("NIFTY 50 Data Sample:")
print(nifty_data.head())

# Step 2: Calculate technical indicators
df = pd.read_csv('AAPL Data (2).csv')
nifty_data = calculate_technical_indicators(nifty_data)
nifty_data['Returns'] = (nifty_data['Close']-nifty_data['Open'])/nifty_data['Open']
nifty_data['Target'] = nifty_data['Returns'].shift(-1)
nifty_data = pd.concat([nifty_data, df], axis=1)
# Drop rows with NaN values from technical indicators (due to lookback periods)
nifty_data = nifty_data.dropna().reset_index(drop=True)

"""# Step 3: Load quarterly macroeconomic data from CSV
macro_data = pd.read_csv("macrodata.csv")
macro_data['Quarter Start'] = pd.to_datetime(macro_data['Quarter'].apply(lambda x: f"{x.split()[0]}-{int(x.split()[1][1])*3-2:02d}-01"))
print("\nMacroeconomic Data Sample:")
print(macro_data.head())"""

"""# Step 4: Depreciation function with weighting factor
def depreciate_value(start_value, end_value, days_in_quarter, day_index, weight=0.1):
   #Linearly depreciate a value and apply a weighting factor to reduce influence.
    if day_index >= days_in_quarter:
        return end_value * weight
    raw_value = start_value + (end_value - start_value) * (day_index / days_in_quarter)
    return raw_value * weight  # Scale down by weight (e.g., 0.1)

# Step 5: Integrate and depreciate macroeconomic data with daily stock data
integrated_data = nifty_data.copy()
macro_columns = ['GDP_Growth_Rate', 'Inflation_Rate', 'Interest_Rate', 'Unemployment_Rate', 
                 'Consumer_Confidence_Index', 'Industrial_Production_Index']

for col in macro_columns:
    integrated_data[col] = np.nan

weight_factor = 0.1  # Reduce macroeconomic influence to 10% relative to technical indicators
for i in range(len(macro_data) - 1):
    start_date = macro_data['Quarter Start'].iloc[i]
    end_date = macro_data['Quarter Start'].iloc[i + 1]
    next_values = macro_data.iloc[i + 1]
    
    # Filter daily data for this quarter
    mask = (integrated_data['Date'] >= start_date) & (integrated_data['Date'] < end_date)
    quarter_data = integrated_data.loc[mask]
    
    # Number of days in this quarter
    days_in_quarter = (end_date - start_date).days
    
    for col in macro_columns:
        start_value = macro_data[col].iloc[i]
        end_value = next_values[col]
        # Apply depreciation with weight factor
        integrated_data.loc[mask, col] = [
            depreciate_value(start_value, end_value, days_in_quarter, (row['Date'] - start_date).days, weight_factor)
            for _, row in quarter_data.iterrows()
        ]

# Handle the last quarter (Q4 2024 to March 1, 2025)
last_quarter_start = macro_data['Quarter Start'].iloc[-1]
last_quarter_data = integrated_data[integrated_data['Date'] >= last_quarter_start]
days_to_end = (pd.to_datetime(end_date) - last_quarter_start).days

for col in macro_columns:
    start_value = macro_data[col].iloc[-1]
    end_value = start_value * 0.98  # Slight depreciation to hypothetical Q1 2025
    integrated_data.loc[integrated_data['Date'] >= last_quarter_start, col] = [
        depreciate_value(start_value, end_value, days_to_end, (row['Date'] - last_quarter_start).days, weight_factor)
        for _, row in last_quarter_data.iterrows()
    ]

# Drop any remaining NaN values
integrated_data = integrated_data.dropna()

# Step 6: Save the integrated data to a new CSV file
output_file = "nifty_macro_technical_integrated_2015_2025.csv"
integrated_data.to_csv(output_file, index=False)
print(f"\nIntegrated data saved to {output_file}")

# Verify the data
print("\nSample of Integrated Data:")
print(integrated_data.head())
print("\nLast Few Rows of Integrated Data:")
print(integrated_data.tail())"""

nifty_data.columns = ['_'.join(col).strip() for col in nifty_data.columns.values]
print(nifty_data.columns)
# Step 7: Prepare for LSTM - Scaling and Sequence Creation
scaler = MinMaxScaler(feature_range=(0, 1))
features = nifty_data[['Open_AAPL', 'High_AAPL', 'Low_AAPL',
       'Volume_AAPL', 'MA5_', 'MA10_', 'ADX_', '+DI14_', '-DI14_', 'DIF_',
       'DEA_', 'MACD_', 'RSI5_', 'RSI14_', '%K_', '%D_', '%R_', 'ATR14_',
       'BB_Upper_', 'BB_Middle_', 'BB_Lower_', 'STD20_',
       'G_D_P___G_r_o_w_t_h___R_a_t_e', 'I_n_f_l_a_t_i_o_n___R_a_t_e',
       'U_n_e_m_p_l_o_y_m_e_n_t___R_a_t_e',
       'I_n_d_u_s_t_r_i_a_l___P_r_o_d_u_c_t_i_o_n___I_n_d_e_x_ _(_%_)']]
target = nifty_data['Close_AAPL']

scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

# Create sequences for LSTM (e.g., 20-day lookback)
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 20
X, y = create_sequences(scaled_features, scaled_target, seq_length)

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nX_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Step 8: Build and Train the Hybrid LSTM Model
# Define input shape for the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_length, num_features)

# Build the model
lstm_model = build_hybrid_lstm_model(input_shape, num_layers=2, units=50)

# Print model summary
lstm_model.summary()

#Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1)
test_loss, test_mae = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")


# Make predictions
y_pred = lstm_model.predict(X_test)

# Calculate R²
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")


# Inverse transform the scaled predictions and actual values
y_pred_actual = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, label='Actual Closing Price')
plt.plot(y_pred_actual, label='Predicted Closing Price')
plt.title('Actual vs Predicted Closing Price')
plt.xlabel('Time')
plt.ylabel('Returns')
plt.legend()
plt.show()