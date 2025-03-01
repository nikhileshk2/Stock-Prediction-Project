import pandas as pd
import numpy as np
import talib

def calculate_technical_indicators(stock_df):
    # Ensure the DataFrame has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in stock_df.columns:
            raise ValueError(f"Column '{col}' is missing in the DataFrame.")
    
    # Convert DataFrame columns to NumPy arrays for TA-Lib
    close = stock_df['Close'].to_numpy()
    high = stock_df['High'].to_numpy()
    low = stock_df['Low'].to_numpy()
    volume = stock_df['Volume'].to_numpy()

    # Debugging: Check the shape of the input arrays
    print(f"Close shape: {close.shape}")
    print(f"High shape: {high.shape}")
    print(f"Low shape: {low.shape}")
    print(f"Volume shape: {volume.shape}")

    # Ensure the arrays are 1-dimensional
    if close.ndim != 1:
        close = close.flatten()
    if high.ndim != 1:
        high = high.flatten()
    if low.ndim != 1:
        low = low.flatten()
    if volume.ndim != 1:
        volume = volume.flatten()

    # Moving Averages
    stock_df['MA5'] = talib.SMA(close, timeperiod=5)
    stock_df['MA10'] = talib.SMA(close, timeperiod=10)

    # Average Directional Index (ADX) and Directional Movement
    stock_df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
    stock_df['+DI14'] = talib.PLUS_DI(high, low, close, timeperiod=14)
    stock_df['-DI14'] = talib.MINUS_DI(high, low, close, timeperiod=14)

    # MACD
    stock_df['DIF'], stock_df['DEA'], stock_df['MACD'] = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    """# Parabolic SAR
    stock_df['PSAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # Align PSAR and Close columns to handle NaN values
    psar_aligned, close_aligned = stock_df['PSAR'].align(stock_df['Close'], axis=0, copy=False)
    stock_df['PSAR_TREND'] = np.where(psar_aligned < close_aligned, 1, -1)"""

    # Relative Strength Index (RSI)
    stock_df['RSI5'] = talib.RSI(close, timeperiod=5)
    stock_df['RSI14'] = talib.RSI(close, timeperiod=14)

    # Stochastic Oscillator
    stock_df['%K'], stock_df['%D'] = talib.STOCH(
        high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )

    # Williams %R
    stock_df['%R'] = talib.WILLR(high, low, close, timeperiod=14)

    # Average True Range (ATR)
    stock_df['ATR14'] = talib.ATR(high, low, close, timeperiod=14)

    # Bollinger Bands
    stock_df['BB_Upper'], stock_df['BB_Middle'], stock_df['BB_Lower'] = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )

    # Standard Deviation
    stock_df['STD20'] = talib.STDDEV(close, timeperiod=20, nbdev=1)

    return stock_df