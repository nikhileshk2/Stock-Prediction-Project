import pandas_ta as ta
import numpy as np
def calculate_technical_indicators(stock_df):
# Calculate Trend Indicators
    # Moving Averages
    stock_df['MA5'] = ta.sma(stock_df['Close'], length=5)
    stock_df['MA10'] = ta.sma(stock_df['Close'], length=10)

    # Average Directional Index (ADX) and Directional Movement
    adx = ta.adx(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14)
    stock_df['ADX'] = adx['ADX_14']
    stock_df['+DI14'] = adx['DMP_14']  # Positive Directional Movement
    stock_df['-DI14'] = adx['DMN_14']  # Negative Directional Movement

    # Ichimoku Cloud
    ichimoku = ta.ichimoku(stock_df['High'], stock_df['Low'], stock_df['Close'], tenkan=7, kijun=22, senkou=44)
    stock_df['Tenkan_sen'] = ichimoku[0]['ISA_7']  # Conversion Line
    stock_df['Kijun_sen'] = ichimoku[0]['ISB_22']  # Base Line
    stock_df['Senkou_A'] = ichimoku[0]['ITS_7']    # Leading Span A (shifted forward)
    stock_df['Senkou_B'] = ichimoku[0]['IKS_44']   # Leading Span B (shifted forward)

    # MACD
    macd = ta.macd(stock_df['Close'], fast=12, slow=26, signal=9)
    stock_df['DIF'] = macd['MACD_12_26_9']      # MACD Line
    stock_df['DEA'] = macd['MACDs_12_26_9']     # Signal Line
    stock_df['MACD'] = 2 * macd['MACDh_12_26_9']  # Histogram (scaled as per your script)

    # Parabolic SAR
    stock_df['PSAR'] = ta.psar(stock_df['High'], stock_df['Low'], stock_df['Close'], af=0.02, max_af=0.2)['PSARl_0.02_0.2']
    stock_df['PSAR_TREND'] = np.where(stock_df['PSAR'] < stock_df['Close'], 1, -1)  # 1 for bullish, -1 for bearish

    # Momentum Indicators
    # Relative Strength Index (RSI)
    stock_df['RSI5'] = ta.rsi(stock_df['Close'], length=5)
    stock_df['RSI14'] = ta.rsi(stock_df['Close'], length=14)

    # Stochastic Oscillator
    stoch = ta.stoch(stock_df['High'], stock_df['Low'], stock_df['Close'], k=14, d=3, smooth_k=1)
    stock_df['%K'] = stoch['STOCHk_14_3_1']
    stock_df['%D'] = stoch['STOCHd_14_3_1']

    # Williams %R
    stock_df['%R'] = ta.willr(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14)

    # Volatility Indicators
    # Average True Range (ATR)
    stock_df['ATR14'] = ta.atr(stock_df['High'], stock_df['Low'], stock_df['Close'], length=14)

    # Bollinger Bands
    bbands = ta.bbands(stock_df['Close'], length=20, std=2)
    stock_df['BB_Middle'] = bbands['BBM_20_2.0']
    stock_df['BB_Upper'] = bbands['BBU_20_2.0']
    stock_df['BB_Lower'] = bbands['BBL_20_2.0']

    # Standard Deviation
    stock_df['STD20'] = ta.stdev(stock_df['Close'], length=20)

    return stock_df   