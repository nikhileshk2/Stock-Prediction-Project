def preprocess_data(stock_data):
    stock_data = stock_data.dropna()

    # Reset the index
    stock_data = stock_data.reset_index()

    # Remove unnecessary columns (e.g., 'Date' can be kept if needed for visualization, but not used as a feature)
    technical_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'BB_Middle', 'BB_Upper', 'BB_Lower']]

    # Display the preprocessed data
    return technical_data