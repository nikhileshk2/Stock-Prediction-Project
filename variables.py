from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(input_shape, num_layers=2, units=50):
    """
    Build a standard LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of the input data (seq_length, num_features).
        num_layers (int): Number of LSTM layers.
        units (int): Number of units in each LSTM layer.
    
    Returns:
        model: Compiled LSTM model.
    """
    model = Sequential()
    
    # Add LSTM layers
    for i in range(num_layers):
        return_sequences = (i != num_layers - 1)  # Return sequences for all layers except the last
        if i == 0:
            # Pass input_shape only to the first layer
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_sequences))
        model.add(Dropout(0.2))  # Add dropout for regularization
    
    # Dense layer to map LSTM output to a single value (prediction)
    model.add(Dense(units=1))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model