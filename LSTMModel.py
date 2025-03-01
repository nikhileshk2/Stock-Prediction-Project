from HybridLSTMmodel import HybridLSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
def build_hybrid_lstm_model(input_shape, num_layers=2, units=50, ratio=0.9):
    model = Sequential()
    
    # Add HybridLSTM layers
    for i in range(num_layers):
        return_sequences = (i != num_layers - 1)  # Return sequences for all layers except the last
        if i == 0:
            # Pass input_shape only to the first layer
            model.add(HybridLSTM(units=units, ratio=ratio, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(HybridLSTM(units=units, ratio=ratio, return_sequences=return_sequences))
        model.add(Dropout(0.2))  # Add dropout for regularization
    
    # Dense layer to map LSTM output to a single value (prediction)
    model.add(Dense(units=1))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model