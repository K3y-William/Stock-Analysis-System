import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Download stock data
ticker = "MSFT"
df = yf.download(ticker, start="2010-01-01", end="2024-01-01")

# Step 2: Prepare the data
# Use 'Close' prices for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data (values between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define training data size (80% for training)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]

# Create sequences of 60 timesteps for LSTM input
def create_sequences(data, timestep=60):
    x, y = [], []
    for i in range(timestep, len(data)):
        x.append(data[i - timestep:i, 0])  # Input sequence (60 timesteps)
        y.append(data[i, 0])  # Corresponding target
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_data)

# Reshape for LSTM: (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 3: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))  # Final output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
model.fit(x_train, y_train, batch_size=64, epochs=20)

# Step 5: Test data preparation
test_data = scaled_data[train_size - 60:]
x_test, y_test = create_sequences(test_data)

# Reshape the test data to LSTM input shape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Step 6: Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Scale back to original values

# Step 7: Plot results
train = df[:train_size]
valid = df[train_size:]
valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.title('LSTM Model Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'], label='Train')
plt.plot(valid[['Close', 'Predictions']], label=['Actual', 'Predicted'])
plt.legend(loc='upper left')
plt.show()
