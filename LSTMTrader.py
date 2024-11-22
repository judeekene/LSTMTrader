import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Data Collection
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Data Preprocessing
# Normalize the data
data['Close'] = data['Close'] / data['Close'].max()

# Create features and labels
window_size = 10
features = []
labels = []

for i in range(len(data) - window_size):
    features.append(data['Close'].values[i:i + window_size])
    labels.append(data['Close'].values[i + window_size])

features = np.array(features)
labels = np.array(labels)

# Reshape features for LSTM layer
features = np.reshape(features, (features.shape[0], features.shape[1], 1))

# Step 3: Build the Neural Network
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the Model
model.fit(features, labels, epochs=50, batch_size=32)

# Step 5: Make Predictions
predictions = model.predict(features)

# Print the first 10 predictions
print(predictions[:10])