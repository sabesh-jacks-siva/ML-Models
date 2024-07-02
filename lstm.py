# lstm.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data.csv', usecols=['date', 'value'])
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Split into training and test sets
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

# Create dataset function
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Create datasets
look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Create and train the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Plot baseline and predictions
plt.plot(data.index, scaler.inverse_transform(data_scaled), label='True Data')
plt.plot(data.index[:len(train_predict)], train_predict, label='Train Prediction')
plt.plot(data.index[len(train_predict)+look_back:], test_predict, label='Test Prediction')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('LSTM Time Series Forecasting')
plt.legend()
plt.show()
