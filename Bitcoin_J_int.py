import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
bitcoin_price_data = pd.read_csv("bitcoin_price_data.csv")

# Convert Date column to datetime format
bitcoin_price_data['Date'] = pd.to_datetime(bitcoin_price_data['Date'])

# Prepare the data for modeling
X = np.array(pd.to_datetime(bitcoin_price_data['Date']).apply(lambda x: x.value)).reshape(-1, 1) # Convert dates to numeric format
y = np.array(bitcoin_price_data['Close'])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions for the next month
last_date = bitcoin_price_data['Date'].max()
future_dates = pd.date_range(last_date, periods=30, freq='D')[1:]
X_future = np.array(future_dates.map(lambda x: x.value)).reshape(-1, 1)
y_pred = model.predict(X_future)

# Plot the actual and predicted prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(bitcoin_price_data['Date'], y, color='blue', label='Actual')
ax.plot(future_dates, y_pred, color='green', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Bitcoin Price (USD)')
ax.set_title('Bitcoin Price Prediction for Next Month')
ax.legend()
plt.show()
