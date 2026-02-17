import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 📌 Show Logo
st.image("logo.png", width=200)

st.title("FinFusion AI - Financial Time Series Predictor (₹ INR)")

USD_TO_INR = 83

# Company Name → Stock Symbol mapping
stock_dict = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Microsoft": "MSFT"
}

# Dropdown with company names
company_name = st.selectbox("Select Company", list(stock_dict.keys()))

# Get actual symbol
stock_option = stock_dict[company_name]

days_to_predict = st.slider("Days to Predict", 1, 7, 3)

# Load Data
data = yf.download(stock_option, start="2020-01-01", end="2024-01-01")
data = data[['Close']]

# Convert to INR
data['Close'] = data['Close'] * USD_TO_INR

# Create prediction column
data['Prediction'] = data['Close'].shift(-1)
data = data.dropna()

# Training
X = np.array(data[['Close']])
y = np.array(data[['Prediction']])

model = LinearRegression()
model.fit(X, y)

# Predict full dataset
data['Predicted_Line'] = model.predict(X)

# Accuracy
accuracy = r2_score(y, data['Predicted_Line'])
st.subheader(f"Model Accuracy: {accuracy*100:.2f}%")

# Future prediction
last_price = float(data['Close'].iloc[-1])
future_prices = []
current_price = last_price

for i in range(days_to_predict):
    next_price = model.predict(np.array([[current_price]]))[0][0]
    future_prices.append(next_price)
    current_price = next_price

# Buy / Sell Signal
st.subheader("AI Trading Signal")

if future_prices[-1] > last_price:
    st.success("BUY Signal 📈")
else:
    st.error("SELL Signal 📉")

# Show future predictions
st.subheader(f"Predicted Future Prices for {company_name} (₹)")

for i, price in enumerate(future_prices):
    st.write(f"Day {i+1}: ₹{price:,.2f}")

# Graph
st.subheader(f"{company_name} Stock Price Graph (₹)")

plt.figure(figsize=(10,5))

plt.plot(data['Close'].values, label="Actual Price", linewidth=2)
plt.plot(data['Predicted_Line'].values, color='red', linestyle='dashed', label="Predicted Price")

future_days = list(range(len(data), len(data) + days_to_predict))
plt.plot(future_days, future_prices, color='red', marker='o')

plt.legend()
plt.xlabel("Days")
plt.ylabel("Price (₹)")

st.pyplot(plt)
