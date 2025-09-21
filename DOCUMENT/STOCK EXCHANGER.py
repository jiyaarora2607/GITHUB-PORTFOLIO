from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

# Serve the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction logic
def predict_stock_price(stock_symbol, date_range="3mo"):
    try:
        # Get stock data
        data = yf.download(stock_symbol, period=date_range, interval="1d")
    except Exception as e:
        raise ValueError("Error fetching data from yfinance: " + str(e))

    if data.empty or 'Close' not in data.columns:
        raise ValueError("No closing price data available.")

    close_prices = data['Close'].values
    days = len(close_prices)

    if days < 10:
        raise ValueError("Not enough data to make a prediction for the selected date range.")

    # Keep for graph
    last_prices = close_prices
    last_dates = data.index.strftime('%Y-%m-%d').tolist()

    # Prepare data
    close_prices = close_prices.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    x, y = [], []
    window_size = min(60, days - 1)
    for i in range(window_size, len(scaled)):
        x.append(scaled[i - window_size:i].flatten())
        y.append(scaled[i])
    x, y = np.array(x), np.array(y)

    model = LinearRegression()
    model.fit(x, y)

    # Predict next day's price
    x_test = scaled[-window_size:].reshape(1, -1)
    predicted_scaled = model.predict(x_test)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))[0][0]

    return {
        'predicted_price': round(float(predicted_price), 2),
        'dates': last_dates,
        'prices': [round(float(p), 2) for p in last_prices]
    }

# API endpoint
@app.route('/predict', methods=['GET'])
def predict():
    stock = request.args.get('stock', 'AAPL')
    date_range = request.args.get('dateRange', '3mo')
    try:
        result = predict_stock_price(stock, date_range)
        return jsonify(result)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)