from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
from datetime import datetime, timedelta
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import pandas as pd
import traceback

app = Flask(__name__)

# Constants
NEWS_API_KEY = "2025548ba34a4294a0f3c18c36311f39"

# Load models at startup
models = {}
try:
    for symbol in ["AAPL", "TSLA", "UNP", "META", "MSFT", "NVDA", "XOM", "INTC", "AVGO", "PLTR", "NXPI"]:
        with open(f"models/{symbol}_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"models/{symbol}_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        models[symbol] = (model, scaler)
except Exception as e:
    print(f"Error loading models: {str(e)}")

def fetch_news_sentiment(symbol, days=5):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&pageSize=100"
        response = requests.get(url)
        if response.status_code != 200:
            return 0
            
        news_data = response.json()
        if not news_data.get('articles'):
            return 0
            
        analyzer = SentimentIntensityAnalyzer()
        sentiments = []
        
        for article in news_data['articles'][:days]:  # Get recent articles
            if article.get('title') and article.get('description'):
                text = article['title'] + " " + article['description']
                sentiment = analyzer.polarity_scores(text)['compound']
                sentiments.append(sentiment)
                
        return np.mean(sentiments) if sentiments else 0
        
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return 0

def get_price_for_past_date(symbol, target_date):
    try:
        # Convert string date to datetime if needed
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Get data for the specific date
        end_date = target_date + timedelta(days=1)  # Include target date
        start_date = target_date - timedelta(days=5)  # Look back 5 days in case of holidays
        
        stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Find the closest available date not exceeding target date
        available_dates = stock.index[stock.index <= target_date]
        if len(available_dates) > 0:
            closest_date = available_dates[-1]
            price = float(stock.loc[closest_date, 'Close'])
            return price
            
        return None
        
    except Exception as e:
        print(f"Error getting price for date {target_date}: {str(e)}")
        return None, None

def calculate_rsi(prices, periods=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Add trend detection
def detect_trend(prices, window=20):
    if isinstance(prices, np.ndarray):
        if len(prices.shape) > 1:
            prices = prices.ravel()

    prices = pd.Series(prices)
    
    if len(prices) < (window * 2):
        return 0, 0.0
    
    # Calculate moving averages
    ma_short = prices.iloc[-window:].mean()
    ma_long = prices.iloc[-(window * 2):-window].mean()
    
    if pd.isna(ma_short) or pd.isna(ma_long) or ma_long == 0:
        return 0, 0.0
        
    trend = 1 if ma_short > ma_long else -1
    trend_strength = abs(ma_short - ma_long) / ma_long
    
    return trend, trend_strength

def add_technical_indicators(df, window=14):
    # Basic price data already exists: Open, High, Low, Close, Volume
    
    # Calculate rolling stats first
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    
    # Basic indicators
    df['volatility'] = df['Close'].rolling(window=window).std()
    df['rsi'] = calculate_rsi(df['Close'], periods=14)  # Explicitly set to 14
    df['ma'] = df['Close'].rolling(window=window).mean()
    df['volume_ma'] = df['Volume'].rolling(window=window).mean()
    df['price_momentum'] = df['Close'].pct_change(periods=5)
    
    # Add Bollinger Bands
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df.ffill().bfill()

def prepare_stock_data(symbol, target_date, days=60):  # Match training sequence length
    try:
        # Convert target_date to datetime if it's a string
        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d')
            
        # Get historical data with a buffer for sufficient data
        start_date = target_date - timedelta(days=days + 100)
        end_date = target_date if target_date <= datetime.now() else datetime.now()
        
        print(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        # Download stock data with progress=False to reduce noise
        stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if stock.empty:
            print(f"No data found for {symbol}")
            return None
            
        print(f"Downloaded {len(stock)} rows of data")
        
        # Convert to DataFrame and handle column names
        df = pd.DataFrame(stock).reset_index()
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
            
        if len(df) < days:
            print(f"Insufficient data points: {len(df)} < {days}")
            return None
            
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Feature columns
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'volatility', 'rsi', 'ma', 'volume_ma',
            'price_momentum', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Verify all columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return None
            
        stock_data = df[feature_columns].iloc[-days:].values
        print(f"Prepared stock data shape: {stock_data.shape}")
        
        if stock_data.shape[0] != days:
            print(f"Final data shape mismatch: got {stock_data.shape[0]}, expected {days}")
            return None
            
        # Add sentiment
        sentiment = fetch_news_sentiment(symbol)
        sentiment_column = np.full((days, 1), sentiment)
        
        # Combine features
        combined_data = np.hstack([stock_data, sentiment_column])
        print(f"Final combined data shape: {combined_data.shape}")
        
        return combined_data
        
    except Exception as e:
        print(f"Error in prepare_stock_data: {str(e)}")
        return None

def get_current_price(symbol):
    try:
        # Try to get the most recent day's data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Get last 5 days in case of holidays/weekends
        
        stock = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if len(stock) > 0:  # Changed condition
            last_price = float(stock['Close'].iloc[-1])
            return last_price
            
        print(f"No recent data found for {symbol}, trying fallback method")
        # Fallback method using Ticker
        ticker = yf.Ticker(symbol)
        price = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose')
        if price is not None:
            price = float(price)
        print(f"Fallback price for {symbol}: {price}")
        return price
        
    except Exception as e:
        print(f"Error getting current price for {symbol}: {str(e)}")
        return None

def make_recursive_prediction(model, scaler, initial_data, days_ahead):
    predictions = []
    current_data = initial_data.copy()
    
    # Create copies of the last row for prediction updates
    last_row = current_data[-1].copy()
    for _ in range(days_ahead):
        # Scale current data
        scaled_data = scaler.transform(current_data)
        
        # Prepare inputs
        X_stock = scaled_data[np.newaxis, :, :-1]
        X_sentiment = scaled_data[np.newaxis, :, -1:]
        
        # Make single prediction
        prediction = model.predict([X_stock, X_sentiment])
        
        # dummy row for inverse transform
        dummy_row = np.zeros_like(scaled_data[-1:])
        dummy_row[0, 3] = prediction[0][0]  # Close price
        
        # Get predicted price
        predicted_full = scaler.inverse_transform(dummy_row)
        predicted_price = predicted_full[0, 3]
        predictions.append(predicted_price)
        
        # Update new row for next prediction
        new_row = last_row.copy()
        new_row[0] = new_row[3]  # Set Open to previous Close
        new_row[3] = predicted_price  # Set new Close
        new_row[1] = max(new_row[0], predicted_price)  # High
        new_row[2] = min(new_row[0], predicted_price)  # Low
        
        # Update data for next prediction by removing oldest and adding newest
        current_data = np.vstack([current_data[1:], [new_row]])
        last_row = new_row.copy()
    
    return predictions

def is_business_day(date):
    return date.weekday() < 5  # Monday = 0, Friday = 4

def get_business_days_until(start_date, end_date):
    """Calculate number of business days between two dates"""
    days = 0
    current = start_date
    while current <= end_date:
        if is_business_day(current):
            days += 1
        current += timedelta(days=1)
    return days

def get_next_business_dates(start_date, num_days):
    """Get list of next business dates"""
    dates = []
    current = start_date
    while len(dates) < num_days:
        current += timedelta(days=1)
        if is_business_day(current):
            dates.append(current)
    return dates

def format_date(date_obj):
    """Format datetime object to dd/mm/yyyy string"""
    if isinstance(date_obj, str):
        # If it's already a string in yyyy-mm-dd format, convert it first
        date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
    return date_obj.strftime('%d/%m/%Y')

STOCK_SYMBOLS = ["AAPL", "TSLA", "UNP", "META", "MSFT", "NVDA", "XOM", "INTC", "AVGO", "PLTR", "NXPI"]

@app.route('/')
def home():
    return render_template("index.html", symbols=STOCK_SYMBOLS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form.get("symbol")
        target_date = request.form.get("date")
        
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Parse target date
        target_datetime = datetime.strptime(target_date, '%Y-%m-%d')
        current_datetime = datetime.now()
        is_future = target_datetime > current_datetime
        
        # Determine if this is a historical prediction
        is_historical = target_datetime.date() < current_datetime.date()
        
        # Calculate business days difference for future predictions
        if is_future:
            business_days = get_business_days_until(current_datetime, target_datetime)
            days_ahead = business_days if business_days > 0 else 1
        else:
            days_ahead = 0

        model, scaler = models[symbol]
        
        # Handle past/present dates differently from future dates
        if not is_future:
            actual_price = get_price_for_past_date(symbol, target_date)
            if actual_price is None:
                return jsonify({"error": f"Could not find price data for {target_date}"}), 400
        else:
            actual_price = get_current_price(symbol)
            
        result = prepare_stock_data(symbol, target_date, days=60)
        if result is None:
            return jsonify({"error": "Failed to fetch stock data"}), 400
            
        stock_data = result
        
        if is_future:
            predictions = make_recursive_prediction(
                model, 
                scaler, 
                stock_data, 
                days_ahead
            )
            
            predicted_price = predictions[-1]
            
            intermediate_predictions = {}
            current = current_datetime
            pred_idx = 0
            
            while current <= target_datetime and pred_idx < len(predictions):
                current += timedelta(days=1)
                if is_business_day(current):
                    intermediate_predictions[current.strftime('%Y-%m-%d')] = round(predictions[pred_idx], 2)
                    if current.date() == target_datetime.date():
                        predicted_price = predictions[pred_idx]
                    pred_idx += 1
        else:
            scaled_data = scaler.transform(stock_data)
            X_stock = scaled_data[np.newaxis, :, :-1]
            X_sentiment = scaled_data[np.newaxis, :, -1:]
            prediction = model.predict([X_stock, X_sentiment])
            dummy_row = np.zeros_like(scaled_data[-1:])
            dummy_row[0, 3] = prediction[0][0]
            predicted_full = scaler.inverse_transform(dummy_row)
            predicted_price = predicted_full[0, 3]

        # Calculate price change
        price_change = ((predicted_price - actual_price) / actual_price) * 100

        # Validate predicted price
        if np.isnan(predicted_price):
            return jsonify({"error": "Prediction resulted in NaN value"}), 500

        response_data = {
            "symbol": symbol,
            "date": target_date,
            "current_price": round(actual_price, 2),
            "predicted_price": round(predicted_price, 2),
            "price_change": round(price_change, 2),
            "prediction_direction": "up" if price_change > 0 else "down",
            "timestamp": datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "is_future_prediction": is_future,
            "is_historical": is_historical,
            "intermediate_predictions": {
                date: price
                for date, price in (intermediate_predictions or {}).items()
            } if is_future else None,
            "days_ahead": days_ahead
        }

        return jsonify(response_data)
        
    except ValueError as ve:
        return jsonify({"error": f"Invalid date format. Use YYYY-MM-DD format: {str(ve)}"}), 400
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)