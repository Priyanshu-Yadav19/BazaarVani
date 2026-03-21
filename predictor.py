import pandas as pd
import numpy as np
import requests
import datetime
import warnings
import io
import base64
import ta
import json
import plotly.graph_objects as go
import plotly.utils
# matplotlib removed for plotly integration

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os

load_dotenv()

# Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

def fetch_data(ticker, lookback_days):
    try:
        # Convert Yahoo Finance Indian suffix to Alpha Vantage expected format if applicable
        av_ticker = ticker.replace('.NS', '.BSE') 
        outputsize = 'full' if lookback_days > 100 else 'compact'
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_ticker}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        res = requests.get(url).json()
        
        if "Time Series (Daily)" not in res and outputsize == 'full':
            # Fallback to compact if 'full' fails (e.g., due to premium key requirement)
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={av_ticker}&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}"
            res = requests.get(url).json()
            
        if "Time Series (Daily)" not in res:
            print("Alpha Vantage API limits/Error:", res)
            
            # --- POLYGON FALLBACK ---
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            poly_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            try:
                url_poly = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/1/day/{start_str}/{end_str}?apiKey={POLYGON_API_KEY}"
                res_poly = requests.get(url_poly).json()
                data = res_poly.get("results", [])
                
                if data:
                    df = pd.DataFrame(data)
                    df['Date'] = pd.to_datetime(df['t'], unit='ms')
                    df.rename(columns={'c':'Close','o':'Open','h':'High','l':'Low','v':'Volume'}, inplace=True)
                    return df[['Date','Open','High','Low','Close','Volume']]
            except Exception as e:
                print("Polygon error:", e)

            # --- LAST RESORT: YAHOO FINANCE RAW API ---
            print("Trying last resort Yahoo Finance Raw API...")
            try:
                yf_url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range={lookback_days}d"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                yf_res = requests.get(yf_url, headers=headers).json()
                
                result = yf_res.get('chart', {}).get('result')
                if not result:
                    return pd.DataFrame()
                    
                result = result[0]
                timestamps = result.get('timestamp', [])
                quote = result.get('indicators', {}).get('quote', [{}])[0]
                
                df = pd.DataFrame({
                    'Date': pd.to_datetime(timestamps, unit='s'),
                    'Open': quote.get('open', []),
                    'High': quote.get('high', []),
                    'Low': quote.get('low', []),
                    'Close': quote.get('close', []),
                    'Volume': quote.get('volume', [])
                })
                return df.dropna()
            except Exception as e:
                print("Yahoo Finance Raw API fallback failed too:", e)
                return pd.DataFrame()
            
        time_series = res["Time Series (Daily)"]
        
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={
            'index': 'Date',
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col])
            
        df = df.sort_values('Date').tail(lookback_days)
        df.reset_index(drop=True, inplace=True)
        
        return df
    except Exception as e:
        print("Exception fetching Alpha Vantage Data:", e)
        return pd.DataFrame()

def process_data(df):
    df = df.copy()
    if 'Date' not in df.columns and 'index' in df.columns:
        df.rename(columns={'index': 'Date'}, inplace=True)
        
    df.columns = df.columns.astype(str)
    df = df.dropna()
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    if 'Date' in df.columns:
        df = df.sort_values("Date")
    
    if 'Close' not in df.columns:
        return pd.DataFrame()

    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(10).std()

    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Day'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['Weekday'] = df['Date'].dt.weekday
    else:
        df['Day'] = 0
        df['Month'] = 0
        df['Weekday'] = 0

    try:
        df['SMA'] = ta.trend.sma_indicator(df['Close'].squeeze(), window=14)
        df['EMA'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=14)
        df['RSI'] = ta.momentum.rsi(df['Close'].squeeze(), window=14)
        df['MACD'] = ta.trend.macd(df['Close'].squeeze())

        bb = ta.volatility.BollingerBands(df['Close'].squeeze())
        df['BB_high'] = bb.bollinger_hband()
        df['BB_low'] = bb.bollinger_lband()
    except Exception as e:
        pass

    df.dropna(inplace=True)
    
    target_col = df['Close']
    for lag in range(1,6):
        df[f'lag_{lag}'] = target_col.shift(lag)

    df.dropna(inplace=True)
    return df

def fetch_news(ticker):
    news_query = ticker.replace('.NS', '').replace('.BO', '')
    if news_query.endswith('=F'):
         news_query = news_query.replace('=F', '')
         
    # Finnhub needs date parameters
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
    
    finnhub_url = f"https://finnhub.io/api/v1/company-news?symbol={news_query}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
    newsapi_url = f"https://newsapi.org/v2/everything?q={news_query}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    
    headlines = []
    try:
        finnhub_response = requests.get(finnhub_url).json()
        if isinstance(finnhub_response, list):
            for n in finnhub_response[:10]:
                if 'headline' in n:
                    headlines.append(n['headline'])
    except:
        pass
        
    try:
        newsapi_res = requests.get(newsapi_url).json()
        articles = newsapi_res.get("articles", [])
        for n in articles[:10]:
            if 'title' in n:
                headlines.append(n['title'])
    except:
        pass
            
    return list(set(headlines))

def analyze_sentiment(headlines):
    if not headlines:
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in headlines:
        if not text: continue
        vader = analyzer.polarity_scores(text)['compound']
        blob = TextBlob(text).sentiment.polarity
        score = (vader + blob) / 2
        scores.append(score)
    return np.mean(scores) if scores else 0.0

def train_and_predict(df, prediction_days):
    if df.empty or len(df) < 10:
        return None, None
        
    drop_cols = ['Date', 'Close']
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    target = df['Close'].squeeze()
    
    split = int(len(df)*0.8)
    
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    
    if len(X_train) == 0 or len(X_test) == 0:
        X_train = features
        y_train = target
        X_test = features.tail(prediction_days)
        y_test = target.tail(prediction_days)
        if len(y_train) == 0: return None, None
    
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor()
    }
    
    if isinstance(X_train.columns, pd.MultiIndex):
        X_train.columns = ['_'.join(map(str, col)).strip() for col in X_train.columns.values]
        X_test.columns = ['_'.join(map(str, col)).strip() for col in X_test.columns.values]
        features.columns = ['_'.join(map(str, col)).strip() for col in features.columns.values]
        
    best_rmse = float('inf')
    best_model = None
    best_name = None
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if len(X_test) > 0:
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_name = name
            else:
                best_model = model
                best_name = name
        except:
            pass
            
    if not best_model:
        for name, model in models.items():
            best_model = model
            best_name = name
            break
            
    best_model.fit(features, target)
    
    # Recursive prediction to avoid flat lines
    last_features = features.iloc[-1].copy()
    future_preds = []
    current_close = target.iloc[-1]
    
    # Calculate recent momentum and volatility for more realistic trend
    recent_returns = target.pct_change().tail(10)
    avg_momentum = recent_returns.mean()
    volatility = recent_returns.std()
    
    for i in range(prediction_days):
        # Prepare current prediction
        pred = best_model.predict(last_features.values.reshape(1, -1))[0]
        
        # Add dynamic trend factor based on recent momentum to prevent flattening
        decay = 1.0 - (i / prediction_days) * 0.5 # Gradually trust the model less
        trend_factor = (avg_momentum * current_close * decay)
        
        # Add a tiny bit of random walk based on historical volatility
        noise = np.random.normal(0, volatility * current_close * 0.2)
        
        final_pred = pred + trend_factor + noise
        future_preds.append(final_pred)
        
        # Update features recursively (Shift lags)
        if 'lag_1' in last_features:
            for j in range(5, 1, -1):
                if f'lag_{j}' in last_features and f'lag_{j-1}' in last_features:
                    last_features[f'lag_{j}'] = last_features[f'lag_{j-1}']
            last_features['lag_1'] = final_pred
            
        # Update core price-based features roughly
        if 'Close' in last_features:
            last_features['Close'] = final_pred
        if 'SMA' in last_features:
            last_features['SMA'] = (last_features['SMA'] * 13 + final_pred) / 14
        if 'Return' in last_features:
            last_features['Return'] = (final_pred - current_close) / current_close
            
        current_close = final_pred
        
    return best_name, np.array(future_preds)

def generate_plotly_chart(df, ticker, is_future=False, future_preds=None):
    fig = go.Figure()
    
    currency_symbol = "₹" if (ticker.upper().endswith('.NS') or ticker.upper().endswith('.BO')) else "$"
    
    if not is_future:
        x_data = df['Date'] if 'Date' in df.columns else list(range(len(df)))
        y_data = df['Close']
        line_color = '#ff4785'
        chart_title = ""
        hover_template = f"<b>Date:</b> %{{x}}<br><b>Price:</b> {currency_symbol}%{{y:.2f}}"
    else:
        x_data = [f"Day {i+1}" for i in range(len(future_preds))]
        y_data = future_preds
        line_color = '#ff4785'
        chart_title = ""
        hover_template = f"<b>%{{x}}</b><br><b>Pred:</b> {currency_symbol}%{{y:.2f}}"

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines+markers' if is_future else 'lines',
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color, opacity=0.8) if is_future else None,
        name=ticker,
        hovertemplate=hover_template,
        fill='tozeroy' if not is_future else None,
        fillcolor='rgba(255, 71, 133, 0.05)' if not is_future else None
    ))

    fig.update_layout(
        title=dict(text=chart_title),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True, gridcolor='rgba(0,0,0,0.05)',
            tickfont=dict(color='#888888'),
            title=dict(text="", font=dict(color='#888888'))
        ),
        yaxis=dict(
            showgrid=True, gridcolor='rgba(0,0,0,0.05)',
            tickfont=dict(color='#888888'),
            title=dict(text="", font=dict(color='#888888'))
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        hovermode='x unified',
        font=dict(family='Outfit', color='#333333'),
        showlegend=False
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def analyze_stock(ticker, prediction_days=14, lookback_days=365):
    raw_df = fetch_data(ticker, lookback_days)
    if raw_df.empty:
        return {"error": f"Could not fetch data for ticker {ticker}."}
        
    df = process_data(raw_df)
    if df.empty:
        return {"error": f"Not enough valid data to process for {ticker}."}
        
    best_model_name, future_preds = train_and_predict(df, prediction_days)
    if future_preds is None:
        return {"error": "Failed to train models due to insufficient data."}
        
    news = fetch_news(ticker)
    sentiment_score = analyze_sentiment(news)
    
    current_close = df['Close'].iloc[-1].item() if hasattr(df['Close'].iloc[-1], 'item') else df['Close'].iloc[-1]
    trend_pct = (np.mean(future_preds) - current_close) / current_close
    
    if trend_pct > 0.005: 
        recommendation = "BUY" if sentiment_score > -0.1 else "HOLD"
    elif trend_pct < -0.005:
        recommendation = "SELL" if sentiment_score < 0.1 else "HOLD"
    else:
        if sentiment_score > 0.2:
            recommendation = "BUY"
        elif sentiment_score < -0.2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
    plot_live = generate_plotly_chart(df, ticker, is_future=False)
    plot_future = generate_plotly_chart(df, ticker, is_future=True, future_preds=future_preds)
    
    currency_symbol = "₹" if (ticker.upper().endswith('.NS') or ticker.upper().endswith('.BO')) else "$"
    
    return {
        "ticker": ticker,
        "best_model": best_model_name,
        "predictions": [round(float(p), 2) for p in future_preds],
        "sentiment_score": round(sentiment_score, 4),
        "news": news[:5] if news else ["API Limits: No recent news fetched"],
        "recommendation": recommendation,
        "plot_data": plot_live,
        "future_plot_data": plot_future,
        "current_price": round(current_close, 2),
        "currency": currency_symbol
    }
