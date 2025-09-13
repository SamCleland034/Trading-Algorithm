import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
from alpaca_trade_api import REST, TimeFrame
from datetime import datetime, timedelta
from zipfile import ZipFile
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import pandas_datareader as web
import datetime
import warnings
from datetime import datetime

# Alpaca API credentials
API_KEY = 'PKGJV4FWYKV0V1HX7FJX'
API_SECRET = 'ybZq5el0r5mZ0ct1IP7YPGf6AyzIuJ1bTV0Qup8k'
BASE_URL = 'https://paper-api.alpaca.markets'  # Change to 'https://api.alpaca.markets' for live

# Initialize API
api = REST(API_KEY, API_SECRET, BASE_URL)

# Parameters
MACD_FAST = 12  # For MACD
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
ATR_PERIOD = 14  # For stop-loss
BAR_TIMEFRAME = TimeFrame.Day
POSITION_SIZE = 100  # Shares per stock
POLL_INTERVAL = 300  # 5 min; adjust
NUM_STOCKS = 10
DATA_PERIOD = '1y'

# Fetch S&P 500 tickers
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        sp = pd.read_html(response.text, header=0)[0]
        print(f"Ticker List = {sp['Symbol'].to_list()}")
        return sp['Symbol'].to_list()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
    except IndexError:
        print("No table found with the specified header.")

# Download and parse Fama-French 5 factors (daily)
def get_ff_factors():
    end = datetime.today()
    start = end - timedelta(days=365)
    factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start, end)[0]
    factor_data /= 100 
    return factor_data

# Calculate Sharpe and FF alpha for a stock
def evaluate_stock(ticker, ff_df):
    try:
        data = yf.download(ticker, period=DATA_PERIOD, auto_adjust=False)['Adj Close'].resample('M').last().pct_change().dropna()
        data.index = data.index.strftime('%Y-%m')
        data = pd.DataFrame(data)
        data.columns = ['returns']
        merged = pd.merge(data, ff_df, left_index=True, right_index=True, how='left').dropna()
        # Sharpe Ratio
        rf_mean = merged['RF'].mean()
        excess_returns = merged['returns'] - merged['RF']
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(12)  # Monthly annualized
        # FF Alpha (regression)
        X = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X)
        y = excess_returns
        model = sm.OLS(y, X).fit()
        alpha = model.params['const'] * 12  # Monthly annualized
        return sharpe, alpha
    except Exception as e:
        return 0, 0

# Screen and select top stocks
def screen_stocks():
    tickers = get_sp500_tickers()
    ff_df = get_ff_factors()
    ff_df.index = ff_df.index.strftime("%Y-%m")
    results = []
    
    for ticker in tickers:
        sharpe, alpha = evaluate_stock(ticker, ff_df)
        if sharpe is not None:
            score = sharpe + alpha
            results.append({'ticker': ticker, 'sharpe': sharpe, 'alpha': alpha, 'score': score})
    
    df = pd.DataFrame(results).sort_values('score', ascending=False).head(NUM_STOCKS)
    print("Top stocks selected:\n", df)
    return df['ticker'].tolist()

# Get historical bars (now includes high/low/open for ATR/RSI)
def get_historical_data(ticker, timeframe, limit=200):
    bars = api.get_bars(ticker, timeframe, limit=limit).df
    return bars[['open', 'high', 'low', 'close']]  # Need OHLC for indicators

# Calculate indicators (MACD, RSI, ATR)
def calculate_indicators(data):
    # MACD
    ema_fast = data['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = data['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    data['macd'] = ema_fast - ema_slow
    data['macd_signal'] = data['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    data['atr'] = tr.rolling(window=ATR_PERIOD).mean()
    
    return data

# Generate signal with new conditions
def generate_signal(data):
    data = calculate_indicators(data)
    macd_cross_up = (data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]) and (data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2])
    macd_cross_down = (data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]) and (data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2])
    rsi_bullish = data['rsi'].iloc[-1] > 50
    rsi_bearish = data['rsi'].iloc[-1] < 50
    
    if macd_cross_up and rsi_bullish:
        return 1  # Buy
    elif macd_cross_down and rsi_bearish:
        return -1  # Sell
    return 0  # Hold

# Check stop-loss (returns True if triggered)
def check_stop_loss(current_price, entry_price, atr):
    if current_price < entry_price - (2 * atr):
        return True
    return False

# Get position
def get_position(ticker):
    try:
        return int(api.get_position(ticker).qty)
    except:
        return 0

# Trading loop
def trading_loop():
    print("Starting enhanced trading...")
    watchlist = screen_stocks()
    data_dict = {ticker: get_historical_data(ticker, BAR_TIMEFRAME, max(MACD_SLOW, RSI_PERIOD, ATR_PERIOD) * 2) for ticker in watchlist}  # Longer history for indicators
    entry_prices = {ticker: None for ticker in watchlist}  # Track entry
    last_screen_time = datetime.now()
    
    while True:
        if not api.get_clock().is_open:
            time.sleep(450)
            continue
        
        # Re-screen daily
        print('Checking if stocks are available...')
        if (datetime.now() - last_screen_time) > timedelta(days=1):
            watchlist = screen_stocks()
            data_dict = {ticker: get_historical_data(ticker, BAR_TIMEFRAME, max(MACD_SLOW, RSI_PERIOD, ATR_PERIOD) * 2) for ticker in watchlist}
            entry_prices = {ticker: None for ticker in watchlist}
            last_screen_time = datetime.now()
        
        for ticker in watchlist:
            # Append latest bar
            latest_bar = api.get_bars(ticker, BAR_TIMEFRAME, limit=1).df[['open', 'high', 'low', 'close']]
            data_dict[ticker] = pd.concat([data_dict[ticker], latest_bar]).tail(200)  # Keep reasonable length
            
            signal = generate_signal(data_dict[ticker])
            current_position = get_position(ticker)
            current_price = data_dict[ticker]['close'].iloc[-1]
            
            if signal == 1 and current_position == 0:
                print(f"Buy {ticker}: {POSITION_SIZE} shares")
                api.submit_order(symbol=ticker, qty=POSITION_SIZE, side='buy', type='market', time_in_force='gtc')
                entry_prices[ticker] = current_price  # Record entry
            
            elif signal == -1 and current_position > 0:
                print(f"Sell {ticker}: {current_position} shares (signal)")
                api.submit_order(symbol=ticker, qty=current_position, side='sell', type='market', time_in_force='gtc')
                entry_prices[ticker] = None
            
            # Check stop-loss if holding
            elif current_position > 0 and entry_prices[ticker] is not None:
                atr = data_dict[ticker]['atr'].iloc[-1]
                if check_stop_loss(current_price, entry_prices[ticker], atr):
                    print(f"Sell {ticker}: {current_position} shares (stop-loss)")
                    api.submit_order(symbol=ticker, qty=current_position, side='sell', type='market', time_in_force='gtc')
                    entry_prices[ticker] = None
            
            else:
                print("No stocks to buy or sell currently...")
        
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    trading_loop()