import time
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from alpaca_trade_api import REST, TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import requests
import pytz

# Alpaca API credentials
API_KEY = 'PKGJV4FWYKV0V1HX7FJX'
API_SECRET = 'ybZq5el0r5mZ0ct1IP7YPGf6AyzIuJ1bTV0Qup8k'
BASE_URL = 'https://paper-api.alpaca.markets'

api = REST(API_KEY, API_SECRET, BASE_URL)

# Parameters
MACD_FAST = 4
MACD_SLOW = 11
MACD_SIGNAL = 6
RSI_PERIOD = 6
BAR_TIMEFRAME = TimeFrame(15, TimeFrameUnit.Minute)
POSITION_SIZE = 1
DATA_PERIOD = '1d'
WATCHLIST_SIZE = 3
UPDATE_INTERVAL = 3600
PERCENT_PROFIT = 2

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        sp = pd.read_html(response.text, header=0)[0]
        return sp['Symbol'].to_list()
    except:
        return []

def get_trending_stocks(tickers, limit=WATCHLIST_SIZE):
    if not tickers:
        return []
    try:
        if not hasattr(get_trending_stocks, "last_timeframe"):
            get_trending_stocks.last_timeframe = TimeFrame.Day
        timeframe = get_trending_stocks.last_timeframe

        bars = api.get_bars(tickers, timeframe, limit=limit).df
        if bars.empty:
            print("Warning: No data returned for trending stocks.")
            return []

        daily_data = bars.groupby('symbol').agg({'open': 'first', 'close': 'last'}).reset_index()
        daily_data['pct_change'] = ((daily_data['close'] - daily_data['open']) / daily_data['open']) * 100
        trending = daily_data['pct_change'].abs().sort_values(ascending=False).head(limit).index
        return daily_data.loc[trending, 'symbol'].tolist()
    except Exception as e:
        print(f"Error fetching trending stocks: {e}")
        return []

def get_multi_historical_data(tickers, timeframe, limit=200):
    try:
        # Define the timeframe as UTC and format as ISO 8601 string
        UTC = pytz.timezone('UTC')
        start_time = dt.datetime.now(tz=UTC) - dt.timedelta(hours=3)
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        bars = api.get_bars(tickers, timeframe, start=start_time_str, limit=limit).df
        time.sleep(2)
        return bars if not bars.empty else pd.DataFrame()
    except Exception as e:
        print(f"Error fetching multi-historical data: {e}")
        return pd.DataFrame()

# Calculate only MACD and RSI (no ATR)
def calculate_indicators(data):
    if data.empty or len(data) < max(MACD_SLOW, RSI_PERIOD):
        return data

    ema_fast = data['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = data['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    data['macd'] = ema_fast - ema_slow
    data['macd_signal'] = data['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']

    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs.replace(0, 1e-10)))

    return data

def generate_signal(data):
    if data.empty or len(data) < max(MACD_SLOW, RSI_PERIOD):
        return 0
    data = calculate_indicators(data)
    macd_cross_up = (data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]) and (data['macd'].iloc[-2] <= data['macd_signal'].iloc[-2])
    macd_cross_down = (data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]) and (data['macd'].iloc[-2] >= data['macd_signal'].iloc[-2])
    rsi_bullish = data['rsi'].iloc[-1] > 40
    rsi_bearish = data['rsi'].iloc[-1] < 40

    if macd_cross_up and rsi_bullish:
        return 1  # Buy
    elif macd_cross_down and rsi_bearish:
        return -1  # Sell
    return 0

# Now check_exit only uses profit target
def check_exit(current_price, entry_price):
    profit_target = entry_price * (1 + PERCENT_PROFIT / 100)
    return current_price >= profit_target

def get_position(ticker):
    try:
        return int(api.get_position(ticker).qty)
    except:
        return 0

def trading_loop():
    all_tickers = get_sp500_tickers()
    current_positions = [pos.symbol for pos in api.list_positions()]
    watchlist = list(set(get_trending_stocks(all_tickers) + current_positions))
    data_dict = {ticker: pd.DataFrame() for ticker in watchlist}
    entry_prices = {ticker: None for ticker in watchlist}
    last_update_time = datetime.now()

    for ticker in watchlist:
        pos = get_position(ticker)
        if pos > 0:
            try:
                entry_prices[ticker] = float(api.get_position(ticker).avg_entry_price)
            except:
                pass

    while True:
        if not api.get_clock().is_open:
            print("Market closed, waiting...")
            time.sleep(300)
            continue

        if (datetime.now() - last_update_time).total_seconds() >= UPDATE_INTERVAL:
            current_positions = [pos.symbol for pos in api.list_positions()]
            watchlist = list(set(get_trending_stocks(all_tickers) + current_positions))
            data_dict = {ticker: pd.DataFrame() for ticker in watchlist}
            entry_prices = {ticker: None for ticker in watchlist}
            last_update_time = datetime.now()

        account = api.get_account()
        available_cash = float(account.cash)

        if watchlist:
            bars = get_multi_historical_data(watchlist, BAR_TIMEFRAME)
            if not bars.empty:
                for ticker in watchlist:
                    if 'symbol' in bars.columns:
                        ticker_data = bars[bars['symbol'] == ticker].copy()
                    elif bars.index.names and 'symbol' in bars.index.names:
                        ticker_data = bars.xs(ticker, level='symbol').copy()
                    else:
                        ticker_data = pd.DataFrame()

                    if not ticker_data.empty:
                        data_dict[ticker] = pd.concat([data_dict[ticker], ticker_data]).tail(50)
                        data_dict[ticker] = calculate_indicators(data_dict[ticker])

                for ticker in watchlist:
                    if len(data_dict[ticker]) < max(MACD_SLOW, RSI_PERIOD):
                        print(f"Skipping Ticker {ticker} for now...")
                        continue

                    signal = generate_signal(data_dict[ticker])
                    current_position = get_position(ticker)
                    current_price = data_dict[ticker]['close'].iloc[-1]

                    if signal == 1 and current_position == 0:
                        affordable_shares = int(available_cash // current_price)
                        if affordable_shares > 0:
                            api.submit_order(symbol=ticker, qty=affordable_shares, side='buy', type='market', time_in_force='ioc')
                            entry_prices[ticker] = current_price
                            available_cash -= affordable_shares * current_price
                            print(f"Buy {ticker}: {affordable_shares} shares at {current_price:.2f}")
                    elif current_position > 0:
                        if entry_prices[ticker] is not None and check_exit(current_price, entry_prices[ticker]):
                            api.submit_order(symbol=ticker, qty=current_position, side='sell', type='market', time_in_force='ioc')
                            available_cash += current_position * current_price
                            print(f"Sell {ticker}: {current_position} shares at {current_price:.2f}")
                            entry_prices[ticker] = None

        time.sleep(0.5)

if __name__ == "__main__":
    trading_loop()
