import time
import schedule
import numpy as np
from datetime import datetime, timedelta
from trading_env_hourly import HourlyTradingEnv
from stable_baselines3 import PPO
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from config.keys import ALPACA_API_KEY, ALPACA_SECRET_KEY, FINNHUB_KEY

# Initialize clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Load environment and model
print("Loading environment and model...")
env = HourlyTradingEnv(finnhub_key=FINNHUB_KEY)
model = PPO.load("hourly_trading_agent")
print("Ready to trade!")

# Tickers to trade
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM"]

def is_market_open():
	try:
		clock = trading_client.get_clock()
		return clock.is_open
	except:
		return False

def get_account():
	account = trading_client.get_account()
	portfolio_value = float(account.portfolio_value)
	actual_cash = float(account.cash)
	return portfolio_value, actual_cash

def get_position(ticker):
	try:
		position = trading_client.get_open_position(ticker)
		return int(position.qty)
	except:
		return 0

def get_live_price(ticker):
	try:
		request = StockLatestBarRequest(
			symbol_or_symbols=ticker,
			feed=DataFeed.IEX
		)
		bars = data_client.get_stock_latest_bar(request)
		return float(bars[ticker].close)
	except Exception as e:
		print(f"  Price error {ticker}: {e}")
		return None

def get_live_observation(ticker, portfolio_value, actual_cash):
	try:
		current_price = get_live_price(ticker)
		if current_price is None:
			return None

		end = datetime.now()
		start = end - timedelta(days=5)
		request = StockBarsRequest(
			symbol_or_symbols=ticker,
			timeframe=TimeFrame.Hour,
			start=start,
			end=end,
			feed=DataFeed.IEX
		)
		bars = data_client.get_stock_bars(request)
		df = bars.df

		if df is None or len(df) < 25:
			return None

		if hasattr(df.index, 'levels'):
			df = df.reset_index(level=0, drop=True)

		closes = df["close"]

		ma5 = closes.rolling(window=5).mean().iloc[-1]
		ma20 = closes.rolling(window=20).mean().iloc[-1]
		daily_return = closes.pct_change().iloc[-1]
		volume_change = df["volume"].pct_change().iloc[-1]

		delta = closes.diff()
		gain = delta.where(delta > 0, 0).rolling(window=14).mean()
		loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
		rs = gain / loss
		rsi = (100 - (100 / (1 + rs))).iloc[-1] / 100.0

		price_vs_ma5 = (current_price - ma5) / ma5 if ma5 > 0 else 0.0
		price_vs_ma20 = (current_price - ma20) / ma20 if ma20 > 0 else 0.0

		shares_held = get_position(ticker)
		shares_value = shares_held * current_price
		net_worth = portfolio_value
		portfolio_invested = shares_value / net_worth if net_worth > 0 else 0.0
		initial_balance = 100000.0
		portfolio_return = (net_worth - initial_balance) / initial_balance
		cash_ratio = actual_cash / initial_balance

		buy_price = 0.0
		unrealized_pnl = 0.0
		try:
			position = trading_client.get_open_position(ticker)
			buy_price = float(position.avg_entry_price)
			unrealized_pnl = (current_price - buy_price) / buy_price if buy_price > 0 else 0.0
		except:
			pass

		spy_price = get_live_price("SPY")
		spy_request = StockBarsRequest(
			symbol_or_symbols="SPY",
			timeframe=TimeFrame.Hour,
			start=start,
			end=end,
			feed=DataFeed.IEX
		)
		spy_bars = data_client.get_stock_bars(spy_request)
		spy_df = spy_bars.df
		if hasattr(spy_df.index, 'levels'):
			spy_df = spy_df.reset_index(level=0, drop=True)
		spy_closes = spy_df["close"]
		spy_ma20 = spy_closes.rolling(window=20).mean().iloc[-1]
		spy_ma50 = spy_closes.rolling(window=50).mean().iloc[-1]
		spy_regime = (spy_price - spy_ma50) / spy_ma50 if spy_ma50 > 0 else 0.0
		spy_trend = (spy_ma20 - spy_ma50) / spy_ma50 if spy_ma50 > 0 else 0.0

		env.verbose_sentiment = True
		sentiment = env.get_sentiment(ticker)

		obs = np.array([
			price_vs_ma5,
			price_vs_ma20,
			daily_return,
			volume_change,
			rsi,
			portfolio_invested,
			portfolio_return,
			float(buy_price > 0),
			cash_ratio,
			float(sentiment),
			spy_regime,
			spy_trend,
			unrealized_pnl
		], dtype=np.float32)

		obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
		print(f"  Price: ${current_price:.2f} | MA20: ${ma20:.2f} | RSI: {rsi*100:.1f} | SPY regime: {spy_regime:.3f}")
		return obs

	except Exception as e:
		print(f"  Observation error {ticker}: {e}")
		return None

def place_buy(ticker, qty):
	if qty <= 0:
		return
	try:
		order = MarketOrderRequest(
			symbol=ticker,
			qty=qty,
			side=OrderSide.BUY,
			time_in_force=TimeInForce.DAY
		)
		trading_client.submit_order(order)
		print(f"  BUY {qty} shares of {ticker}")
	except Exception as e:
		print(f"  BUY error {ticker}: {e}")

def place_sell(ticker, qty):
	if qty <= 0:
		return
	try:
		order = MarketOrderRequest(
			symbol=ticker,
			qty=qty,
			side=OrderSide.SELL,
			time_in_force=TimeInForce.DAY
		)
		trading_client.submit_order(order)
		print(f"  SELL {qty} shares of {ticker}")
	except Exception as e:
		print(f"  SELL error {ticker}: {e}")

def run_trading():
	now = datetime.now()
	print(f"\n{'='*50}")
	print(f"Trading cycle: {now.strftime('%Y-%m-%d %H:%M')}")
	print(f"{'='*50}")

	if not is_market_open():
		print("Market is closed - skipping cycle")
		return

	portfolio_value, actual_cash = get_account()
	print(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${actual_cash:,.2f}")

	for ticker in TICKERS:
		try:
			print(f"\n-- {ticker} --")

			obs = get_live_observation(ticker, portfolio_value, actual_cash)
			if obs is None:
				print(f"  Could not get observation for {ticker}, skipping")
				continue

			action, _ = model.predict(obs)
			action_name = ["Hold", "Buy", "Sell"][action]
			print(f"  Decision: {action_name}")

			current_shares = get_position(ticker)
			current_price = get_live_price(ticker)

			if action == 1 and current_shares == 0:
				budget = actual_cash / len(TICKERS)
				qty = int(budget // current_price)
				if qty > 0:
					place_buy(ticker, qty)

			elif action == 2 and current_shares > 0:
				place_sell(ticker, current_shares)

			else:
				print(f"  Holding {current_shares} shares")

		except Exception as e:
			print(f"  Error processing {ticker}: {e}")

	print(f"\nCycle complete at {datetime.now().strftime('%H:%M:%S')}")

def main():
	print("Paper trading bot started!")
	print(f"Trading: {TICKERS}")
	print("Runs every hour during market hours")

	run_trading()

	schedule.every().hour.at(":00").do(run_trading)

	while True:
		schedule.run_pending()
		time.sleep(60)

if __name__ == "__main__":
	main()
