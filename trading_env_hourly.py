import gymnasium as gym
import numpy as np
import yfinance as yf
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import finnhub
from datetime import datetime, timedelta

np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

def calculate_rsi(series, period=14):
	delta = series.diff()
	gain = delta.where(delta > 0, 0).rolling(window=period).mean()
	loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
	rs = gain / loss
	return 100 - (100 / (1 + rs))

class HourlyTradingEnv(gym.Env):
	def __init__(self, start="2024-05-01", end="2026-04-01", finnhub_key=None, verbose_sentiment=False):
		super(HourlyTradingEnv, self).__init__()

		self.trade_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM"]
		self.all_tickers = self.trade_tickers + ["SPY"]
		self.start = start
		self.end = end
		self.verbose_sentiment = verbose_sentiment

		self.finnhub_client = finnhub.Client(api_key=finnhub_key) if finnhub_key else None

		print("Loading FinBERT model...")
		self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
		self.finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
		self.finbert.eval()
		print("FinBERT loaded!")

		print("Loading all stock data...")
		self.stock_data = {}
		for ticker in self.all_tickers:
			self.stock_data[ticker] = self._load_data(ticker)
		print("All stocks loaded!")

		print("Loading SPY market regime data...")
		self.spy_data = self._load_spy_regime()
		print("Market regime data loaded!")

		# 5 actions — absolute portfolio targeting
		self.action_space = gym.spaces.Discrete(5)

		# 13 observations
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
		)

		self.initial_balance = 10000
		self.stop_loss_pct = 0.04
		self.max_position_pct = 0.50

		self.reset()

	def _load_data(self, ticker):
		df = yf.download(
			ticker,
			start=self.start,
			end=self.end,
			interval="1h",
			auto_adjust=True
		)
		df = df[["Close", "Volume"]].dropna()
		df["MA20"] = df["Close"].rolling(window=20).mean()
		df["Daily_Return"] = df["Close"].pct_change()
		df["Volume_Change"] = df["Volume"].pct_change()
		df["RSI"] = calculate_rsi(df["Close"].squeeze())
		df = df.dropna()
		if len(df) == 0:
			raise ValueError(f"No data returned for {ticker} - check date range")
		return df

	def _load_spy_regime(self):
		spy = yf.download(
			"SPY",
			start=self.start,
			end=self.end,
			interval="1h",
			auto_adjust=True
		)
		spy = spy[["Close"]].dropna()
		spy.columns = ["Close"]
		close = spy["Close"]
		spy["MA20"] = close.rolling(window=20).mean()
		spy["MA50"] = close.rolling(window=50).mean()
		spy["Regime"] = (close - spy["MA50"]) / spy["MA50"]
		spy["Trend"] = (spy["MA20"] - spy["MA50"]) / spy["MA50"]
		spy = spy.dropna()
		return spy

	def get_sentiment(self, ticker, timestamp=None):
		if self.finnhub_client is None:
			return 0.0
		try:
			if timestamp:
				end_date = timestamp
				start_date = timestamp - timedelta(hours=24)
			else:
				end_date = datetime.now()
				start_date = end_date - timedelta(hours=24)

			news = self.finnhub_client.company_news(
				ticker,
				_from=start_date.strftime("%Y-%m-%d"),
				to=end_date.strftime("%Y-%m-%d")
			)

			if not news:
				return 0.0

			sentiments = []
			for article in news[:5]:
				headline = article.get("headline", "")
				if not headline:
					continue
				inputs = self.tokenizer(
					headline,
					return_tensors="pt",
					truncation=True,
					max_length=512
				)
				with torch.no_grad():
					outputs = self.finbert(**inputs)
				scores = torch.softmax(outputs.logits, dim=1)
				sentiment = scores[0][0].item() - scores[0][2].item()
				sentiments.append(sentiment)
				if self.verbose_sentiment:
					source = article.get("source", "unknown")
					print(f"  [{ticker}] {source}: \"{headline[:80]}\" -> {sentiment:.3f}")

			if sentiments:
				avg_sentiment = np.mean(sentiments)
				if self.verbose_sentiment:
					print(f"  [{ticker}] Average sentiment: {avg_sentiment:.3f}")
				return float(avg_sentiment)

		except Exception as e:
			if self.verbose_sentiment:
				print(f"  [{ticker}] Sentiment error: {e}")
		return 0.0

	def _get_spy_regime(self, current_step):
		try:
			if len(self.spy_data) == 0:
				return 0.0, 0.0
			idx = min(current_step, len(self.spy_data) - 1)
			row = self.spy_data.iloc[idx]
			regime = float(row["Regime"].iloc[0]) if hasattr(row["Regime"], 'iloc') else float(row["Regime"])
			trend = float(row["Trend"].iloc[0]) if hasattr(row["Trend"], 'iloc') else float(row["Trend"])
			return regime, trend
		except:
			return 0.0, 0.0

	def _get_price(self, ticker, step):
		df = self.stock_data[ticker]
		idx = min(step, len(df) - 1)
		val = df.iloc[idx]["Close"]
		return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val)

	def _get_portfolio_value(self):
		total = self.balance
		for ticker in self.trade_tickers:
			shares = self.positions.get(ticker, 0)
			if shares > 0:
				price = self._get_price(ticker, self.current_hour)
				total += shares * price
		return total

	def _get_portfolio_stats(self):
		net_worth = self._get_portfolio_value()
		open_positions = sum(1 for t in self.trade_tickers if self.positions.get(t, 0) > 0)
		max_concentration = 0.0
		for ticker in self.trade_tickers:
			shares = self.positions.get(ticker, 0)
			if shares > 0 and net_worth > 0:
				price = self._get_price(ticker, self.current_hour)
				pct = (shares * price) / net_worth
				max_concentration = max(max_concentration, pct)
		return net_worth, open_positions, max_concentration

	def reset(self, seed=None):
		self.current_hour = 0
		self.current_ticker_idx = 0
		self.balance = self.initial_balance
		self.net_worth = self.initial_balance
		self.prev_net_worth = self.initial_balance

		# Portfolio tracking across all tickers
		self.positions = {t: 0 for t in self.trade_tickers}
		self.cost_basis = {t: 0.0 for t in self.trade_tickers}
		self.avg_entry = {t: 0.0 for t in self.trade_tickers}
		self.current_sentiment = {t: 0.0 for t in self.trade_tickers}

		# Find valid starting hour where all tickers have data
		max_start = min(len(self.stock_data[t]) for t in self.all_tickers) - 100
		self.current_hour = np.random.randint(0, max(1, max_start // 2))

		return self._get_observation(), {}

	def _get_observation(self):
		ticker = self.trade_tickers[self.current_ticker_idx]
		df = self.stock_data[ticker]
		idx = min(self.current_hour, len(df) - 1)
		row = df.iloc[idx]

		current_price = float(row["Close"].iloc[0]) if hasattr(row["Close"], 'iloc') else float(row["Close"])
		ma20 = float(row["MA20"].iloc[0]) if hasattr(row["MA20"], 'iloc') else float(row["MA20"])
		daily_return = float(row["Daily_Return"].iloc[0]) if hasattr(row["Daily_Return"], 'iloc') else float(row["Daily_Return"])
		volume_change = float(row["Volume_Change"].iloc[0]) if hasattr(row["Volume_Change"], 'iloc') else float(row["Volume_Change"])
		rsi_normalized = float(row["RSI"].iloc[0]) / 100.0 if hasattr(row["RSI"], 'iloc') else float(row["RSI"]) / 100.0

		price_vs_ma20 = (current_price - ma20) / ma20 if ma20 > 0 else 0.0

		# Position info for current ticker
		shares_held = self.positions.get(ticker, 0)
		position_value = shares_held * current_price
		net_worth, open_positions, portfolio_concentration = self._get_portfolio_stats()
		current_position_pct = position_value / net_worth if net_worth > 0 else 0.0

		# Portfolio level
		portfolio_return = (net_worth - self.initial_balance) / self.initial_balance
		available_cash_ratio = self.balance / net_worth if net_worth > 0 else 1.0

		# SPY regime
		spy_regime, spy_trend = self._get_spy_regime(self.current_hour)

		# Unrealized PnL
		unrealized_pnl = 0.0
		avg_entry = self.avg_entry.get(ticker, 0.0)
		if shares_held > 0 and avg_entry > 0:
			unrealized_pnl = (current_price - avg_entry) / avg_entry

		# Normalize portfolio stats
		total_positions_normalized = open_positions / len(self.trade_tickers)

		obs = np.array([
			price_vs_ma20,
			daily_return,
			volume_change,
			rsi_normalized,
			current_position_pct,
			portfolio_return,
			available_cash_ratio,
			float(self.current_sentiment.get(ticker, 0.0)),
			spy_regime,
			spy_trend,
			unrealized_pnl,
			total_positions_normalized,
			portfolio_concentration,
		], dtype=np.float32)

		obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
		return obs

	def _execute_target(self, ticker, target_pct, current_price):
		target_value = self.net_worth * target_pct
		current_shares = self.positions.get(ticker, 0)
		current_value = current_shares * current_price
		difference = target_value - current_value

		if difference > 0:
			# Check SPY regime guard
			spy_regime, _ = self._get_spy_regime(self.current_hour)
			if spy_regime < -0.03:
				return

			shares_to_buy = int(difference // current_price)
			cost = shares_to_buy * current_price

			if shares_to_buy > 0 and cost <= self.balance:
				self.positions[ticker] = current_shares + shares_to_buy
				self.balance -= cost
				old_basis = self.cost_basis.get(ticker, 0.0)
				new_basis = old_basis + cost
				self.cost_basis[ticker] = new_basis
				total_shares = self.positions[ticker]
				self.avg_entry[ticker] = new_basis / total_shares if total_shares > 0 else 0.0

		elif difference < 0:
			shares_to_sell = int(abs(difference) // current_price)
			shares_to_sell = min(shares_to_sell, current_shares)

			if shares_to_sell > 0:
				proceeds = shares_to_sell * current_price
				self.balance += proceeds
				new_shares = current_shares - shares_to_sell
				self.positions[ticker] = new_shares

				if current_shares > 0:
					sold_ratio = shares_to_sell / current_shares
					self.cost_basis[ticker] *= (1 - sold_ratio)

				if new_shares == 0:
					self.avg_entry[ticker] = 0.0
					self.cost_basis[ticker] = 0.0

	def _check_stop_loss(self, ticker, current_price):
		shares_held = self.positions.get(ticker, 0)
		avg_entry = self.avg_entry.get(ticker, 0.0)

		if shares_held > 0 and avg_entry > 0:
			loss_pct = (current_price - avg_entry) / avg_entry
			if loss_pct <= -self.stop_loss_pct:
				proceeds = shares_held * current_price
				self.balance += proceeds
				self.positions[ticker] = 0
				self.avg_entry[ticker] = 0.0
				self.cost_basis[ticker] = 0.0
				return True
		return False

	def step(self, action):
		ticker = self.trade_tickers[self.current_ticker_idx]
		current_price = self._get_price(ticker, self.current_hour)

		# Check stop loss first — environmental constraint
		stop_loss_triggered = self._check_stop_loss(ticker, current_price)

		if not stop_loss_triggered:
			# Execute action
			# 0: Target 0%  — exit
			# 1: Target 10% — small
			# 2: Target 25% — medium
			# 3: Target 50% — large
			# 4: Hold
			target_pcts = [0.0, 0.10, 0.25, 0.50, None]
			target = target_pcts[action]
			if target is not None:
				self._execute_target(ticker, target, current_price)

		# Advance ticker index
		self.current_ticker_idx += 1

		# If all tickers processed this hour — advance to next hour
		if self.current_ticker_idx >= len(self.trade_tickers):
			self.current_ticker_idx = 0
			self.current_hour += 1

		# Check episode done
		min_data_len = min(len(self.stock_data[t]) for t in self.trade_tickers)
		done = self.current_hour >= min_data_len - 1

		# Calculate net worth
		self.net_worth = self._get_portfolio_value()

		# Reward — only calculated when full hour is complete
		if self.current_ticker_idx == 0:
			reward = (self.net_worth - self.prev_net_worth) / self.initial_balance
			self.prev_net_worth = self.net_worth
		else:
			reward = 0.0

		return self._get_observation(), reward, done, False, {}
