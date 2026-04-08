from trading_env_hourly import HourlyTradingEnv
from stable_baselines3 import PPO

# Load with UNSEEN 2026 data only
env = HourlyTradingEnv(start="2026-01-01", end="2026-04-01")
model = PPO.load("hourly_trading_agent")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "SPY"]

print(f"\nUNSEEN DATA TEST — Jan 2026 to Apr 2026")
print(f"\n{'Stock':<8} {'Start':>10} {'End':>12} {'Profit/Loss':>12} {'Return':>10}")
print("-" * 55)

for ticker in tickers:
	env.df = env.stock_data[ticker]
	obs, _ = env.reset(seed=42)
	done = False

	while not done:
		action, _ = model.predict(obs)
		obs, reward, done, truncated, info = env.step(action)

	final_worth = env.net_worth
	profit = final_worth - env.initial_balance
	returns = (profit / env.initial_balance) * 100

	print(f"{ticker:<8} ${env.initial_balance:>9,.2f} ${final_worth:>11,.2f} ${profit:>11,.2f} {returns:>9.1f}%")

print("-" * 55)
print("\nNote: This is data the agent has NEVER seen before!")
