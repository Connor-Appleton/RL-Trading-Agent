import os
import sys
import numpy as np
from datetime import datetime
from trading_env_hourly import HourlyTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from config.keys import FINNHUB_KEY

class Logger(object):
	def __init__(self, filename):
		self.terminal = sys.stdout
		self.log = open(filename, "a")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.terminal.flush()
		self.log.flush()

	def isatty(self):
		return False

os.makedirs("./training_logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
log_filename = f"./training_logs/hourly_training_{timestamp}.txt"
sys.stdout = Logger(log_filename)
print(f"Logging to: {log_filename}")

np.set_printoptions(suppress=True, formatter={'float_kind':'{:.2f}'.format})

env = HourlyTradingEnv(finnhub_key=FINNHUB_KEY)

if os.path.exists("hourly_trading_agent.zip"):
	print("Loading existing model...")
	model = PPO.load("hourly_trading_agent", env=env)
	model.learning_rate = lambda progress: max(0.00001, 0.0001 * progress)
	model.clip_range = lambda _: 0.1
else:
	print("Creating new model...")
	policy_kwargs = dict(
		net_arch=[256, 256, 128]
	)
	model = PPO(
		"MlpPolicy",
		env,
		verbose=1,
		ent_coef=0.01,
		learning_rate=lambda progress: max(0.00001, 0.0001 * progress),
		clip_range=0.1,
		policy_kwargs=policy_kwargs
	)

new_logger = configure("./tensorboard_logs_hourly", ["stdout", "tensorboard"])
model.set_logger(new_logger)

print("Training started...")
model.learn(total_timesteps=5000000, reset_num_timesteps=False)
model.save("hourly_trading_agent")
print("Training complete! Model saved.")
