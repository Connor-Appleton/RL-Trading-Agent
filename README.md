# RL Trading Agent

A reinforcement learning stock trading agent built from scratch using Proximal Policy Optimization (PPO). The agent learns to manage a portfolio of 6 stocks using live market data, NLP sentiment analysis, and market regime detection — deployed to a live paper trading account via Alpaca.

## Overview

This project implements a complete end-to-end ML trading pipeline including data ingestion, model training, backtesting, and live deployment. The agent makes hourly trading decisions across AAPL, MSFT, GOOGL, AMZN, TSLA, and JPM — sizing positions based on market conditions, sentiment, and portfolio state.

Built entirely on consumer hardware (ThinkPad T14 Gen 2, no GPU) using a Proxmox home lab with two Ubuntu Server VMs.

## Key Features

- **PPO Training** — Proximal Policy Optimization via Stable-Baselines3 with decaying learning rate to prevent catastrophic forgetting
- **Position Sizing** — 5-action absolute portfolio targeting (0%, 10%, 25%, 50%, Hold) replacing binary buy/sell decisions
- **Multi-Ticker Episodes** — agent manages all 6 stocks simultaneously per episode, experiencing real portfolio dynamics during training
- **FinBERT Sentiment** — ProsusAI/finbert scores live Finnhub news headlines to produce per-ticker sentiment observations
- **Market Regime Detection** — SPY moving average crossover identifies bull/bear conditions, gates buying behavior
- **Portfolio-Level Observations** — agent sees open position count, concentration risk, and available cash at every decision
- **Hard Stop Loss** — 4% environmental constraint, not a reward signal — protects capital and training stability
- **Live Paper Trading** — Alpaca IEX feed provides real-time prices, decisions logged to CSV with full observation audit trail

## Architecture

### Training Environment (`trading_env_hourly.py`)
Custom Gymnasium environment. Each episode spans ~3,300 hours of market data. Every hour the agent makes sequential decisions across all 6 tickers with full portfolio state awareness between decisions.

**Observation Space (13 features):**
- `price_vs_ma20` — price relative to 20-hour moving average
- `daily_return` — hourly price change
- `volume_change` — volume relative to previous hour
- `rsi_normalized` — RSI indicator normalized 0-1
- `current_position_pct` — current ticker allocation as % of portfolio
- `portfolio_return` — overall account return since episode start
- `available_cash_ratio` — liquid cash as % of portfolio
- `finbert_sentiment` — FinBERT news sentiment score (-1 to +1)
- `spy_regime` — SPY distance from 50-hour MA (bull/bear indicator)
- `spy_trend` — SPY MA20 vs MA50 momentum direction
- `unrealized_pnl` — current position profit/loss
- `total_positions_open` — number of stocks currently held (normalized)
- `portfolio_concentration` — largest single position as % of portfolio

**Action Space (5 discrete actions):**
- Target 0% — exit position entirely
- Target 10% — small/exploratory position
- Target 25% — medium conviction
- Target 50% — high conviction (hard cap)
- Hold — maintain current allocation

### Training Pipeline (`train_hourly.py`)
PPO with decaying learning rate (`lambda progress: max(0.00001, 0.0001 * progress)`). Solved catastrophic forgetting between 5M-10M step runs that caused reward decline with fixed learning rates. Network architecture: `[256, 256, 128]`.

### Paper Trading Bot (`paper_trade.py`)
Lightweight inference-only deployment. Pulls live hourly bars from Alpaca IEX feed, calculates all 13 observations in real time, runs FinBERT on current Finnhub headlines, executes market orders. Logs every decision — including holds — to `trade_log.csv` with full observation state for analysis.

## Infrastructure

- **Hypervisor:** Proxmox VE on ThinkPad T14 Gen 2
- **Training VM:** Ubuntu 22.04 Server (hourly-trading-lab) — 4 cores, 8GB RAM
- **Deployment VM:** Ubuntu 22.04 Server (trading-lab-01) — 4 cores, 8GB RAM
- **Training Speed:** ~280-370 fps on CPU
- **Data Source:** Yahoo Finance (historical), Alpaca IEX (live)
- **Sentiment:** Finnhub news API + ProsusAI/finbert

## Training History

| Run | Total Steps | Final Reward | Key Change |
|-----|-------------|--------------|------------|
| Baseline | 6M | 0.319 | Initial build |
| + Market Regime | 5M | 1.73 | SPY regime observations added |
| Fixed LR runs | 10M | 1.15-1.36 | Reward declined — catastrophic forgetting |
| + Decaying LR | 10M | 1.86 | Solved forgetting, reward climbed |
| Extended | 20M | 2.12 | Continued improvement |
| Full run | 35M | 2.12-2.14 | Plateau — ceiling reached for this architecture |
| + Position Sizing | In progress | TBD | New architecture with portfolio targeting |

## Results

Tested on unseen Jan-Apr 2026 data (5-run average) during a genuine bear market driven by Middle East conflict, oil price spikes, and elevated VIX:

| Stock | Agent Return | Market Return | Outperformance |
|-------|-------------|---------------|----------------|
| AAPL | -3.1% | ~-10% | +6.9% |
| MSFT | -1.5% | ~-12% | +10.5% |
| GOOGL | -7.6% | ~-8% | Roughly even |
| AMZN | -15.1% | ~-15% | Roughly even |
| TSLA | -2.0% | ~-15% | +13.0% |
| JPM | -4.7% | ~-17% | +12.3% |
| SPY | -5.3% | ~-7% | +1.7% |

Agent beats buy-and-hold on 5/7 stocks in a bear market environment.

## Current Status

- Binary buy/sell model (35M steps) deployed to live Alpaca paper trading account
- Position sizing model training in progress — multi-ticker episode architecture
- Trade decisions logged hourly to CSV for analysis and future retraining

## Stack
Python 3.10
stable-baselines3
gymnasium
torch
transformers (ProsusAI/finbert)
yfinance
alpaca-py
finnhub-python
pandas/numpy

## What I Learned

This project required working through several non-obvious RL challenges:

- **Catastrophic forgetting** — fixed learning rate caused reward decline after 5M steps. Solved with linear decay schedule.
- **Credit assignment** — binary buy/sell gave no signal about position sizing quality. Solved with absolute portfolio targeting.
- **Action aliasing** — percentage-of-cash sizing produced inconsistent outcomes depending on current position. Solved by switching to absolute portfolio percentage targets.
- **Reward hacking risk** — shaped rewards can be exploited without achieving the underlying goal. Kept reward function as pure net worth change.
- **Environmental constraints vs reward signals** — stop loss implemented as hard constraint rather than penalty, protecting both capital and training stability.

---

*Built by Connor Appleton — Fort Smith, AR*
*Active development — updated regularly*
