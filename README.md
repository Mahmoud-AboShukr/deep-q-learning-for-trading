# Deep Q-Learning for Stock Trading

This repository presents a **from-scratch Deep Q-Network (DQN)** project for **single-asset trading** using historical market data.  
The agent is trained to interact with a simplified trading environment built on top of **S&P 500 index data** retrieved from Yahoo Finance, and learns to choose among three actions: **hold, buy, and sell**.

Unlike many reinforcement learning projects that rely heavily on high-level frameworks, this project implements the core DQN components manually with NumPy, making the training mechanics more transparent and educational.

## Project Overview

The notebook builds a reinforcement learning pipeline for algorithmic trading with the following components:

- historical data acquisition using `yfinance`
- state construction from rolling market returns
- portfolio simulation with transaction costs
- discrete trading actions: hold / buy / sell
- reward calculation based on realized profit
- a manually implemented neural network Q-function
- epsilon-greedy exploration
- experience replay
- target-network style synchronization
- training and performance visualization

The project is designed as a compact experimental environment for studying how value-based reinforcement learning can be applied to sequential financial decision-making.

## Main Features

- Download and preprocess historical **S&P 500** data
- Build a simple RL trading environment from scratch
- Define a state using a lookback window of returns
- Simulate cash balance, held shares, and transaction costs
- Train a DQN agent with:
  - feedforward neural network
  - replay memory
  - epsilon-greedy exploration
  - temporal-difference learning
  - periodic target updates
- Evaluate trading performance over time
- Visualize portfolio value and learning behavior

## Repository Structure

```text
dqn-stock-trading-agent/
├── code.ipynb
├── README.md
└── requirements.txt
```

## Methodology

### 1. Market Data
The notebook uses `yfinance` to download historical price data for the **S&P 500 index (`^GSPC`)**.

### 2. State Representation
A rolling window of past market returns is used to represent the trading state seen by the agent.

### 3. Action Space
The agent chooses one of three actions at each step:
- **0 — Hold**
- **1 — Buy**
- **2 — Sell**

### 4. Portfolio Simulation
The environment tracks:
- current cash balance,
- quantity of asset held,
- transaction commission,
- total portfolio value over time.

### 5. Reward Design
Rewards are linked to realized trading profit, allowing the agent to improve its policy through reinforcement learning.

### 6. Deep Q-Learning
The project implements DQN components manually, including:
- forward pass through a small multilayer perceptron,
- backpropagation updates,
- replay memory sampling,
- Bellman target computation,
- target-network parameter updates.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For notebook execution:

```bash
pip install notebook
jupyter notebook
```

## Usage

Run the notebook:

```bash
jupyter notebook code.ipynb
```

The notebook workflow includes:
1. downloading market data,
2. defining the state and reward structure,
3. initializing the DQN,
4. training the trading agent,
5. evaluating the learned strategy,
6. plotting portfolio behavior and results.

## Dependencies

The project mainly relies on:
- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`

## Notes

- This project is a simplified educational trading environment, not a production trading system.
- The agent is trained on historical data and does **not** account for many real-world market constraints such as slippage, liquidity, risk limits, or multi-asset portfolio dynamics.
- Since Yahoo Finance data can change slightly over time, results may vary across runs.
- For a cleaner public repository, avoid committing large output images or checkpoint files unless you want to share experiment artifacts.

## Possible Improvements

- replace the manual NumPy DQN with a PyTorch implementation,
- extend the environment to multiple assets,
- add better reward shaping and risk-adjusted metrics,
- benchmark against buy-and-hold,
- use train/validation/test temporal splits more explicitly,
- integrate Gymnasium-compatible environment wrappers.

## License

This repository is shared as part of a personal portfolio in reinforcement learning and quantitative finance experimentation.
