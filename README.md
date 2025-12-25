# Automated Hourly Trading Bot 📈

![Build Status](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Alpaca%20Paper-yellow)

An automated algorithmic trading engine designed for the **Alpaca Paper Trading API**. This bot executes an hourly mean-reversion and trend-following strategy on **QQQ** (Nasdaq-100), utilizing a suite of advanced signal processing techniques and dynamic risk management.

## 🚀 Key Features

* **Multi-Factor Signal Generation:** Combines 9 distinct signals including:
    * **Kalman Filters:** For denoising price data and detecting true trend shifts.
    * **Yang-Zhang Volatility:** For robust variance estimation using Open-High-Low-Close data.
    * **Regime Switching:** Automatically detects Bull/Bear and High/Low Volatility regimes to adjust strategy behavior.
    * **Machine Learning Overlay:** Uses `LogisticRegression` (Scikit-Learn) to probability-weight trade signals.
* **Dynamic Risk Engine:**
    * **Volatility Targeting:** Automatically sizes positions to target 15% annualized volatility.
    * **Drawdown Control:** Reduces exposure (scales down) if the strategy experiences recent drawdowns.
* **Automated Execution:** Designed to run via **GitHub Actions** (CRON schedule) or a cloud server, rebalancing hourly between 10:00 AM and 4:00 PM ET.

## 🛠️ Technical Architecture

The system operates on a linear pipeline:
1.  **Ingestion:** Fetches 3 months of hourly OHLCV data via `AlpacaHistoricalDataClient`.
2.  **Feature Engineering:** Computes complex indicators (McGinley Dynamic, V4 Ensemble, Crisis Shield).
3.  **Signal Aggregation:** Normalizes signals and applies ML probability weights.
4.  **Position Sizing:** Calculates target dollar exposure based on Account Equity and Risk Scalars.
5.  **Execution:** Submits market orders to rebalance the portfolio to the target state.

## 📦 Installation & Usage

### Prerequisites
* Python 3.9+
* An Alpaca Paper Trading Account

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
