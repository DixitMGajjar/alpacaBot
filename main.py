# PAIRS TRADING
"""
PAIRS TRADING
════════════════════════════════════════════════════════════════════════════
    ALPACA HOURLY TRADING BOT - HMMM4 UNIFIED MASTER STRATEGY (REVISED)
    
    Status: Production-Ready (Paper)
    - 1-hour candlestick data
    - Corrected "Overnight Gap" logic for hourly data
    - Live price fetching for execution
    - Intraday signal generation
════════════════════════════════════════════════════════════════════════════
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import warnings
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
API_KEY = os.environ.get('PAIRS_TRADING_API_KEY', 'YOUR_API_KEY_HERE')
API_SECRET = os.environ.get('PAIRS_TRADING_SECRET', 'YOUR_SECRET_KEY_HERE')
# ALPACA API CREDENTIALS (Paper Trading)


# Trading Parameters
SYMBOL = 'QQQ'
LOOKBACK_HOURS = 2520  # ~3 months

# Volatility & Risk Settings (Hourly Adjusted)
VOL_ANNUALIZATION_FACTOR = np.sqrt(252 * 6.5)  # 6.5 trading hours/day
VOL_PERCENTILE = 0.75
TARGET_VOL = 0.15      # 15% Target Vol
MAX_POSITION_SIZE = 0.90 

# Rebalance Schedule (ET)
REBALANCE_HOURS = [10, 11, 12, 13, 14, 15, 16]

# Indicator Windows (Hourly)
REGIME_WINDOW = 1260   
SLOW_MA_WINDOW = 1000  
FAST_MA_WINDOW = 250   

print("╔══════════════════════════════════════════════════════════════════╗")
print("║     ALPACA HOURLY TRADING BOT - HMMM4 STRATEGY (REVISED)        ║")
print("╚══════════════════════════════════════════════════════════════════╝\n")

# ═══════════════════════════════════════════════════════════════════
# CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

# ═══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════

def fetch_historical_data(symbol, hours=2520):
    print(f"\n[1] Fetching {hours} hours of data for {symbol}...")
    
    end_date = datetime.now()
    days_needed = int(hours / 6.5) + 14  # Buffer added
    start_date = end_date - timedelta(days=days_needed)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date,
        adjustment='all'
    )
    
    bars = data_client.get_stock_bars(request_params)
    df = bars.df
    
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=0, drop=True)
    
    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume'
    })
    
    # FIX: Widened to '09:00' to capture the opening candle (9:30-10:00)
    # which is often timestamped at the start of the hour (09:00)
    df = df.between_time('09:00', '16:00')
    
    print(f"   ✓ Fetched {len(df)} hourly bars")
    return df

def get_live_price(symbol):
    """Fetch real-time ask price for accurate execution"""
    try:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        res = data_client.get_stock_latest_quote(req)
        price = res[symbol].ask_price
        if price > 0:
            return price
        return res[symbol].bp  # Fallback to bid if ask is 0
    except Exception as e:
        print(f"   ! Warning: Could not get live price ({e})")
        return None

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def prepare_features(df):
    print("[2] Engineering features (hourly timeframe)...")
    
    # ------------------------------------------------------------------
    # FIX: CORRECT OVERNIGHT GAP CALCULATION
    # ------------------------------------------------------------------
    # We must calculate gap based on (Today Open - Yesterday Close)
    # not (Current Hour Open - Previous Hour Close)
    
    # 1. Map dates to rows
    df['Date'] = df.index.date
    
    # 2. Get the first Open of each day
    df['Day_Open'] = df.groupby('Date')['Open'].transform('first')
    
    # 3. Get the Close of the previous day
    # Create a mapping of Date -> Last Close
    daily_closes = df.groupby('Date')['Close'].last()
    # Map 'Date' in main df to yesterday's close
    df['Prev_Day_Close'] = df['Date'].map(daily_closes.shift(1))
    
    # 4. Calculate proper overnight gap
    df['Overnight_Gap'] = (df['Day_Open'] - df['Prev_Day_Close']) / df['Prev_Day_Close']
    
    # Fill NaN gaps (first day) with 0
    df['Overnight_Gap'] = df['Overnight_Gap'].fillna(0)
    # ------------------------------------------------------------------

    open_p, high, low, close = df['Open'], df['High'], df['Low'], df['Close']
    vol = df['Volume']
    
    # Returns
    df['Ret'] = close.pct_change()
    df['LogRet'] = np.log(close / close.shift(1))
    
    # Volatility
    df['Vol5'] = df['Ret'].rolling(5).std() * VOL_ANNUALIZATION_FACTOR
    df['Vol20'] = df['Ret'].rolling(20).std() * VOL_ANNUALIZATION_FACTOR
    df['Vol60'] = df['Ret'].rolling(60).std() * VOL_ANNUALIZATION_FACTOR
    
    # Yang-Zhang Vol
    v_over = np.log(open_p / close.shift(1)).rolling(20).var()
    v_oc = np.log(close / open_p).rolling(20).var()
    v_rs = ((np.log(high/close)*np.log(high/open_p)) + 
            (np.log(low/close)*np.log(low/open_p))).rolling(20).mean()
    df['Vol_YZ'] = np.sqrt(v_over + 0.34*v_oc + 0.66*v_rs) * VOL_ANNUALIZATION_FACTOR
    
    # Garman-Klass
    log_hl = np.log(high / low)
    log_co = np.log(close / open_p)
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df['GK_Vol'] = np.sqrt(gk_var.rolling(20).mean())
    
    # Moving Averages
    df['SMA200'] = close.rolling(SLOW_MA_WINDOW).mean()
    df['SMA50'] = close.rolling(FAST_MA_WINDOW).mean()
    df['SMA200_Slope'] = df['SMA200'].diff(5)
    
    # McGinley
    mg = [close.iloc[0]]
    for i in range(1, len(close)):
        prev, price = mg[-1], close.iloc[i]
        mg.append(prev + (price - prev) / (14 * (price/prev)**4))
    df['McGinley'] = pd.Series(mg, index=df.index)
    
    # Kalman Z-Score
    xhat, P = np.zeros(len(df)), np.ones(len(df))
    xhat[0], Q, R = close.iloc[0], 0.01, 1.0
    for k in range(1, len(df)):
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (close.iloc[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    kalman = pd.Series(xhat, index=df.index)
    dev = close - kalman
    df['Kalman_Z'] = (dev / dev.rolling(20).std()).clip(-5, 5).fillna(0)
    
    # Oscillators
    delta = close.diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = -delta.where(delta<0, 0).rolling(14).mean()
    df['RSI'] = 100 - (100/(1+gain/loss))
    df['IBS'] = (close - low) / (high - low + 1e-8)
    
    # CRSI
    df['RSI3'] = 100 - (100/(1 + delta.where(delta>0,0).rolling(3).mean()/
                                  -delta.where(delta<0,0).rolling(3).mean()))
    streak = np.sign(df['LogRet'])
    streak_groups = (streak != streak.shift()).cumsum()
    streak_count = streak.groupby(streak_groups).cumsum()
    df['Streak_RSI'] = 100 - (100/(1 + 
        streak_count.where(streak_count>0,0).rolling(2).mean()/
        -streak_count.where(streak_count<0,0).rolling(2).mean()))
    df['CRSI'] = (df['RSI3'] + df['Streak_RSI'].fillna(50) + 
                  df['Ret'].rolling(100).rank(pct=True)*100) / 3
    
    # MFI & CCI
    tp = (high + low + close) / 3
    mf = tp * vol
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf/(neg_mf + 1e-8)))
    
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * mad)
    
    # ADX
    tr = pd.concat([high-low, abs(high-close.shift(1)), 
                   abs(low-close.shift(1))], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    pdm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    ndm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean()
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean()
    df['ADX'] = (abs(pdi - ndi) / (pdi + ndi + 1e-8) * 100).ewm(alpha=1/14).mean()
    
    # WVF
    highest_c = close.rolling(130).max()
    df['WVF'] = ((highest_c - low) / highest_c) * 100
    df['WVF_Rank'] = df['WVF'].rolling(100).rank(pct=True)
    
    # ER
    change = close.diff(10).abs()
    path = close.diff().abs().rolling(10).sum()
    df['ER_10'] = change / (path + 1e-8)
    
    # Vortex
    df['VI_Plus'] = abs(high - low.shift(1)).rolling(14).sum() / tr.rolling(14).sum()
    df['VI_Minus'] = abs(low - high.shift(1)).rolling(14).sum() / tr.rolling(14).sum()
    
    # Dollar Flow
    df['DollarVal'] = close * vol
    df['Activity_Ratio'] = df['DollarVal'] / df['DollarVal'].rolling(20).mean()
    
    # Imbalance
    df['AbsImbal'] = abs(close - open_p) * vol
    imbal_agg = df['AbsImbal'].rolling(20).sum()
    imbal_ret = imbal_agg.pct_change()
    df['Imbal_Z'] = ((imbal_ret - imbal_ret.rolling(20).mean()) / 
                     imbal_ret.rolling(20).std()).fillna(0)
    
    # Chop
    df['Chop'] = 100 * np.log10(tr.rolling(14).sum() / 
                 (high.rolling(14).max() - low.rolling(14).min() + 1e-8)) / np.log10(14)
    
    # Drop NaNs after feature creation
    df = df.drop(columns=['Date', 'Day_Open', 'Prev_Day_Close']) # Clean up helpers
    return df.dropna()

# ═══════════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_signals(df):
    print("[3] Generating signals...")
    s = pd.DataFrame(index=df.index)
    
    bear = df['Close'] < df['SMA200']
    bull = ~bear
    vol_thresh = df['Vol20'].rolling(REGIME_WINDOW).quantile(VOL_PERCENTILE)
    high_vol = df['Vol20'] > vol_thresh
    low_vol = ~high_vol
    
    # 1. Var7_RegimeSwitch
    mcg_sig = np.where(df['Close'] > df['McGinley'], 1.5, -0.5)
    imbal_sig = np.where(df['Imbal_Z'] < -2, 1.5, 
                        np.where(df['Imbal_Z'] > 2, -1.0, 1.0))
    s['Var7_RegimeSwitch'] = np.where(df['VI_Plus'] > df['VI_Minus'], 
                                      mcg_sig, imbal_sig)
    
    # 2. The_Architect
    trend = np.where((df['Close'] > df['SMA200']) & (df['Vol20'] < 0.25), 0.5, 0.0)
    panic = np.where(df['RSI'] < 30, 1.0, 0.0)
    s['The_Architect'] = (trend + panic).clip(0, 1.5)
    
    # 3. S6_NoSeptFri
    s['S6_NoSeptFri'] = np.where((df['Vol20'] > 0.20) & (df['IBS'] < 0.15) & 
                                 (df.index.month != 9) & (df.index.dayofweek != 4), 
                                 1.5, 0.0)
    
    # 4. Sig_Kelly (ML)
    feats = ['RSI', 'Vol20', 'Chop', 'Vol_YZ', 'ADX', 'MFI', 'WVF_Rank', 'ER_10']
    s['Sig_Kelly'] = 0.0
    
    try:
        ml_df = df[feats + ['Ret']].copy()
        if len(ml_df) > 200:
            ml_df['Target'] = (ml_df['Ret'] > 0).astype(int)
            lr = LogisticRegression(max_iter=200, solver='lbfgs')
            train_size = int(len(ml_df) * 0.8)
            train = ml_df.iloc[:train_size]
            
            if train['Target'].nunique() > 1:
                mu, sd = train[feats].mean(), train[feats].std().replace(0, 1.0)
                lr.fit((train[feats] - mu) / sd, train['Target'])
                
                # Predict only for latest bar to save time in loop, or batch
                # Here we do batch for simplicity of code structure
                x_all = (ml_df[feats] - mu) / sd
                probs = lr.predict_proba(x_all)[:, 1]
                s['Sig_Kelly'] = ((3 * probs - 1) / 2) * 2
    except Exception as e:
        print(f"   ! ML Signal Error: {e}")

    # 5. Gap Fade (Corrected Logic)
    # Using the corrected 'Overnight_Gap' calculated in prepare_features
    s['Gap_Fade'] = np.where(high_vol & (df['Overnight_Gap'] < -0.01), 1.0,
                            np.where(high_vol & (df['Overnight_Gap'] > 0.01), -1.5, 0.0))

    # 6. Oscillator Ensemble
    score = (np.where(df['RSI']<30, 1, 0) + 
             np.where(df['MFI']<20, 1, 0) + 
             np.where(df['CCI']<-150, 1, 0))
    s['Oscillator_Ensemble'] = np.where(score >= 2, 1.5, 0.0)
    
    # 7. Vol Regime
    vol_ratio = df['Vol20'] / df['Vol60']
    vol_accel = df['Vol20'].diff() - df['Vol20'].diff().shift(5)
    expansion = (vol_ratio > vol_ratio.rolling(REGIME_WINDOW).quantile(0.80)) & (vol_accel > 0)
    compression = (vol_ratio < vol_ratio.rolling(REGIME_WINDOW).quantile(0.20)) & bull & (vol_accel < 0)
    normalized = (vol_ratio > 0.9) & (vol_ratio < 1.1) & bull & low_vol
    s['Vol_Regime_Switch'] = np.select([expansion, compression, normalized],
                                       [-1.0, 1.5, 1.0], default=0.0)
    
    # 8. V4 Ensemble
    sig_vol = pd.Series(1.0, index=df.index)
    sig_vol[df['Vol20'] < 0.20] = 1.5
    mask_h = df['Vol20'] > 0.30
    accel = df['Vol5'] > df['Vol20']
    sig_vol[mask_h & accel] = -0.5
    sig_vol[mask_h & ~accel] = 0.5
    
    sig_dol = pd.Series(0.5, index=df.index)
    act = df['Activity_Ratio'] > 1.2
    trend = df['Close'] > df['Close'].shift(20)
    sig_dol[act & trend] = 1.5
    sig_dol[act & ~trend] = -1.0
    
    sig_tr = pd.Series(0.0, index=df.index)
    sig_tr[df['SMA200_Slope'] > 0] = 1.5
    
    s['V4_Ensemble1'] = ((sig_vol + sig_dol + sig_tr)/3).clip(-1.0, 1.5)
    
    # 9. Crisis Shield
    vol_thresh_e5 = df['Vol20'].rolling(REGIME_WINDOW).quantile(0.60)
    high_vol_e5 = df['Vol20'] > vol_thresh_e5
    vol_rising = df['Vol5'] > df['Vol20']
    
    sig_e5 = pd.Series(0.0, index=df.index)
    sig_e5[bull] = 0.8
    sig_e5[bear & high_vol_e5] = -1.2
    sig_e5[bear & high_vol_e5 & vol_rising] = -1.5
    s['Crisis_Shield'] = sig_e5
    
    # Crash Guard
    crash = bear & high_vol
    toxic = ['The_Architect', 'S6_NoSeptFri', 'Oscillator_Ensemble', 
             'Gap_Fade', 'Sig_Kelly', 'V4_Ensemble1']
    for strat in toxic:
        if strat in s.columns:
            s[strat] = np.where(crash, 0.0, s[strat])
            
    return s.fillna(0)

# ═══════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════

def load_or_create_weights(signals_df):
    try:
        with open('unified_master_weights.pkl', 'rb') as f:
            saved = pickle.load(f)
            weights = pd.Series(saved['weights'])
            print(f"   ✓ Loaded saved weights (Train Sharpe: {saved.get('train_sharpe', 0):.2f})")
            return weights
    except:
        print("   ! No saved weights found, using equal weighting")
        weights = pd.Series(1/len(signals_df.columns), index=signals_df.columns)
        return weights

def calculate_position_size(df, signals, weights, account_value):
    print("[4] Calculating position size...")
    
    latest_idx = df.index[-1]
    port_signal = (signals.loc[latest_idx] * weights).sum()
    
    # Risk Overlays (Shifted to ensure no lookahead)
    sma200 = df['Close'].rolling(SLOW_MA_WINDOW).mean()
    vol20 = df['Ret'].rolling(20).std() * VOL_ANNUALIZATION_FACTOR
    vol_thresh = vol20.rolling(REGIME_WINDOW).quantile(VOL_PERCENTILE)
    
    # Use previous bar for decisions
    is_bull = df['Close'].iloc[-2] > sma200.iloc[-2]
    is_high_vol = vol20.iloc[-2] > vol_thresh.iloc[-2]
    
    # Regime Scalar
    if not is_bull and not is_high_vol:
        regime_scalar = 0.0
    elif not is_bull and is_high_vol:
        regime_scalar = 0.6
    else:
        regime_scalar = 1.0
    
    # Vol Scalar
    port_returns = (signals.shift(1) * df['Ret']).sum(axis=1).dropna()
    if len(port_returns) >= 20:
        real_vol = port_returns.iloc[-20:].std() * VOL_ANNUALIZATION_FACTOR
        vol_scalar = min(1.5, max(0.5, TARGET_VOL / (real_vol + 1e-6)))
    else:
        vol_scalar = 1.0
    
    # Drawdown Scalar (20-period lookback for hourly)
    cum_ret = (1 + port_returns).cumprod()
    dd = (cum_ret / cum_ret.cummax() - 1).iloc[-1]
    if dd < -0.15: dd_scalar = 0.5
    elif dd < -0.10: dd_scalar = 0.7
    elif dd < -0.05: dd_scalar = 0.9
    else: dd_scalar = 1.0
    
    final_exposure = port_signal * regime_scalar * vol_scalar * dd_scalar
    final_exposure = np.clip(final_exposure, -1.5, 1.5)
    
    target_position_pct = final_exposure * MAX_POSITION_SIZE
    target_position_value = account_value * target_position_pct
    
    print(f"   Signal: {port_signal:.2f} | Regime: {regime_scalar:.1f} | VolScale: {vol_scalar:.1f}")
    print(f"   Target Exposure: {final_exposure:.2f} (${target_position_value:,.2f})")
    
    return target_position_value, final_exposure

# ═══════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════

def execute_rebalance(symbol, target_value):
    print(f"\n[5] Executing rebalance for {symbol}...")
    
    # 1. Get Live Price for accurate share calculation
    live_price = get_live_price(symbol)
    if live_price is None:
        print("   ❌ Failed to get live price. Aborting trade.")
        return

    # 2. Get Current Position
    try:
        position = trading_client.get_open_position(symbol)
        current_qty = float(position.qty)
        current_value = float(position.market_value)
    except:
        current_qty = 0.0
        current_value = 0.0
        
    # 3. Calculate Delta
    target_qty = int(target_value / live_price)
    delta_qty = target_qty - current_qty
    
    print(f"   Live Price: ${live_price:.2f}")
    print(f"   Current: {current_qty} shares | Target: {target_qty} shares")
    print(f"   Delta: {delta_qty} shares")
    
    if abs(delta_qty) < 1:
        print("   → Delta too small, skipping.")
        return
        
    # 4. Submit Order
    side = OrderSide.BUY if delta_qty > 0 else OrderSide.SELL
    order_request = MarketOrderRequest(
        symbol=symbol,
        qty=abs(delta_qty),
        side=side,
        time_in_force=TimeInForce.DAY
    )
    
    try:
        order = trading_client.submit_order(order_request)
        print(f"   ✓ Order Submitted: {side.value} {abs(delta_qty)} shares")
    except Exception as e:
        print(f"   ❌ Order Failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def should_run_now():
    """Check if we are in a rebalance hour window"""
    now = datetime.now()
    if now.weekday() >= 5: return False # No weekends
    
    # Run if current hour is in schedule and we are in first 5 mins
    if now.hour in REBALANCE_HOURS and now.minute < 5:
        return True
    return False

def run_hourly_strategy():
    print(f"\n{'='*70}")
    print(f"Running HOURLY strategy at {datetime.now()}")
    print(f"{'='*70}\n")
    
    try:
        # Check Account
        account = trading_client.get_account()
        if float(account.equity) < 25000:
            print("! WARNING: Equity < $25k. Pattern Day Trading restrictions may apply.")
            
        # Run Pipeline
        df = fetch_historical_data(SYMBOL, LOOKBACK_HOURS)
        df = prepare_features(df)
        signals = generate_signals(df)
        weights = load_or_create_weights(signals)
        
        target_value, exposure = calculate_position_size(df, signals, weights, float(account.portfolio_value))
        
        # Execute
        if abs(exposure) > 0.05:
            execute_rebalance(SYMBOL, target_value)
        else:
            print("\n[5] Signal weak/neutral. Closing positions if any.")
            execute_rebalance(SYMBOL, 0) # Close out if signal is flat
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

# if __name__ == "__main__":
#     print("Hourly Bot started.")
#     print(f"Monitoring for hours: {REBALANCE_HOURS} ET")
    
#     last_run_hour = None
    
#     while True:
#         now = datetime.now()
#         current_hour_sig = (now.date(), now.hour)
        
#         if should_run_now() and current_hour_sig != last_run_hour:
#             run_hourly_strategy()
#             last_run_hour = current_hour_sig
        
#         time.sleep(60) # Check every minute

if __name__ == "__main__":
    print("--- Starting Hourly Bot Execution ---")
    
    # Check if market is open (0=Mon ... 4=Fri)
    if datetime.now().weekday() > 4:
        print("Today is Weekend. No trading.")
    else:
        try:
            # RUN ONCE and EXIT
            run_hourly_strategy()
        except Exception as e:
            print(f"❌ Critical Error: {e}")
            # This exit code tells GitHub to send you a failure email
            exit(1)
