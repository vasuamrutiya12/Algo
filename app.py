# ==========================================================
# AI BASED NIFTY INTRADAY PAPER TRADING (KITE CONNECT)
# ==========================================================

from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from datetime import datetime, timedelta
import time

# ===================== CONFIG =====================
API_KEY = "cdtkozma3tyjs6rc"
API_SECRET = "mk8sfb5yxtxwyyf8ydos37tl5tnjhhf6"

SYMBOL = "NIFTY 50"
TOKEN = 256265
INTERVAL = "5minute"

CAPITAL = 100000
LOT_SIZE = 50
RISK_PER_TRADE = 0.01
SL_MULT = 1.2
TP_MULT = 2.0

# ===================== LOGIN =====================
kite = KiteConnect(api_key=API_KEY)
print("Login URL:", kite.login_url())
request_token = input("Enter request token: ")

session = kite.generate_session(request_token, api_secret=API_SECRET)
kite.set_access_token(session["access_token"])
print("✅ Logged in successfully")

# ===================== DATA FETCH =====================
def get_data():
    df = kite.historical_data(
        instrument_token=TOKEN,
        from_date=datetime.now() - timedelta(days=7),
        to_date=datetime.now(),
        interval=INTERVAL
    )
    df = pd.DataFrame(df)
    df.set_index("date", inplace=True)
    return df

# ===================== INDICATORS =====================
def add_indicators(df):
    df["EMA20"] = ta.trend.ema_indicator(df["close"], 20)
    df["EMA50"] = ta.trend.ema_indicator(df["close"], 50)
    df["RSI"] = ta.momentum.rsi(df["close"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], 14)
    df["ADX"] = ta.trend.adx(df["high"], df["low"], df["close"], 14)
    df["RET"] = df["close"].pct_change()
    return df.dropna()

# ===================== ML TRAINING =====================
def train_model(df):
    df["TARGET"] = np.where(df["close"].shift(-3) > df["close"], 1, 0)
    df.dropna(inplace=True)

    features = ["EMA20", "EMA50", "RSI", "ATR", "RET"]
    X = df[features]
    y = df["TARGET"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X, y)
    df["ML_PROB"] = model.predict_proba(X)[:, 1]
    return df, model

# ===================== SIGNAL LOGIC =====================
def get_signal(row):
    if row["ADX"] > 25 and row["EMA20"] > row["EMA50"] and row["ML_PROB"] > 0.65:
        return 1
    elif row["ADX"] > 25 and row["EMA20"] < row["EMA50"] and row["ML_PROB"] < 0.35:
        return -1
    return 0

# ===================== PAPER TRADING ENGINE =====================
capital = CAPITAL
position = 0
entry_price = 0
trade_log = []

print("\n🚀 Bot Started...\n")

while True:
    try:
        df = get_data()
        df = add_indicators(df)
        df, model = train_model(df)

        row = df.iloc[-1]
        signal = get_signal(row)
        price = row["close"]

        # ENTRY
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            print(f"ENTRY @ {price} | {'BUY' if signal==1 else 'SELL'}")

        # EXIT LOGIC
        if position != 0:
            sl = row["ATR"] * SL_MULT
            tp = row["ATR"] * TP_MULT

            exit_trade = False

            if position == 1:
                if price <= entry_price - sl or price >= entry_price + tp:
                    exit_trade = True
            else:
                if price >= entry_price + sl or price <= entry_price - tp:
                    exit_trade = True

            if exit_trade:
                pnl = (price - entry_price) * LOT_SIZE * position
                capital += pnl

                trade_log.append({
                    "Time": datetime.now(),
                    "Entry": entry_price,
                    "Exit": price,
                    "PnL": round(pnl, 2),
                    "Capital": round(capital, 2)
                })

                print(f"EXIT @ {price} | PnL: {round(pnl,2)} | Capital: {capital}")
                position = 0

        time.sleep(300)

    except Exception as e:
        print("ERROR:", e)
        time.sleep(10)
