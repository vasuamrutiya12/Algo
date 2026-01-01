# ==========================================================
# AI POWERED LIVE PAPER TRADING SYSTEM (FULL VERSION)
# ==========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from datetime import datetime

# ------------------ CONFIG ------------------
SYMBOL = "^NSEI"
INTERVAL = "5m"
HISTORY_DAYS = "60d"
CAPITAL_START = 10000
SL_MULT = 1.2
TP_MULT = 2.0

# ------------------ SESSION INIT ------------------
if "capital" not in st.session_state:
    st.session_state.capital = CAPITAL_START
if "position" not in st.session_state:
    st.session_state.position = 0
if "entry_price" not in st.session_state:
    st.session_state.entry_price = 0.0
if "trades" not in st.session_state:
    st.session_state.trades = []
if "model" not in st.session_state:
    st.session_state.model = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ------------------ DATA LOAD ------------------
@st.cache_data
def load_data():
    df = yf.download(SYMBOL, period=HISTORY_DAYS, interval=INTERVAL)
    df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

# ------------------ FEATURES ------------------
def add_features(df):
    df["EMA20"] = ta.trend.ema_indicator(df["Close"], 20)
    df["EMA50"] = ta.trend.ema_indicator(df["Close"], 50)
    df["RSI"] = ta.momentum.rsi(df["Close"], 14)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
    df["RET"] = df["Close"].pct_change()
    return df.dropna()

# ------------------ MODEL TRAIN ------------------
def train_model(df):
    df["TARGET"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[["EMA20", "EMA50", "RSI", "ATR", "RET"]]
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
    return model

# ------------------ INITIALIZE ------------------
if not st.session_state.initialized:
    df = add_features(load_data())
    st.session_state.model = train_model(df)
    st.session_state.initialized = True

# ------------------ LIVE DATA ------------------
df = add_features(load_data())
latest = df.iloc[-1]

# ------------------ SIGNAL LOGIC ------------------
X_live = latest[["EMA20", "EMA50", "RSI", "ATR", "RET"]].values.reshape(1, -1)
prob = st.session_state.model.predict_proba(X_live)[0][1]

signal = 0
if prob > 0.6 and latest["EMA20"] > latest["EMA50"]:
    signal = 1
elif prob < 0.4 and latest["EMA20"] < latest["EMA50"]:
    signal = -1

# ------------------ TRADING LOGIC ------------------
if st.session_state.position == 0 and signal != 0:
    st.session_state.entry_price = latest["Close"]
    st.session_state.position = signal

elif st.session_state.position != 0:
    sl = latest["ATR"] * SL_MULT
    tp = sl * TP_MULT

    exit_cond = (
        (latest["Close"] <= st.session_state.entry_price - sl) or
        (latest["Close"] >= st.session_state.entry_price + tp)
    )

    if exit_cond:
        pnl = (latest["Close"] - st.session_state.entry_price) * st.session_state.position
        st.session_state.capital += pnl

        st.session_state.trades.append({
            "Time": datetime.now(),
            "Side": "BUY" if st.session_state.position == 1 else "SELL",
            "Entry": st.session_state.entry_price,
            "Exit": latest["Close"],
            "PnL": round(pnl, 2)
        })
        st.session_state.position = 0

# ================= DASHBOARD =================
st.title("📊 AI LIVE PAPER TRADING DASHBOARD")

col1, col2, col3 = st.columns(3)
col1.metric("Capital", f"₹{st.session_state.capital:,.0f}")
col2.metric("Open Position", st.session_state.position)
col3.metric("Trades", len(st.session_state.trades))

if st.session_state.trades:
    df_trades = pd.DataFrame(st.session_state.trades)

    st.subheader("📜 Trade Log")
    st.dataframe(df_trades)

    st.subheader("📈 Equity Curve")
    st.line_chart(df_trades["PnL"].cumsum())

    st.subheader("📊 Performance")
    st.write("Win Rate:", round((df_trades["PnL"] > 0).mean() * 100, 2), "%")
    st.write("Net Profit:", round(df_trades["PnL"].sum(), 2))
else:
    st.info("Waiting for trades...")

st.caption("Live paper trading | Auto-refresh every reload")
