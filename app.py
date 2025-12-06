import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone
import numpy as np

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Sniper Bot Command Center",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
    }
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | Live Execution Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

asset_map = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold (GC=F)": "GC=F",
    "Euro/USD (EURUSD=X)": "EURUSD=X",
    "Nasdaq 100 (NQ=F)": "NQ=F"
}
selected_asset = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
SYMBOL = asset_map[selected_asset]

PERIOD = st.sidebar.select_slider("Data Lookback", options=["1mo", "3mo", "6mo", "1y"], value="3mo")
CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
INTERVAL = "1h"

st.sidebar.markdown("---")
st.sidebar.caption(f"Strategy: **XGBoost Trend + Session Filter**")

# ==========================================
# 3. BACKEND ENGINE
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Calculate Indicators
        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['MACD'] = ta.macd(data['Close'])['MACD_12_26_9']
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['SMA_50'] = ta.sma(data['Close'], length=50)
        data['SMA_200'] = ta.sma(data['Close'], length=200)
        
        # Session Filter (UTC Hours)
        data['Hour'] = data.index.hour
        data['Session'] = "ASIAN"
        data.loc[(data['Hour'] >= 7) & (data['Hour'] <= 16), 'Session'] = "LON"
        data.loc[(data['Hour'] >= 13) & (data['Hour'] <= 21), 'Session'] = "NY"
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()

with st.spinner(f"Connecting to Exchange for {SYMBOL}..."):
    df = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df.empty:
    st.error("❌ Connection Failed. Market may be closed or ticker invalid.")
    st.stop()

# ==========================================
# 4. AI & BACKTEST LOGIC
# ==========================================
def run_simulation(df, threshold):
    # 1. Prepare Data
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'ATR']
    
    # 2. Train Model
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X, y)
    
    # 3. Get Probabilities for ALL candles
    all_probs = model.predict_proba(df[features]) # [[Prob_Down, Prob_Up], ...]
    
    # 4. Live Signal (Last Candle)
    live_prob_buy = all_probs[-1][1]
    live_prob_sell = all_probs[-1][0]
    
    # 5. Run Backtest Loop to Generate History Table
    trades = []
    balance = 0
    
    # Iterate through history (skip first 50 for stability)
    for i in range(50, len(df)-1):
        date = df.index[i]
        session = df['Session'].iloc[i]
        
        # Skip Asian Session
        if session == "ASIAN":
            continue
            
        prob_buy = all_probs[i][1]
        prob_down = all_probs[i][0]
        
        action = "WAIT"
        if prob_buy > threshold:
            action = "BUY"
        elif prob_down > threshold:
            action = "SELL"
            
        if action != "WAIT":
            entry_price = df['Open'].iloc[i+1] # Enter next candle open
            close_price = df['Close'].iloc[i+1] # Exit next candle close (Intraday)
            atr = df['ATR'].iloc[i]
            
            # Simple PnL Logic (1 Hour Hold)
            pnl = 0
            if action == "BUY":
                pnl = close_price - entry_price
            elif action == "SELL":
                pnl = entry_price - close_price
                
            trades.append({
                "Date (UTC)": date.strftime('%Y-%m-%d %H:%M'),
                "Session": session,
                "Action": action,
                "Price": f"{entry_price:.2f}",
                "PnL": f"{pnl:.2f}"
            })
            
    return live_prob_buy, live_prob_sell, pd.DataFrame(trades)

prob_buy, prob_sell, trade_history = run_simulation(df, CONFIDENCE)

# ==========================================
# 5. DASHBOARD UI
# ==========================================
# Session Logic for Display
current_hour = datetime.now(timezone.utc).hour
current_session = "ASIAN (Sleep)"
session_color = "off"
if 7 <= current_hour <= 16:
    current_session = "LONDON (Active)"
    session_color = "normal"
elif 13 <= current_hour <= 21:
    current_session = "NEW YORK (Active)"
    session_color = "normal"

# Live Signal Logic
signal = "WAIT"
if prob_buy > CONFIDENCE:
    signal = "BUY"
elif prob_sell > CONFIDENCE:
    signal = "SELL"

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
change = current_price - df['Open'].iloc[-1]
atr = df['ATR'].iloc[-1]

col1.metric("Live Price", f"${current_price:,.2f}", f"{change:.2f}")
col2.metric("Current Session", current_session, delta_color=session_color)
col3.metric("AI Signal", signal, f"{max(prob_buy, prob_sell)*100:.1f}% Conf")
col4.metric("Stop Loss (1x ATR)", f"${current_price - atr:.2f}" if signal == "BUY" else f"${current_price + atr:.2f}")

# --- CHART ---
st.markdown("### 📊 Live Market Analysis")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name="SMA 200"))
fig.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0,r=0,t=0,b=0))
st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION PANEL ---
c1, c2 = st.columns([2, 1])
with c1:
    bull_conf = float(prob_buy)
    bear_conf = float(prob_sell)
    st.info(f"**AI Prediction:** {bull_conf*100:.1f}% Bullish vs {bear_conf*100:.1f}% Bearish")
    st.progress(bull_conf)
with c2:
    if st.button("🔄 REFRESH DATA"):
        st.rerun()

# --- NEW SECTION: TRADE HISTORY ---
st.markdown("---")
st.subheader(f"📜 Recent Trade History ({PERIOD})")

if not trade_history.empty:
    # Reverse order to show newest trades first
    trade_history = trade_history.iloc[::-1]
    
    # Display as a clean interactive table
    st.dataframe(
        trade_history, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "PnL": st.column_config.TextColumn(
                "PnL ($)",
                help="Profit/Loss per 1 Unit",
                validate="^-?\d+(\.\d{1,2})?$" # Validates numbers
            ),
            "Action": st.column_config.TextColumn(
                "Signal",
            ),
        }
    )
else:
    st.warning("No trades found in this period matching your Confidence Threshold.")
