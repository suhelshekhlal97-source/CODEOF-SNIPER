import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from xgboost import XGBClassifier
from datetime import datetime, timezone

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
    .stButton>button {
        width: 100%;
        background-color: #00CC00;
        color: black;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Sniper Bot | Live Execution Terminal")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

# Asset Selector
asset_map = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Gold (GC=F)": "GC=F",
    "Euro/USD (EURUSD=X)": "EURUSD=X",
    "Nasdaq 100 (NQ=F)": "NQ=F"
}
selected_asset = st.sidebar.selectbox("Select Asset", list(asset_map.keys()))
SYMBOL = asset_map[selected_asset]

# Settings
PERIOD = st.sidebar.select_slider("Data Lookback", options=["1mo", "3mo", "6mo", "1y"], value="3mo")
CONFIDENCE = st.sidebar.slider("Min Confidence %", 50, 95, 65) / 100
INTERVAL = "1h"

st.sidebar.markdown("---")
st.sidebar.caption(f"Bot Status: **Active**")
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
        
        data.dropna(inplace=True)
        return data
    except Exception as e:
        return pd.DataFrame()

# Load Data
with st.spinner(f"Connecting to Exchange for {SYMBOL}..."):
    df = fetch_market_data(SYMBOL, PERIOD, INTERVAL)

if df.empty:
    st.error("❌ Connection Failed. Market may be closed or ticker invalid.")
    st.stop()

# AI Prediction
def get_ai_signal(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    features = ['RSI', 'MACD', 'ATR']
    
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    
    # Train XGBoost
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')
    model.fit(X, y)
    
    # Predict Current Candle
    last_candle = df[features].iloc[[-1]]
    probs = model.predict_proba(last_candle)[0]
    return probs[1], probs[0] # Prob Up, Prob Down

prob_buy, prob_sell = get_ai_signal(df)

# ==========================================
# 4. DASHBOARD UI
# ==========================================
# Session Logic
hour = datetime.now(timezone.utc).hour
session = "ASIAN (Sleep)"
session_color = "off"
if 7 <= hour <= 16:
    session = "LONDON (Active)"
    session_color = "normal"
elif 13 <= hour <= 21:
    session = "NEW YORK (Active)"
    session_color = "normal"

# Signal Logic
signal = "WAIT"
if prob_buy > CONFIDENCE:
    signal = "BUY"
elif prob_sell > CONFIDENCE:
    signal = "SELL"

# --- TOP ROW METRICS ---
col1, col2, col3, col4 = st.columns(4)

current_price = df['Close'].iloc[-1]
change = current_price - df['Open'].iloc[-1]
atr = df['ATR'].iloc[-1]

col1.metric("Live Price", f"${current_price:,.2f}", f"{change:.2f}")
col2.metric("Market Session (UTC)", session, delta_color=session_color)
col3.metric("AI Signal", signal, f"{max(prob_buy, prob_sell)*100:.1f}% Conf")
col4.metric("Stop Loss (1x ATR)", f"${current_price - atr:.2f}" if signal == "BUY" else f"${current_price + atr:.2f}")

# --- CHART AREA ---
st.markdown("### 📊 Market Analysis")
fig = go.Figure()

# Candlesticks
fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Price"))

# Moving Averages
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"))
fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name="SMA 200"))

# Layout
fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig, use_container_width=True)

# --- EXECUTION PANEL ---
c1, c2 = st.columns([2, 1])

with c1:
    # FIX: Convert numpy values to standard python floats using float()
    bullish_conf = float(prob_buy)
    bearish_conf = float(prob_sell)

    st.info(f"**AI Logic:** Analyzing {len(df)} candles. XGBoost Confidence: **{bullish_conf*100:.1f}% Bullish** vs **{bearish_conf*100:.1f}% Bearish**.")
    
    # FIX: Use the converted float value for the progress bar
    st.progress(bullish_conf)

with c2:
    if st.button("🔄 REFRESH DATA"):
        st.rerun()
