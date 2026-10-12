import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit.components.v1 as components

# مكتبات المؤشرات الفنية والتعلم الآلي
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# إعدادات الواجهة
st.set_page_config(page_title="AI Financial Agent", layout="wide")

st.sidebar.title("🎛️ الوكيل المالي الذكي")
market_choice = st.sidebar.radio("اختر السوق:", ["السوق السعودي (تداول)", "الأسهم الأمريكية (US)"])

saudi_stocks = {
    "أرامكو السعودية (2222)": "2222.SR",
    "مصرف الراجحي (1120)": "1120.SR",
    "سابك (2010)": "2010.SR",
    "الاتصالات STC (7010)": "7010.SR"
}

us_stocks = {
    "Apple (AAPL)": "AAPL",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA"
}

if market_choice == "السوق السعودي (تداول)":
    selected_name = st.sidebar.selectbox("اختر السهم:", list(saudi_stocks.keys()))
    symbol = saudi_stocks[selected_name]
else:
    selected_name = st.sidebar.selectbox("اختر السهم:", list(us_stocks.keys()))
    symbol = us_stocks[selected_name]

custom_sym = st.sidebar.text_input("أو أدخل رمز سهم محدد:")
if custom_sym.strip():
    symbol = custom_sym.strip().upper()

# جلب ومعالجة البيانات
@st.cache_data(ttl=900)
def load_data(sym):
    df = yf.Ticker(sym).history(period="2y", interval="1d")
    if df.empty or len(df) < 50:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # مؤشرات فنية
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA_200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    bb = BollingerBands(close=df['Close'], window=20)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df.dropna()

data = load_data(symbol)

st.title(f"📈 التحليل المالي الذكي: {symbol}")

if data is None:
    st.error("تعذر جلب بيانات السهم، يرجى التأكد من الرمز.")
else:
    # تدريب الذكاء الاصطناعي (XGBoost 80/20)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
    X = data[features]
    y = data['Target']
    split = int(len(X) * 0.8)
    
    model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    pred = model.predict(X.iloc[-1:])[0]
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))
    
    signal = "BUY (شراء)" if pred == 1 and prob >= 0.52 else ("SELL (بيع)" if pred == 0 and (1-prob) >= 0.52 else "HOLD (حياد)")
    color = "#00e676" if "BUY" in signal else ("#ff5252" if "SELL" in signal else "#ffb300")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("السعر الأخير", f"{data['Close'].iloc[-1]:.2f}")
    c2.markdown(f"<div style='background:#1e222d;padding:10px;border-radius:6px;border-left:4px solid {color};'><b style='color:{color};'>{signal}</b></div>", unsafe_allow_html=True)
    c3.metric("احتمالية الصعود", f"{prob*100:.1f}%")
    c4.metric("دقة الاختبار (Test Acc)", f"{acc*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # شارت TradingView
    tv_symbol = f"TADAWUL:{symbol.replace('.SR','')}" if ".SR" in symbol else symbol
    tv_widget = f"""
    <div style="height:500px;width:100%;">
        <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&theme=dark&style=1"
                style="width:100%;height:500px;border:none;border-radius:8px;"></iframe>
    </div>
    """
    components.html(tv_widget, height=520)

    # التحليل الفني المحلي
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="السعر"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange'), name="SMA 50"))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], line=dict(color='blue'), name="SMA 200"))
    fig.update_layout(template="plotly_dark", height=400, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
