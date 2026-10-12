import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components

# مكتبات المؤشرات الفنية والتعلم الآلي
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor

# إعدادات الواجهة الرئيسية
st.set_page_config(page_title="AI Financial Agent", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🎛️ الوكيل المالي الذكي")
market_choice = st.sidebar.radio("اختر السوق:", ["السوق السعودي (تداول)", "الأسهم الأمريكية (US)"])

saudi_stocks = {
    "المطاحن الأولى (2283)": "2283.SR",
    "أرامكو السعودية (2222)": "2222.SR",
    "مصرف الراجحي (1120)": "1120.SR",
    "سابك (2010)": "2010.SR",
    "الاتصالات STC (7010)": "7010.SR",
    "معادن (1211)": "1211.SR",
    "البنك الأهلي (1180)": "1180.SR"
}

us_stocks = {
    "Apple (AAPL)": "AAPL",
    "NVIDIA (NVDA)": "NVDA",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet / Google (GOOGL)": "GOOGL"
}

active_stock_dict = saudi_stocks if market_choice == "السوق السعودي (تداول)" else us_stocks
selected_name = st.sidebar.selectbox("اختر السهم من القائمة:", list(active_stock_dict.keys()))
symbol = active_stock_dict[selected_name]

# معالجة الأرقام والرموز
def normalize_symbol(user_input):
    if not user_input:
        return ""
    ar_digits = "٠١٢٣٤٥٦٧٨٩"
    en_digits = "0123456789"
    cleaned = user_input.translate(str.maketrans(ar_digits, en_digits)).strip().upper()
    if cleaned.isdigit() and len(cleaned) == 4:
        cleaned += ".SR"
    return cleaned

custom_sym = st.sidebar.text_input("أو اكتب أي رمز سهم (مثال: 2283 أو AMZN):")
if custom_sym.strip():
    symbol = normalize_symbol(custom_sym)

# -------------------------------------------------------------
# دوال جلب البيانات والمؤشرات الفنية
# -------------------------------------------------------------
@st.cache_data(ttl=900)
def load_stock_data(sym):
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="2y", interval="1d")
        if df.empty or len(df) < 40:
            return None, None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # مؤشرات التحليل الفني
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

        # المؤشرات المالية
        fundamentals = {}
        try:
            info = ticker.info
            fundamentals['PE'] = info.get('trailingPE', info.get('forwardPE', 'N/A'))
            fundamentals['ROE'] = info.get('returnOnEquity', 'N/A')
            fundamentals['DebtToEquity'] = info.get('debtToEquity', 'N/A')
        except Exception:
            fundamentals = {'PE': 'N/A', 'ROE': 'N/A', 'DebtToEquity': 'N/A'}

        return df.dropna(), fundamentals
    except Exception:
        return None, None

# -------------------------------------------------------------
# دالة تحليل مشاعر الأخبار (Financial News Sentiment)
# -------------------------------------------------------------
@st.cache_data(ttl=1800)
def get_news_sentiment(sym):
    try:
        ticker = yf.Ticker(sym)
        news_items = ticker.news
        if not news_items:
            return 0.0, []

        pos_words = {'gain', 'surge', 'jump', 'rise', 'record', 'profit', 'growth', 'bullish', 'high', 'dividend', 'strong', 'beat', 'ارتفاع', 'نمو', 'أرباح', 'صعود', 'توزيعات'}
        neg_words = {'drop', 'fall', 'plunge', 'decline', 'loss', 'bearish', 'cut', 'slump', 'down', 'miss', 'risk', 'crisis', 'انخفاض', 'هبوط', 'خسائر', 'تراجع', 'عجز'}

        scored_news = []
        scores = []
        for item in news_items[:6]:
            title = item.get('title', '')
            link = item.get('link', '#')
            publisher = item.get('publisher', 'Market News')
            words = set(title.lower().split())
            
            p_count = len(words.intersection(pos_words))
            n_count = len(words.intersection(neg_words))
            score = (p_count - n_count) / max((p_count + n_count), 1) if (p_count + n_count) > 0 else 0.0
            scores.append(score)
            
            sentiment_label = "🟢 إيجابي" if score > 0 else ("🔴 سلبي" if score < 0 else "⚪ محايد")
            scored_news.append({"title": title, "publisher": publisher, "link": link, "label": sentiment_label})

        avg_sentiment = float(np.mean(scores)) if scores else 0.0
        return round(avg_sentiment, 2), scored_news
    except Exception:
        return 0.0, []

# -------------------------------------------------------------
# دالة توقع الأسعار للأيام الخمسة القادمة (5-Day Price Forecast)
# -------------------------------------------------------------
def predict_future_prices(df):
    features = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'BB_Upper', 'BB_Lower']
    X = df[features].iloc[:-5]
    forecast_results = []
    
    last_price = df['Close'].iloc[-1]
    current_features = df[features].iloc[-1:]

    for day in range(1, 6):
        y = df['Close'].shift(-day).iloc[:-5]
        reg = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.08, random_state=42)
        reg.fit(X, y)
        pred_val = float(reg.predict(current_features)[0])
        
        # حساب نطاق تذبذب تقريبي بناءً على تقلبات السهم
        volatility = df['Close'].pct_change().std() * np.sqrt(day) * last_price
        forecast_results.append({
            "Day": f"اليوم +{day}",
            "Predicted": round(pred_val, 2),
            "Lower": round(pred_val - (volatility * 0.8), 2),
            "Upper": round(pred_val + (volatility * 0.8), 2),
            "ChangePct": round(((pred_val - last_price) / last_price) * 100, 2)
        })
    return pd.DataFrame(forecast_results)

# -------------------------------------------------------------
# دالة تصنيف وترتيب الأسهم (Stock Ranking)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def generate_market_ranking(stock_dict):
    ranking_data = []
    for name, sym in stock_dict.items():
        try:
            d = yf.Ticker(sym).history(period="6mo", interval="1d")
            if d.empty or len(d) < 30:
                continue
            rsi = RSIIndicator(close=d['Close'], window=14).rsi().iloc[-1]
            sma50 = SMAIndicator(close=d['Close'], window=50).sma_indicator().iloc[-1]
            last_p = d['Close'].iloc[-1]
            change_month = ((last_p - d['Close'].iloc[-20]) / d['Close'].iloc[-20]) * 100
            
            # احتساب درجة القوة المركبة (Composite Score من 100)
            score = 50
            if rsi < 35: score += 20
            elif rsi > 70: score -= 15
            else: score += 5
            
            if last_p > sma50: score += 15
            if change_month > 0: score += 10
            
            rank_label = "🟢 فرصة شراء قوية" if score >= 75 else ("🟡 شراء تدريجي" if score >= 60 else "⚪ حياد / مراقبة")
            ranking_data.append({
                "السهم": name,
                "الرمز": sym,
                "السعر الأخير": round(last_p, 2),
                "RSI": round(rsi, 1),
                "تغير شهر (%)": f"{change_month:+.1f}%",
                "درجة القوة": min(score, 100),
                "التقييم": rank_label
            })
        except Exception:
            continue
    df_rank = pd.DataFrame(ranking_data)
    return df_rank.sort_values(by="درجة القوة", ascending=False) if not df_rank.empty else df_rank

# -------------------------------------------------------------
# واجهة العرض والتحليل
# -------------------------------------------------------------
data, fundamentals = load_stock_data(symbol)
sentiment_score, news_list = get_news_sentiment(symbol)

st.title(f"📈 الوكيل المالي الذكي: {symbol}")

if data is None:
    st.error(f"تعذر جلب بيانات السهم `{symbol}`. تأكد من صحة الرمز.")
else:
    # تشغيل نموذج الاتجاه (XGBoost Classifier 80/20)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
    X = data[features]
    y = data['Target']
    split = int(len(X) * 0.8)

    model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])

    pred = model.predict(X.iloc[-1:])[0]
    prob = model.predict_proba(X.iloc[-1:])[0][1]
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))

    signal = "BUY (شراء)" if pred == 1 and prob >= 0.52 else ("SELL (بيع)" if pred == 0 and (1 - prob) >= 0.52 else "HOLD (حياد)")
    color = "#00e676" if "BUY" in signal else ("#ff5252" if "SELL" in signal else "#ffb300")

    # بطاقات المؤشرات العلوية
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("السعر الأخير (Close)", f"{data['Close'].iloc[-1]:.2f}")
    c2.markdown(f"<div style='background:#1e222d;padding:10px;border-radius:6px;border-left:4px solid {color};'><b style='color:{color}; font-size:18px;'>{signal}</b></div>", unsafe_allow_html=True)
    c3.metric("احتمالية الصعود (AI)", f"{prob * 100:.1f}%")
    c4.metric("مؤشر المشاعر (Sentiment)", f"{sentiment_score:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # تبويبات النظام الكاملة
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 الشارت الحي (TradingView)",
        "🔮 توقع الأسعار (5-Day Forecast)",
        "📰 مشاعر الأخبار (Sentiment)",
        "🏆 تصنيف الأسهم (Stock Ranking)",
        "📑 المؤشرات الفنية والمالية"
    ])

    # 1. TradingView Widget
    with tab1:
        st.subheader("الرسم البياني الحي من TradingView")
        tv_symbol = f"TADAWUL:{symbol.replace('.SR','')}" if ".SR" in symbol else symbol
        tv_html = f"""
        <div style="height:520px;width:100%;">
            <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&theme=dark&style=1"
                    style="width:100%;height:520px;border:none;border-radius:8px;"></iframe>
        </div>
        """
        components.html(tv_html, height=530)

    # 2. توقع المدى السعري
    with tab2:
        st.subheader("توقع مسار السعر للأيام الخمسة القادمة (AI Price Forecast)")
        df_forecast = predict_future_prices(data)
        
        # رسم بياني لتوقع المسار المستقبلي
        fig_f = go.Figure()
        recent_history = data['Close'].iloc[-15:]
        fig_f.add_trace(go.Scatter(x=list(range(-14, 1)), y=recent_history.values, mode='lines+markers', name='الأسعار السابقة', line=dict(color='#3b82f6', width=2)))
        
        future_x = list(range(1, 6))
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['Predicted'], mode='lines+markers', name='السعر المتوقع', line=dict(color='#00e676', dash='dash', width=2)))
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['Upper'], mode='lines', name='الحد الأعلى', line=dict(width=0), showlegend=False))
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['Lower'], mode='lines', name='نطاق التذبذب', fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)', line=dict(width=0)))
        
        fig_f.update_layout(template="plotly_dark", height=380, xaxis_title="الأيام (0 = اليوم الحالي)", yaxis_title="السعر", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_f, use_container_width=True)
        
        st.dataframe(df_forecast, hide_index=True, use_container_width=True)

    # 3. مشاعر الأخبار
    with tab3:
        st.subheader("تحليل نبرة الأخبار الحالية")
        st.write(f"درجة المشاعر العامة: **`{sentiment_score:+.2f}`** (تتراوح بين +1.0 إيجابي جداً و -1.0 سلبي جداً)")
        if news_list:
            for item in news_list:
                st.markdown(f"**{item['label']}** | [{item['title']}]({item['link']}) — *{item['publisher']}*")
        else:
            st.info("لا توجد أخبار حديثة مسجلة لهذا السهم عبر التغذية الإخبارية.")

    # 4. تصنيف وترتيب الأسهم
    with tab4:
        st.subheader(f"ترتيب فرص {market_choice} بناءً على درجة القوة المركبة")
        with st.spinner("جاري فحص وترتيب أسهم السوق..."):
            df_ranks = generate_market_ranking(active_stock_dict)
            if not df_ranks.empty:
                st.dataframe(df_ranks, hide_index=True, use_container_width=True)
            else:
                st.warning("تعذر إنشاء جدول الترتيب حالياً.")

    # 5. التفاصيل الفنية والأساسية
    with tab5:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            pe = f"{fundamentals['PE']:.2f}" if isinstance(fundamentals['PE'], (int, float)) else str(fundamentals['PE'])
            st.metric("مكرر الأرباح (P/E)", pe)
        with col_f2:
            roe = f"{fundamentals['ROE']*100:.2f}%" if isinstance(fundamentals['ROE'], (int, float)) else str(fundamentals['ROE'])
            st.metric("العائد على الملكية (ROE)", roe)
        with col_f3:
            st.metric("مؤشر RSI (14)", f"{data['RSI'].iloc[-1]:.2f}")

        # شارت الشموع اليابانية المحلي
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="السعر"))
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange'), name="SMA 50"))
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], line=dict(color='blue'), name="SMA 200"))
        fig_candle.update_layout(template="plotly_dark", height=380, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_candle, use_container_width=True)
