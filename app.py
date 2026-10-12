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
from xgboost import XGBClassifier

# -------------------------------------------------------------
# إعدادات الواجهة الرئيسية
# -------------------------------------------------------------
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
    "Google (GOOGL)": "GOOGL"
}

active_stock_dict = saudi_stocks if market_choice == "السوق السعودي (تداول)" else us_stocks
selected_name = st.sidebar.selectbox("اختر السهم من القائمة:", list(active_stock_dict.keys()))
symbol = active_stock_dict[selected_name]

# معالجة الأرقام العربية والرموز
def normalize_symbol(user_input):
    if not user_input:
        return ""
    ar_digits = "٠١٢٣٤٥٦٧٨٩"
    en_digits = "0123456789"
    cleaned = user_input.translate(str.maketrans(ar_digits, en_digits)).strip().upper()
    if cleaned.isdigit() and len(cleaned) == 4:
        cleaned += ".SR"
    return cleaned

custom_sym = st.sidebar.text_input("أو اكتب أي رمز سهم (مثال: 2283 أو 1111 أو AMZN):")
if custom_sym.strip():
    symbol = normalize_symbol(custom_sym)

# -------------------------------------------------------------
# جلب ومعالجة البيانات الفنية والأساسية
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

        # استخراج كافة النسب المالية الأساسية
        fundamentals = {}
        try:
            info = ticker.info
            fundamentals['currentRatio'] = info.get('currentRatio')
            fundamentals['quickRatio'] = info.get('quickRatio')
            fundamentals['PE'] = info.get('trailingPE', info.get('forwardPE'))
            fundamentals['operatingMargins'] = info.get('operatingMargins')
            fundamentals['profitMargins'] = info.get('profitMargins')
            d_e = info.get('debtToEquity')
            fundamentals['debtToEquity'] = (d_e / 100.0) if (d_e and d_e > 5) else d_e
            fundamentals['ROE'] = info.get('returnOnEquity')
            fundamentals['pegRatio'] = info.get('pegRatio')
            fundamentals['dividendYield'] = info.get('dividendYield')
            fundamentals['payoutRatio'] = info.get('payoutRatio')
            
            fcf = info.get('freeCashflow')
            mcap = info.get('marketCap')
            fundamentals['fcfYield'] = (fcf / mcap) if (fcf and mcap) else None
        except Exception:
            pass

        return df.dropna(), fundamentals
    except Exception:
        return None, None

# -------------------------------------------------------------
# دالة تحليل مشاعر الأخبار
# -------------------------------------------------------------
@st.cache_data(ttl=1800)
def get_news_sentiment(sym):
    try:
        ticker = yf.Ticker(sym)
        news_items = ticker.news
        if not news_items:
            return 0.0, []

        pos_words = {'gain', 'surge', 'jump', 'rise', 'profit', 'growth', 'bullish', 'high', 'dividend', 'strong', 'beat', 'ارتفاع', 'نمو', 'أرباح', 'صعود', 'توزيعات'}
        neg_words = {'drop', 'fall', 'plunge', 'decline', 'loss', 'bearish', 'cut', 'slump', 'down', 'risk', 'crisis', 'انخفاض', 'هبوط', 'خسائر', 'تراجع'}

        scored_news, scores = [], []
        for item in news_items[:6]:
            title = item.get('title', '')
            words = set(title.lower().split())
            p_count = len(words.intersection(pos_words))
            n_count = len(words.intersection(neg_words))
            score = (p_count - n_count) / max((p_count + n_count), 1) if (p_count + n_count) > 0 else 0.0
            scores.append(score)
            label = "🟢 إيجابي" if score > 0 else ("🔴 سلبي" if score < 0 else "⚪ محايد")
            scored_news.append({"title": title, "publisher": item.get('publisher', 'Market'), "link": item.get('link', '#'), "label": label})

        return round(float(np.mean(scores)), 2) if scores else 0.0, scored_news
    except Exception:
        return 0.0, []

# -------------------------------------------------------------
# دالة توقع الأسعار (متسقة منطقياً مع اتجاه وقوة الذكاء الاصطناعي)
# -------------------------------------------------------------
def predict_future_prices(df, bull_probability):
    df_temp = df.copy()
    df_temp['Daily_Return'] = df_temp['Close'].pct_change()
    
    last_price = float(df['Close'].iloc[-1])
    volatility = float(df_temp['Daily_Return'].std())
    if np.isnan(volatility) or volatility == 0:
        volatility = 0.015

    # توجيه المسار بناءً على احتمالية النموذج اللحظي
    # إذا كانت الاحتمالية 23.6%، يكون معامل الاتجاه سالباً (-0.528)
    direction_bias = (bull_probability - 0.5) * 2 
    
    forecast_results = []
    for day in range(1, 6):
        expected_change_pct = direction_bias * volatility * np.sqrt(day)
        predicted_p = last_price * (1 + expected_change_pct)
        band = volatility * np.sqrt(day) * last_price * 0.7
        
        forecast_results.append({
            "اليوم": f"يوم +{day}",
            "السعر المتوقع": round(predicted_p, 2),
            "الحد الأدنى": round(max(0.01, predicted_p - band), 2),
            "الحد الأعلى": round(predicted_p + band, 2),
            "التغير (%)": f"{((predicted_p - last_price) / last_price) * 100:+.2f}%"
        })
        
    return pd.DataFrame(forecast_results)

# -------------------------------------------------------------
# دالة تصنيف وترتيب أسهم السوق
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
# دالة بطاقات التقييم المالي الذكية
# -------------------------------------------------------------
def render_metric_card(name, val, formatted_val, condition_pos, condition_warn, explanation):
    if val is None or val == 'N/A':
        st.markdown(f"""
        <div style="background:#1e222d; padding:12px; border-radius:8px; border:1px solid #333; margin-bottom:10px;">
            <span style="color:#aaa; font-size:13px;">{name}</span>
            <h4 style="margin:4px 0; color:#888;">غير متوفر</h4>
            <span style="font-size:11px; color:#666;">{explanation}</span>
        </div>
        """, unsafe_allow_html=True)
        return

    if condition_pos(val):
        color, badge = "#00e676", "🟢 صحي / ممتاز"
    elif condition_warn(val):
        color, badge = "#ffb300", "🟡 مقبول / مراقبة"
    else:
        color, badge = "#ff5252", "🔴 خطر / سلبي"

    st.markdown(f"""
    <div style="background:#1e222d; padding:12px; border-radius:8px; border-left:4px solid {color}; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#ddd; font-weight:bold; font-size:14px;">{name}</span>
            <span style="color:{color}; font-size:12px; font-weight:bold;">{badge}</span>
        </div>
        <h3 style="margin:4px 0; color:{color}; font-size:22px;">{formatted_val}</h3>
        <p style="margin:0; font-size:11px; color:#aaa;">{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# المعالجة وعرض لوحة التحكم
# -------------------------------------------------------------
data, fundamentals = load_stock_data(symbol)
sentiment_score, news_list = get_news_sentiment(symbol)

st.title(f"📈 الوكيل المالي الذكي: {symbol}")

if data is None:
    st.error(f"تعذر جلب بيانات السهم `{symbol}`. تأكد من صحة الرمز.")
else:
    # تدريب وتشغيل نموذج الذكاء الاصطناعي (XGBoost)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
    X = data[features]
    y = data['Target']
    split = int(len(X) * 0.8)

    model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])

    pred = model.predict(X.iloc[-1:])[0]
    prob = float(model.predict_proba(X.iloc[-1:])[0][1])
    acc = accuracy_score(y.iloc[split:], model.predict(X.iloc[split:]))

    signal = "BUY (شراء)" if pred == 1 and prob >= 0.52 else ("SELL (بيع)" if pred == 0 and (1 - prob) >= 0.52 else "HOLD (حياد)")
    color = "#00e676" if "BUY" in signal else ("#ff5252" if "SELL" in signal else "#ffb300")

    # بطاقات الملخص العلوية
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("السعر الأخير (Close)", f"{data['Close'].iloc[-1]:.2f}")
    c2.markdown(f"<div style='background:#1e222d;padding:10px;border-radius:6px;border-left:4px solid {color};'><b style='color:{color}; font-size:18px;'>{signal}</b></div>", unsafe_allow_html=True)
    c3.metric("احتمالية الصعود (AI)", f"{prob * 100:.1f}%")
    c4.metric("مؤشر المشاعر", f"{sentiment_score:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # التبويبات المتكاملة
    tab_tv, tab_fundamentals, tab_forecast, tab_rankings, tab_sentiment, tab_local_chart = st.tabs([
        "📈 الشارت الحي (TradingView)",
        "📑 الملاءة والتحليل الأساسي",
        "🔮 توقع مسار السعر (5 أيام)",
        "🏆 تصنيف وترتيب الأسهم",
        "📰 مشاعر الأخبار",
        "📊 التحليل الفني المحلي"
    ])

    # 1. TradingView Widget
    with tab_tv:
        st.subheader("رسم بياني حي من TradingView")
        tv_symbol = f"TADAWUL:{symbol.replace('.SR','')}" if ".SR" in symbol else symbol
        tv_html = f"""
        <div style="height:520px;width:100%;">
            <iframe src="https://s.tradingview.com/widgetembed/?symbol={tv_symbol}&interval=D&theme=dark&style=1"
                    style="width:100%;height:520px;border:none;border-radius:8px;"></iframe>
        </div>
        """
        components.html(tv_html, height=530)

    # 2. التحليل المالي الأساسي
    with tab_fundamentals:
        st.subheader("🔍 فحص النسب المالية للمستثمر (تقييم السلامة المالية)")
        col_st, col_md, col_lt = st.columns(3)

        with col_st:
            st.markdown("#### ⏱️ المدى القصير (السيولة)")
            cr = fundamentals.get('currentRatio')
            render_metric_card("نسبة التداول (Current Ratio)", cr, f"{cr:.2f}" if cr else "N/A",
                               lambda v: v >= 1.5, lambda v: 1.0 <= v < 1.5, "القدرة على الوفاء بديون السنة الحالية (> 1.5 آمن).")
            qr = fundamentals.get('quickRatio')
            render_metric_card("السيولة السريعة (Quick Ratio)", qr, f"{qr:.2f}" if qr else "N/A",
                               lambda v: v >= 1.0, lambda v: 0.7 <= v < 1.0, "تغطية الديون الفورية دون بيع المخزون (> 1.0 ممتاز).")

        with col_md:
            st.markdown("#### 📊 المدى المتوسط (الربحية والديون)")
            pe = fundamentals.get('PE')
            render_metric_card("مكرر الأرباح (P/E Ratio)", pe, f"{pe:.2f}" if pe else "N/A",
                               lambda v: 0 < v <= 18, lambda v: 18 < v <= 28, "تكلفة السهم مقابل كل ريال ربح (أقل من 18 جذاب).")
            om = fundamentals.get('operatingMargins')
            render_metric_card("الهامش التشغيلي (Operating Margin)", om, f"{om*100:.1f}%" if om else "N/A",
                               lambda v: v >= 0.18, lambda v: 0.08 <= v < 0.18, "كفاءة العمليات وضبط التكاليف (> 18% قوي).")
            de = fundamentals.get('debtToEquity')
            render_metric_card("الديون للملكية (Debt / Equity)", de, f"{de:.2f}" if de else "N/A",
                               lambda v: v <= 0.8, lambda v: 0.8 < v <= 1.4, "حجم القروض مقارنة بأموال الملاك (< 0.8 استقرار).")

        with col_lt:
            st.markdown("#### 🏛️ المدى البعيد (النمو واستدامة رأس المال)")
            roe = fundamentals.get('ROE')
            render_metric_card("العائد على الملكية (ROE)", roe, f"{roe*100:.1f}%" if roe else "N/A",
                               lambda v: v >= 0.15, lambda v: 0.08 <= v < 0.15, "مردود استثمار أموال المساهمين (> 15% ممتاز).")
            peg = fundamentals.get('pegRatio')
            render_metric_card("مكرر الربح للنمو (PEG Ratio)", peg, f"{peg:.2f}" if peg else "N/A",
                               lambda v: 0 < v <= 1.2, lambda v: 1.2 < v <= 2.0, "السعر قياساً بنمو الأرباح (< 1.2 رخيص لنموه).")
            fcf = fundamentals.get('fcfYield')
            render_metric_card("عائد التدفق الحر (FCF Yield)", fcf, f"{fcf*100:.1f}%" if fcf else "N/A",
                               lambda v: v >= 0.06, lambda v: 0.02 <= v < 0.06, "سيولة نقدية باقية للتوزيعات والتوسع (> 6% قوي).")
            dy = fundamentals.get('dividendYield')
            render_metric_card("عائد التوزيعات (Dividend Yield)", dy, f"{dy*100:.1f}%" if dy else "N/A",
                               lambda v: 0.04 <= v <= 0.08, lambda v: (0.01 <= v < 0.04) or (0.08 < v <= 0.11), "الدخل النقدي السنوي للمساهم (4% - 8% مستدام).")

    # 3. توقع الأسعار المتسق مع إشارة النموذج
    with tab_forecast:
        st.subheader("توقع مسار السعر لـ 5 أيام قادمة (متوافق مع قوة واتجاه AI)")
        df_forecast = predict_future_prices(data, prob)

        # رسم بياني تفاعلي للمسار
        fig_f = go.Figure()
        recent_history = data['Close'].iloc[-15:]
        fig_f.add_trace(go.Scatter(x=list(range(-14, 1)), y=recent_history.values, mode='lines+markers', name='الأسعار السابقة', line=dict(color='#3b82f6', width=2)))
        
        future_x = list(range(1, 6))
        line_color = '#00e676' if prob >= 0.5 else '#ff5252'
        area_color = 'rgba(0, 230, 118, 0.15)' if prob >= 0.5 else 'rgba(255, 82, 82, 0.15)'
        
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['السعر المتوقع'], mode='lines+markers', name='السعر المتوقع', line=dict(color=line_color, dash='dash', width=2)))
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['الحد الأعلى'], mode='lines', name='الحد الأعلى', line=dict(width=0), showlegend=False))
        fig_f.add_trace(go.Scatter(x=future_x, y=df_forecast['الحد الأدنى'], mode='lines', name='نطاق التذبذب', fill='tonexty', fillcolor=area_color, line=dict(width=0)))
        
        fig_f.update_layout(template="plotly_dark", height=380, xaxis_title="الأيام (0 = اليوم الحالي)", yaxis_title="السعر", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_f, use_container_width=True)

        st.dataframe(df_forecast, hide_index=True, use_container_width=True)

    # 4. ترتيب وتصنيف الأسهم
    with tab_rankings:
        st.subheader(f"ترتيب فرص {market_choice} بناءً على درجة القوة المركبة")
        with st.spinner("جاري فحص وترتيب أسهم السوق..."):
            df_ranks = generate_market_ranking(active_stock_dict)
            if not df_ranks.empty:
                st.dataframe(df_ranks, hide_index=True, use_container_width=True)
            else:
                st.warning("تعذر إنشاء جدول الترتيب حالياً.")

    # 5. مشاعر الأخبار
    with tab_sentiment:
        st.subheader("الأخبار الحديثة ونبرة السوق")
        st.write(f"مؤشر المشاعر الكلي: **`{sentiment_score:+.2f}`**")
        if news_list:
            for item in news_list:
                st.markdown(f"**{item['label']}** | [{item['title']}]({item['link']}) — *{item['publisher']}*")
        else:
            st.info("لا توجد أخبار مسجلة حالياً.")

    # 6. التحليل الفني والشموع المحلية
    with tab_local_chart:
        st.subheader("الشموع اليابانية والمتوسطات المتحركة 50 و 200")
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="السعر"))
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1.5), name="SMA 50"))
        fig_candle.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], line=dict(color='blue', width=1.5), name="SMA 200"))
        fig_candle.update_layout(template="plotly_dark", height=420, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_candle, use_container_width=True)
