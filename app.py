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

# إعدادات الواجهة
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
# جلب ومعالجة البيانات الفنية والأساسية الشاملة
# -------------------------------------------------------------
@st.cache_data(ttl=900)
def load_stock_data(sym):
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="2y", interval="1d")
        if df.empty or len(df) < 40:
            return None, None
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

        # استخراج كافة النسب المالية الأساسية
        fundamentals = {}
        try:
            info = ticker.info
            # السيولة (قصير المدى)
            fundamentals['currentRatio'] = info.get('currentRatio')
            fundamentals['quickRatio'] = info.get('quickRatio')
            
            # الهوامش والديون والتقييم (متوسط المدى)
            fundamentals['PE'] = info.get('trailingPE', info.get('forwardPE'))
            fundamentals['operatingMargins'] = info.get('operatingMargins')
            fundamentals['profitMargins'] = info.get('profitMargins')
            d_e = info.get('debtToEquity')
            fundamentals['debtToEquity'] = (d_e / 100.0) if (d_e and d_e > 5) else d_e

            # التدفقات والنمو والتوزيعات (بعيد المدى)
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
            p_count, n_count = len(words.intersection(pos_words)), len(words.intersection(neg_words))
            score = (p_count - n_count) / max((p_count + n_count), 1) if (p_count + n_count) > 0 else 0.0
            scores.append(score)
            label = "🟢 إيجابي" if score > 0 else ("🔴 سلبي" if score < 0 else "⚪ محايد")
            scored_news.append({"title": title, "publisher": item.get('publisher', 'Market'), "link": item.get('link', '#'), "label": label})

        return round(float(np.mean(scores)), 2) if scores else 0.0, scored_news
    except Exception:
        return 0.0, []

# -------------------------------------------------------------
# دالة توقع الأسعار للأيام الخمسة القادمة
# -------------------------------------------------------------
def predict_future_prices(df):
    features = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'BB_Upper', 'BB_Lower']
    X = df[features].iloc[:-5]
    forecast_results = []
    last_p = df['Close'].iloc[-1]
    curr_feat = df[features].iloc[-1:]

    for day in range(1, 6):
        y = df['Close'].shift(-day).iloc[:-5]
        reg = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.08, random_state=42)
        reg.fit(X, y)
        p = float(reg.predict(curr_feat)[0])
        vol = df['Close'].pct_change().std() * np.sqrt(day) * last_p
        forecast_results.append({
            "اليوم": f"يوم +{day}",
            "السعر المتوقع": round(p, 2),
            "الحد الأدنى": round(p - (vol * 0.8), 2),
            "الحد الأعلى": round(p + (vol * 0.8), 2),
            "التغير (%)": f"{((p - last_p) / last_p) * 100:+.2f}%"
        })
    return pd.DataFrame(forecast_results)

# -------------------------------------------------------------
# دالة تقييم بطاقات النسب المالية الذكية
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
        return 0

    if condition_pos(val):
        color, badge, score_pt = "#00e676", "🟢 صحي / ممتاز", 1
    elif condition_warn(val):
        color, badge, score_pt = "#ffb300", "🟡 مقبول / مراقبة", 0.5
    else:
        color, badge, score_pt = "#ff5252", "🔴 خطر / سلبي", 0

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
    return score_pt

# -------------------------------------------------------------
# واجهة التطبيق
# -------------------------------------------------------------
data, fundamentals = load_stock_data(symbol)
sentiment_score, news_list = get_news_sentiment(symbol)

st.title(f"📈 الوكيل المالي الذكي: {symbol}")

if data is None:
    st.error(f"تعذر جلب بيانات السهم `{symbol}`. تأكد من صحة الرمز والاتصال.")
else:
    # تشغيل نموذج الذكاء الاصطناعي (XGBoost)
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
    c1.metric("السعر الأخير", f"{data['Close'].iloc[-1]:.2f}")
    c2.markdown(f"<div style='background:#1e222d;padding:10px;border-radius:6px;border-left:4px solid {color};'><b style='color:{color}; font-size:18px;'>{signal}</b></div>", unsafe_allow_html=True)
    c3.metric("احتمالية الصعود (AI)", f"{prob * 100:.1f}%")
    c4.metric("مؤشر المشاعر", f"{sentiment_score:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # التبويبات
    tab_tv, tab_fundamentals, tab_forecast, tab_sentiment = st.tabs([
        "📈 الشارت الحي (TradingView)",
        "📑 الملاءة والتحليل الأساسي (صحي/خطر)",
        "🔮 توقع الأسعار (5-Day Forecast)",
        "📰 مشاعر الأخبار (Sentiment)"
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

    # 2. تبويب التحليل الأساسي الشامل والمصنف زمنياً
    with tab_fundamentals:
        st.subheader("🔍 فحص النسب المالية للمستثمر (تقييم السلامة المالية)")
        st.write("تحليل متقدم يصنف المؤشرات المالية للشركة بحسب الأفق الزمني لاكتشاف فرص القوة ومكامن الخطر:")

        col_st, col_md, col_lt = st.columns(3)

        # المدى القصير (السيولة)
        with col_st:
            st.markdown("#### ⏱️ المدى القصير (السيولة)")
            cr = fundamentals.get('currentRatio')
            render_metric_card(
                "نسبة التداول (Current Ratio)", cr, f"{cr:.2f}" if cr else "N/A",
                lambda v: v >= 1.5, lambda v: 1.0 <= v < 1.5,
                "يقيس القدرة على الوفاء بديون السنة الحالية (> 1.5 آمن)."
            )
            qr = fundamentals.get('quickRatio')
            render_metric_card(
                "السيولة السريعة (Quick Ratio)", qr, f"{qr:.2f}" if qr else "N/A",
                lambda v: v >= 1.0, lambda v: 0.7 <= v < 1.0,
                "تغطية الديون الفورية دون الحاجة لبيع المخزون (> 1.0 ممتاز)."
            )

        # المدى المتوسط (الربحية والديون)
        with col_md:
            st.markdown("#### 📊 المدى المتوسط (الأرباح والديون)")
            pe = fundamentals.get('PE')
            render_metric_card(
                "مكرر الأرباح (P/E Ratio)", pe, f"{pe:.2f}" if pe else "N/A",
                lambda v: 0 < v <= 18, lambda v: 18 < v <= 28,
                "تكلفة السهم مقابل كل ريال ربح (أقل من 18 جذاب استثمارياً)."
            )
            om = fundamentals.get('operatingMargins')
            render_metric_card(
                "الهامش التشغيلي (Operating Margin)", om, f"{om*100:.1f}%" if om else "N/A",
                lambda v: v >= 0.18, lambda v: 0.08 <= v < 0.18,
                "كفاءة العمليات وضبط التكاليف قبل الفوائد والضرائب (> 18% قوي)."
            )
            de = fundamentals.get('debtToEquity')
            render_metric_card(
                "الديون للملكية (Debt / Equity)", de, f"{de:.2f}" if de else "N/A",
                lambda v: v <= 0.8, lambda v: 0.8 < v <= 1.4,
                "حجم القروض مقارنة بأموال الملاك (< 0.8 استقرار مالي عالي)."
            )

        # المدى البعيد (النمو واستدامة رأس المال)
        with col_lt:
            st.markdown("#### 🏛️ المدى البعيد (النمو والتدفقات)")
            roe = fundamentals.get('ROE')
            render_metric_card(
                "العائد على الملكية (ROE)", roe, f"{roe*100:.1f}%" if roe else "N/A",
                lambda v: v >= 0.15, lambda v: 0.08 <= v < 0.15,
                "مردود استثمار أموال المساهمين (> 15% إدارة استثنائية)."
            )
            peg = fundamentals.get('pegRatio')
            render_metric_card(
                "مكرر الربح للنمو (PEG Ratio)", peg, f"{peg:.2f}" if peg else "N/A",
                lambda v: 0 < v <= 1.2, lambda v: 1.2 < v <= 2.0,
                "يقيم السعر قياساً بسرعة نمو الأرباح (< 1.2 سهم رخيص لنموه)."
            )
            fcf = fundamentals.get('fcfYield')
            render_metric_card(
                "عائد التدفق الحر (FCF Yield)", fcf, f"{fcf*100:.1f}%" if fcf else "N/A",
                lambda v: v >= 0.06, lambda v: 0.02 <= v < 0.06,
                "سيولة نقدية حقيقية باقية للتوزيعات والتوسع (> 6% قوي جداً)."
            )
            dy = fundamentals.get('dividendYield')
            render_metric_card(
                "عائد التوزيعات (Dividend Yield)", dy, f"{dy*100:.1f}%" if dy else "N/A",
                lambda v: 0.04 <= v <= 0.08, lambda v: (0.01 <= v < 0.04) or (0.08 < v <= 0.11),
                "الدخل النقدي السنوي المباشر للمساهم (4% - 8% مستدام ومجزٍ)."
            )

    # 3. توقع الأسعار
    with tab_forecast:
        st.subheader("توقع مسار السعر لـ 5 أيام قادمة")
        df_forecast = predict_future_prices(data)
        st.dataframe(df_forecast, hide_index=True, use_container_width=True)

    # 4. الأخبار والمشاعر
    with tab_sentiment:
        st.subheader("الأخبار الحديثة ونبرة السوق")
        st.write(f"مؤشر المشاعر الكلي: **`{sentiment_score:+.2f}`**")
        if news_list:
            for item in news_list:
                st.markdown(f"**{item['label']}** | [{item['title']}]({item['link']}) — *{item['publisher']}*")
        else:
            st.info("لا توجد أخبار مسجلة حالياً.")
