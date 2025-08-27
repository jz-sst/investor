
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Try to import project modules; if any fail, the app will fall back gracefully.
try:
    from data_retrieval import DataRetrieval
    HAVE_DR = True
except Exception:
    HAVE_DR = False
try:
    from technical_analysis import TechnicalAnalysis
    HAVE_TA = True
except Exception:
    HAVE_TA = False
try:
    from fundamental_analysis import FundamentalAnalysis
    HAVE_FA = True
except Exception:
    HAVE_FA = False

st.set_page_config(page_title="InvestTrack Pro", layout="wide")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_prices(ticker: str, lookback_days: int = 260) -> pd.Series:
    if HAVE_DR:
        dr = DataRetrieval(None)
        df = dr.get_historical_data(ticker, period=f"{lookback_days}d")
        if isinstance(df, pd.DataFrame) and "Close" in df.columns:
            return df["Close"]
    # fallback random walk
    np.random.seed(abs(hash(ticker)) % 2**32)
    n = max(60, lookback_days)
    base = 50 + np.random.rand() * 150
    drift = (np.random.rand() - 0.5) * 0.003
    vals = [base]
    for _ in range(1, n):
        vals.append(max(1, vals[-1] * (1 + drift + (np.random.rand() - 0.5) * 0.02)))
    idx = pd.date_range(end=datetime.today(), periods=n, freq="B")
    return pd.Series(vals, index=idx, name="Close")

def rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) <= period:
        return None
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = (gain / loss).iloc[-1]
    if loss.iloc[-1] == 0:
        return 100.0
    return float(100 - (100 / (1 + rs)))

def sma(series: pd.Series, n: int) -> float | None:
    if len(series) < n: return None
    return float(series.tail(n).mean())

def macd_hist(series: pd.Series) -> float | None:
    if len(series) < 26: return None
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(hist.iloc[-1])

def score_tech(series: pd.Series):
    notes = []
    score = 50
    r = rsi(series, 14)
    if r is not None:
        if r < 30: score += 25; notes.append(f"RSI {r:.0f} oversold")
        elif r > 70: score -= 15; notes.append(f"RSI {r:.0f} overbought")
        else: notes.append(f"RSI {r:.0f} neutral")
    ma50, ma200 = sma(series, 50), sma(series, 200)
    if ma50 is not None and ma200 is not None:
        if ma50 > ma200: score += 15; notes.append("50MA > 200MA")
        else: score -= 10; notes.append("50MA < 200MA")
    mh = macd_hist(series)
    if mh is not None:
        if mh > 0: score += 10; notes.append("MACD bullish")
        else: score -= 5; notes.append("MACD bearish")
    score = max(0, min(100, round(score)))
    return {"rsi": r, "tech": score, "notes": notes}

def score_fund(pe: float | None, roe: float | None, de: float | None, sector_pe: float = 20):
    if pe is None and roe is None and de is None:
        return {"pe": None, "fund": None, "notes": ["No fundamentals"]}
    score = 50; notes=[]
    if pe is not None:
        diff = abs(pe - sector_pe)
        score = score*0.6 + (100 - min(100, diff*5))*0.4
        notes.append(f"P/E {pe} vs sector {sector_pe}")
    if roe is not None:
        norm = max(0,min(1,(roe-5)/(25-5)))
        score = score*0.7 + norm*100*0.3
        notes.append(f"ROE {roe}%")
    if de is not None:
        norm = 1-max(0,min(1,de/3.0))
        score = score*0.8 + norm*100*0.2
        notes.append(f"D/E {de}")
    return {"pe": pe, "fund": int(round(score)), "notes": notes}

def score_news(sent: float | None, count: int | None):
    if sent is None and count is None:
        return {"sent": None, "count": None, "news": None, "notes": ["No news"]}
    s = 0 if sent is None else sent
    base = int(round(((s+1)/2) * 100))
    vol_boost = min(1.5, 1 + (0 if count is None else count)/50)
    news = int(round(base * vol_boost))
    return {"sent": sent, "count": count, "news": news, "notes": [f"Sent {s:+.2f}", f"{count or 0} articles"]}

def decision(conf:int)->str:
    return "Buy" if conf>=70 else ("Hold" if conf>=45 else "Sell")

def analyze_ticker(ticker:str, tier:int=1):
    s = get_prices(ticker)
    price = float(s.iloc[-1])
    tech = score_tech(s)
    # fundamentals (fallback mock)
    pe = round(10 + (abs(hash(ticker)) % 40), 2)
    roe = round(5 + (abs(hash(ticker+'roe')) % 25), 2)
    de = round(((abs(hash(ticker+'de')) % 250) / 100), 2)
    fund = score_fund(pe, roe, de, 20)
    # news (fallback mock)
    sent = round(((abs(hash(ticker+'sent')) % 200)/100)-1, 2)
    count = (abs(hash(ticker+'news')) % 30) + 1
    news = score_news(sent, count)
    weights = {"tech":0.5, "fund":0.3 if tier>=2 else 0, "news":0.2 if tier>=3 else 0}
    denom = sum(weights.values()) or 1
    conf = int(round(((tech["tech"] or 0)*weights["tech"] + (fund["fund"] or 0)*weights["fund"] + (news["news"] or 0)*weights["news"]) / denom))
    rationale = tech["notes"] + fund["notes"] + news["notes"]
    return {
        "ticker": ticker, "price": price, "rsi": tech["rsi"], "pe": fund["pe"],
        "sentiment": news["sent"], "newsCount": news["count"],
        "tech": tech["tech"], "fund": fund["fund"], "newsScore": news["news"],
        "confidence": conf, "decision": decision(conf), "rationale": rationale
    }

def analyze_universe(tickers:list[str], tier:int=1):
    out = [analyze_ticker(t, tier) for t in tickers]
    return sorted(out, key=lambda r: r["confidence"], reverse=True)

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
PAGES = ["Dashboard","Portfolio","Stock Analysis","Market Scanner","Opportunities","Projects","Alerts","Settings"]
st.sidebar.title("InvestTrack Pro")
tab = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.caption("Preview build • self-contained engine")

# Top bar
col1, col2 = st.columns([3,1])
with col1:
    st.text_input("Search tickers, news…", key="global_search", placeholder="AAPL, NVDA …")
with col2:
    st.write("")
    st.button("Alerts")

# ---------------- Dashboard ----------------
if tab=="Dashboard":
    st.subheader("Dashboard")
    watch = ["AAPL","NVDA","MSFT","XLE"]
    data = {t: get_prices(t).iloc[-1] for t in watch}
    total = sum(data[t]* (200 if t=='AAPL' else 100) for t in watch)
    invested = sum((160 if t=='AAPL' else 90)* (200 if t=='AAPL' else 100) for t in watch)
    pl = total - invested
    ytd = (pl/invested)*100
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Portfolio Value", f"${total:,.0f}")
    c2.metric("Daily P/L", f"{'+' if pl>=0 else '-'}${abs(pl):,.0f}")
    c3.metric("YTD Return", f"{ytd:+.2f}%")
    c4.metric("Risk Score", "Moderate")
    st.divider()
    cL, cR = st.columns([2,1])
    with cL:
        st.caption("Performance Overview (placeholder)")
        st.line_chart(pd.DataFrame({ 'value':[total*0.9,total*0.92,total*0.95,total*0.97,total] }))
    with cR:
        st.caption("Key Indices")
        st.write("S&P 500 — 5,235.12")
        st.write("NASDAQ — 16,120.45")
        st.write("Dow Jones — 39,210.33")
    st.divider()
    st.caption("Recent News")
    st.write("- S&P 500 rises as tech leads gains")
    st.write("- NASDAQ posts third straight day of chip rally")
    st.write("- Energy stocks slip as oil retreats")

# ---------------- Portfolio ----------------
elif tab=="Portfolio":
    st.subheader("Portfolio")
    holdings = pd.DataFrame([
        {"Ticker":"AAPL","Qty":220,"Cost":162.0},
        {"Ticker":"NVDA","Qty":60,"Cost":620.0},
        {"Ticker":"MSFT","Qty":140,"Cost":340.0},
        {"Ticker":"XLE","Qty":180,"Cost":86.0},
    ])
    prices = {t: get_prices(t).iloc[-1] for t in holdings["Ticker"]}
    holdings["Price"] = holdings["Ticker"].map(prices)
    holdings["Value"] = holdings["Qty"]*holdings["Price"]
    holdings["P/L"] = (holdings["Price"]-holdings["Cost"])*holdings["Qty"]
    st.dataframe(holdings, use_container_width=True)
    total = holdings["Value"].sum()
    st.metric("Total", f"${total:,.0f}")
    # rebalancer
    target = total/len(holdings)
    holdings["Target$"] = target
    holdings["Diff$"] = holdings["Value"]-target
    holdings["Action"] = np.where(holdings["Diff$"]>0, "Sell", "Buy")
    st.caption("Rebalancing Suggestions (equal-weight)")
    st.dataframe(holdings[["Ticker","Action","Diff$"]], use_container_width=True)

# ---------------- Stock Analysis ----------------
elif tab=="Stock Analysis":
    st.subheader("Stock Analysis")
    q = st.text_input("Search ticker", "AAPL")
    tiers = st.segmented_control("Tier", options=[1,2,3], selection_mode="single", default=1)
    run = st.button("Run Analysis")
    if run:
        tickers = [q] if q else ["AAPL","MSFT","NVDA","AMZN","META","TSLA"]
        res = analyze_universe(tickers, int(tiers))
        df = pd.DataFrame(res)[["ticker","price","rsi","pe","sentiment","newsCount","tech","fund","newsScore","confidence","decision"]]
        df.columns = ["Ticker","Price","RSI","P/E","Sentiment","News (#)","Tech","Fund","News","Confidence","Decision"]
        st.dataframe(df, use_container_width=True)
        # per-ticker chat-like rationale
        sel = st.selectbox("Explain rationale for:", [r["ticker"] for r in res])
        chosen = next(r for r in res if r["ticker"]==sel)
        st.info("Why this signal:\n\n- " + "\n- ".join(chosen["rationale"]))
        st.write(f"Scores → Tech: {chosen['tech']}, Fund: {chosen['fund']}, News: {chosen['newsScore']}.  Confidence: {chosen['confidence']}% → **{chosen['decision']}**.")

# ---------------- Market Scanner ----------------
elif tab=="Market Scanner":
    st.subheader("Market Scanner")
    c1,c2,c3 = st.columns(3)
    with c1:
        min_price = st.number_input("Min Price", value=5.0)
    with c2:
        max_price = st.number_input("Max Price", value=1000.0)
    with c3:
        max_rsi = st.number_input("Max RSI", value=70.0)
    tier = st.selectbox("Tier", [1,2,3], index=0)
    if st.button("Run Scan"):
        universe = ["AAPL","MSFT","NVDA","AMZN","META","TSLA","AMD","GOOGL","AVGO","NFLX","ORCL","ADBE","SHOP","CRM","INTC","QCOM","CSCO","BA","JPM","V"]
        res = analyze_universe(universe, int(tier))
        df = pd.DataFrame(res)
        df = df[(df["price"]>=min_price) & (df["price"]<=max_price) & (df["rsi"].fillna(50)<=max_rsi)]
        st.dataframe(df[["ticker","price","rsi","pe","tech","fund","newsScore","confidence"]], use_container_width=True)

# ---------------- Opportunities ----------------
elif tab=="Opportunities":
    st.subheader("Opportunities")
    start = st.number_input("Start ($)", value=10000)
    mu = st.number_input("Expected Return μ", value=0.10)
    sigma = st.number_input("Volatility σ", value=0.22)
    years = st.number_input("Horizon (years)", value=3)
    if st.button("Run Projection"):
        # quick Monte Carlo
        paths=2000; steps=252; dt=years/steps
        rng = np.random.default_rng(42)
        end_vals=[]
        for _ in range(paths):
            v=float(start)
            for _ in range(steps):
                z = rng.standard_normal()
                v = v * np.exp((mu-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
            end_vals.append(v)
        arr = np.sort(np.array(end_vals))
        p5, p50, p95 = np.percentile(arr, [5,50,95])
        loss_prob = float((arr<start).mean())
        st.write(f"P5: ${p5:,.0f} • Median: ${p50:,.0f} • P95: ${p95:,.0f}")
        st.write(f"Probability of loss: {loss_prob*100:.1f}%")

# ---------------- Projects ----------------
elif tab=="Projects":
    st.subheader("Capital Projects")
    cash = st.text_input("Cashflows (, separated)", "-600000, 150000, 180000, 220000, 260000")
    try:
        flows = [float(x.strip()) for x in cash.split(",") if x.strip()]
        # IRR via numpy
        rate = np.irr(flows)
        st.write(f"Computed IRR: {rate*100:.2f}%")
    except Exception as e:
        st.error(f"Invalid cashflows: {e}")

# ---------------- Alerts ----------------
elif tab=="Alerts":
    st.subheader("Alerts Center")
    if st.button("Run Alerts"):
        tickers = ["AAPL","MSFT","NVDA","XLE"]
        res = analyze_universe(tickers, 3)
        rows=[]
        for r in res:
            if (r["rsi"] or 50) > 75: rows.append((r["ticker"], "RSI > 75", "Medium"))
            if isinstance(r["pe"], (int,float)) and r["pe"] > 35: rows.append((r["ticker"], f"High P/E ({r['pe']})", "High"))
            if (r["newsScore"] or 50) < 30: rows.append((r["ticker"], "Negative newsflow", "Low"))
            if r["decision"]=="Sell": rows.append((r["ticker"], f"Model SELL ({r['confidence']}%)", "Critical"))
        df = pd.DataFrame(rows, columns=["Ticker","Message","Severity"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Click **Run Alerts** to generate alerts from current analysis.")

# ---------------- Settings ----------------
elif tab=="Settings":
    st.subheader("Settings")
    market_key = st.text_input("Market API Key", value=st.session_state.get("INVEST_MKT_KEY",""))
    news_key = st.text_input("News API Key", value=st.session_state.get("INVEST_NEWS_KEY",""))
    if st.button("Save Keys"):
        st.session_state["INVEST_MKT_KEY"]=market_key
        st.session_state["INVEST_NEWS_KEY"]=news_key
        st.success("Saved for this session.")
