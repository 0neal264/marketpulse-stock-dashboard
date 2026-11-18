import pathlib
import pickle
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats

from analytics import analysis as an
from analytics import benchmark as bm
from data_loader import STOCKS, load_stock_data, load_company_info, load_fx_history

st.set_page_config(page_title="MarketPulse | Equity Intelligence", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

BG, GRID = "#0b1118", "#233243"
GREEN, RED, BLUE, AMBER, PURPLE, GREY = "#36d399", "#f87171", "#60a5fa", "#fbbf24", "#a78bfa", "#8fa1b5"

st.markdown("""
<style>
:root { --bg:#0b1118; --panel:#111b26; --muted:#8fa1b5; --line:#233243; --green:#36d399; }
.stApp { background:var(--bg); color:#e8eef5; }
[data-testid="stSidebar"] { background:#0e1721; border-right:1px solid var(--line); }
.block-container { padding-top:1.5rem; max-width:1500px; }
h1,h2,h3 { letter-spacing:-.03em; }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:12px 14px; border-radius:12px; overflow:hidden; min-height:118px; box-sizing:border-box; }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { color:#f3f7fb; font-size:1.3rem; }
[data-testid="stMetricLabel"] { font-size:.72rem; }
[data-testid="stMetricLabel"] p { font-size:.72rem; }
[data-testid="stMetricValue"] div, [data-testid="stMetricLabel"] div, [data-testid="stMetricDelta"] div { overflow:visible !important; text-overflow:clip !important; white-space:normal !important; word-break:break-word; }
div[data-testid="stTabs"] button { font-weight:600; }
hr { border-color:var(--line); }
.small-muted { color:var(--muted); font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

TICKERS = list(STOCKS.keys())
CUR = "DKK"  # display currency, set from the sidebar
CACHE_DIR = pathlib.Path(".mp_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _resilient(key, fn, empty):
    """Fetch with retries. Saves the last good result to disk and falls back to it if Yahoo fails."""
    f = CACHE_DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".pkl")
    for attempt in range(1):  # a single attempt: retrying only makes Yahoo rate limits worse
        try:
            out = fn()
            if out is not None and len(out) > 0:
                try:
                    with open(f, "wb") as fh:
                        pickle.dump(out, fh)
                except Exception:
                    pass
                return out
        except Exception:
            pass
        time.sleep(1.0 + attempt)
    if f.exists():
        try:
            with open(f, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    return empty
RULE = {"1d": None, "1wk": "W", "1mo": "MS"}


# ---------------------------------------------------------------- formatting helpers
def _ok(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def dkk(v):
    return f"{CUR} {v:,.2f}" if _ok(v) else "—"


def pct(v, d=2, sign=False):
    return (f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%") if _ok(v) else "—"


def num(v, d=2):
    return f"{v:,.{d}f}" if _ok(v) else "—"


def fmt_big(v):
    if not _ok(v):
        return "—"
    a = abs(v)
    for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"{CUR} {v / div:,.2f}{s}"
    return f"{CUR} {v:,.0f}"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def style(fig, h=380, **kw):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor=BG, plot_bgcolor=BG,
                      margin=dict(l=10, r=10, t=30, b=10), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------- cached data
@st.cache_data(ttl=900, show_spinner=False)
def _data_cached(ticker, period):
    d = _resilient(f"data_{ticker}_{period}", lambda: load_stock_data(ticker, period), pd.DataFrame())
    if d.empty:
        raise ValueError("empty")  # exceptions are not cached, so a failure is retried next run
    return d


def get_data(ticker, period):
    try:
        return _data_cached(ticker, period)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker):
    return load_company_info(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _fx_cached(period):
    d = _resilient(f"fx_{period}", lambda: load_fx_history(period), pd.Series(dtype=float))
    if len(d) == 0:
        raise ValueError("empty")
    return d


def get_fx(period):
    try:
        return _fx_cached(period)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def get_rate(currency, period):
    """Display-currency units per 1 USD, as a daily Series. USD returns a constant 1."""
    if currency == "USD":
        return pd.Series([1.0], index=pd.DatetimeIndex(["1970-01-01"]))
    if currency == "DKK":
        return get_fx(period)
    h = get_index("EURUSD=X", period)  # USD per EUR
    return 1.0 / h if len(h) else pd.Series(dtype=float)


def align_rate(rate, idx):
    """Align a rate Series to a target index (forward/back-filled), matching its timezone convention."""
    r = rate.copy()
    if r.index.tz is not None:
        r.index = r.index.tz_localize(None)
    r.index = r.index.normalize()
    ii = idx.tz_localize(None) if idx.tz is not None else idx
    out = r.reindex(r.index.union(ii)).ffill().bfill().reindex(ii)
    out.index = idx
    return out


@st.cache_data(ttl=900, show_spinner=False)
def get_universe(period, currency):
    """Close prices for the whole universe in the display currency."""
    closes = {}
    for t in TICKERS:
        try:
            d = get_data(t, period)
            if not d.empty:
                closes[t] = d["Close"]
        except Exception:
            pass
    fxs = get_rate(currency, period)
    if not closes or fxs.empty:
        return pd.DataFrame()
    p = pd.DataFrame(closes).sort_index()
    return p.mul(align_rate(fxs, p.index), axis=0)


@st.cache_data(ttl=3600, show_spinner=False)
def get_peers():
    rows = []
    for t in TICKERS:
        try:
            i = get_info(t)
        except Exception:
            i = {}
        rows.append({"Ticker": t, "Name": STOCKS[t]["name"],
                     "Mkt cap (USD bn)": _f(i.get("marketCap")) / 1e9,
                     "P/E": _f(i.get("trailingPE")), "Fwd P/E": _f(i.get("forwardPE")),
                     "P/B": _f(i.get("priceToBook")),
                     "Margin %": _f(i.get("profitMargins")) * 100,
                     "ROE %": _f(i.get("returnOnEquity")) * 100,
                     "Rev growth %": _f(i.get("revenueGrowth")) * 100,
                     "Div yield %": _f(i.get("dividendYield")) * 100,
                     "Beta": _f(i.get("beta"))})
    return pd.DataFrame(rows)


def _naive(s):
    s = s.copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.index = s.index.normalize()
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(ttl=900, show_spinner=False)
def _index_cached(symbol, period):
    h = _resilient(f"idx_{symbol}_{period}",
                   lambda: yf.Ticker(symbol).history(period=period)["Close"].dropna(), pd.Series(dtype=float))
    if len(h) == 0:
        raise ValueError("empty")
    return _naive(h)


def get_index(symbol, period):
    try:
        return _index_cached(symbol, period)
    except Exception:
        return pd.Series(dtype=float)


def build_prices3(close_n, period, currency):
    """Selected stock, S&P 500 (USD) and OMXC25 (DKK) in the display currency on shared dates."""
    sp, omx = get_index("^GSPC", period), get_index("^OMXC25", period)
    if sp.empty or omx.empty:
        return None
    rate_cur = get_rate(currency, period)
    if currency != "USD" and rate_cur.empty:
        return None
    sp_c = sp if currency == "USD" else sp * align_rate(rate_cur, sp.index)
    if currency == "DKK":
        omx_c = omx
    else:
        rate_dkk = get_rate("DKK", period)
        if rate_dkk.empty:
            return None
        omx_usd = omx / align_rate(rate_dkk, omx.index)
        omx_c = omx_usd if currency == "USD" else omx_usd * align_rate(rate_cur, omx.index)
    p = bm.align_close_prices(close_n, sp_c, omx_c)
    return p if len(p) >= 60 else None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ◈ MARKETPULSE")
    st.caption("EQUITY INTELLIGENCE TERMINAL")
    st.divider()
    selected = st.selectbox("SELECT SECURITY", TICKERS, format_func=lambda x: f"{x}  ·  {STOCKS[x]['name']}")
    period = st.selectbox("HISTORICAL WINDOW", ["1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("CHART INTERVAL", ["1d", "1wk", "1mo"], index=0)
    st.caption("DISPLAY CURRENCY")
    CUR = st.selectbox("Currency", ["DKK", "EUR", "USD"], label_visibility="collapsed")
    st.divider()
    st.markdown("**ANALYTICS SETTINGS**")
    rf = st.number_input("Risk-free rate (% p.a.)", 0.0, 10.0, 2.0, 0.25) / 100
    conf = st.select_slider("VaR confidence", [0.90, 0.95, 0.99], value=0.95)
    roll = st.select_slider("Rolling window (trading days)", [21, 63, 126, 252], value=63)
    load_extra = st.checkbox("Load benchmarks & universe", value=False,
                             help="Fetches about 13 extra series from Yahoo (S&P 500, OMXC25, 10 stocks, peers). "
                                  "Leave off if Yahoo is rate-limiting you.")
    st.divider()
    st.markdown("**UNIVERSE**")
    for t, item in STOCKS.items():
        st.caption(f"`{t}`  {item['name']}")
    st.divider()
    if st.button("↻ Refresh market data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data: Yahoo Finance via yfinance. Market data may be delayed.")

# ---------------------------------------------------------------- data pipeline
stock = STOCKS[selected]
with st.spinner(f"Loading {selected} market data..."):
    raw, fx, info = get_data(selected, period), get_rate(CUR, period), get_info(selected)
    uni = get_universe(period, CUR) if load_extra else pd.DataFrame()

if raw.empty:
    st.error("No price history was returned for this ticker. Try refreshing or selecting another security.")
    st.stop()
if fx.empty:
    st.error(f"Could not retrieve USD/{CUR} exchange-rate history. {CUR} conversion is unavailable, so values are not being silently shown in USD.")
    st.stop()

fx = align_rate(fx, raw.index)
if fx.isna().all():
    st.error("Exchange-rate data could not be aligned to the selected stock history.")
    st.stop()
data = raw.copy()
for col in ["Open", "High", "Low", "Close", "Adj Close"]:
    if col in data.columns:
        data[col] = data[col] * fx
data["Volume"] = raw["Volume"]

M = an.calculate_metrics(data, rf, conf)
tech = an.add_technicals(data)
close = data["Close"].dropna()
returns = an.calculate_returns(close)
latest = close.iloc[-1]
previous = close.iloc[-2] if len(close) > 1 else latest
change = (latest / previous - 1) * 100 if previous else 0
company_name = info.get("longName") or stock["name"]
fx_last = float(fx.dropna().iloc[-1])

# benchmarks (all in the display currency)
close_n = _naive(close)
ret_n = close_n.pct_change().dropna()
peer_ret = None
if not uni.empty and selected in uni.columns and uni.shape[1] > 1:
    peer_ret = _naive(uni.pct_change().drop(columns=selected).mean(axis=1))
prices3 = build_prices3(close_n, period, CUR) if load_extra else None
EXTRA_MSG = "Turn on 'Load benchmarks & universe' in the sidebar to use this tab."

# ---------------------------------------------------------------- header
st.markdown(f"<div class='small-muted'>MARKETPULSE / EQUITIES / {selected}</div>", unsafe_allow_html=True)
left, right = st.columns([4, 1])
with left:
    st.title(company_name)
    st.markdown(f"<span class='small-muted'>{selected} · {stock['sector']} · {stock['exchange']}</span>", unsafe_allow_html=True)
with right:
    st.markdown(f"<div style='text-align:right;padding-top:20px;color:#8fa1b5'>QUOTE CURRENCY<br><b style='color:#36d399;font-size:1.15rem'>{CUR}</b></div>", unsafe_allow_html=True)

mc = _f(info.get("marketCap")) * fx_last
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(f"PRICE ({CUR})", num(latest), f"{change:+.2f}%")
c2.metric(f"MKT CAP ({CUR})", fmt_big(mc).replace(f"{CUR} ", ""))
c3.metric("P/E RATIO", num(_f(info.get("trailingPE"))))
c4.metric("52W HIGH", num(M.get("high_52w", np.nan)))
c5.metric("52W LOW", num(M.get("low_52w", np.nan)))
sig = an.signal_table(data, tech)
comp = sig["Score"].mean() if not sig.empty else np.nan
c6.metric("SIGNAL", "—" if not _ok(comp) else "Bullish" if comp > .3 else "Bearish" if comp < -.3 else "Neutral",
          None if not _ok(comp) else f"score {comp:+.2f}", delta_color="off")

st.divider()
tabs = st.tabs(["◈ Overview", "⌁ Technicals & Backtest", "⚠ Risk", "β Market model",
                "▥ Benchmarks", "◐ Portfolio", "🎲 Simulation", "▤ Fundamentals & Profile"])
t_over, t_tech, t_risk, t_mkt, t_cmp, t_port, t_sim, t_fund = tabs

# ================================================================ OVERVIEW
with t_over:
    o1, o2 = st.columns([1, 4])
    with o1:
        chart_type = st.radio("CHART STYLE", ["Line", "Candlestick"])
        overlays = st.multiselect("OVERLAYS", ["SMA 20", "SMA 50", "SMA 200", "Bollinger Bands"], default=["SMA 50", "SMA 200"])
        panels = st.multiselect("PANELS", ["Volume", "RSI", "MACD"], default=["Volume", "RSI", "MACD"])
    with o2:
        rule = RULE[interval]
        cd = data if rule is None else data.resample(rule).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        n = 1 + len(panels)
        heights = [0.55] + [0.45 / max(len(panels), 1)] * len(panels)
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=cd.index, y=cd["Close"], name="Close", line=dict(color=GREEN, width=2.2)), row=1, col=1)
        else:
            fig.add_trace(go.Candlestick(x=cd.index, open=cd["Open"], high=cd["High"], low=cd["Low"], close=cd["Close"],
                                         increasing_line_color=GREEN, decreasing_line_color=RED, name=selected), row=1, col=1)
        for name, col, colr in (("SMA 20", "SMA20", AMBER), ("SMA 50", "SMA50", BLUE), ("SMA 200", "SMA200", PURPLE)):
            if name in overlays:
                fig.add_trace(go.Scatter(x=tech.index, y=tech[col], name=name, line=dict(color=colr, width=1.3)), row=1, col=1)
        if "Bollinger Bands" in overlays:
            fig.add_trace(go.Scatter(x=tech.index, y=tech["BB_up"], name="BB upper", line=dict(color=GREY, width=.8, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=tech.index, y=tech["BB_lo"], name="BB lower", line=dict(color=GREY, width=.8, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(143,161,181,.07)"), row=1, col=1)
        r_i = 2
        if "Volume" in panels:
            vc = np.where(data["Close"].diff() >= 0, GREEN, RED)
            fig.add_trace(go.Bar(x=data.index, y=data["Volume"], marker_color=vc, name="Volume", opacity=.7), row=r_i, col=1)
            r_i += 1
        if "RSI" in panels:
            fig.add_trace(go.Scatter(x=tech.index, y=tech["RSI"], name="RSI", line=dict(color=AMBER, width=1.4)), row=r_i, col=1)
            for lvl in (30, 70):
                fig.add_hline(y=lvl, line_dash="dot", line_color=RED if lvl == 70 else GREEN, row=r_i, col=1)
            r_i += 1
        if "MACD" in panels:
            fig.add_trace(go.Bar(x=tech.index, y=tech["MACD_hist"], name="Hist",
                                 marker_color=np.where(tech["MACD_hist"] >= 0, GREEN, RED), opacity=.6), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=tech.index, y=tech["MACD"], name="MACD", line=dict(color=BLUE, width=1.2)), row=r_i, col=1)
            fig.add_trace(go.Scatter(x=tech.index, y=tech["MACD_sig"], name="Signal", line=dict(color=AMBER, width=1.2)), row=r_i, col=1)
        style(fig, 330 + 150 * len(panels), hovermode="x unified", showlegend=True,
              legend=dict(orientation="h", y=1.04))
        fig.update_xaxes(rangeslider_visible=False, showgrid=False)
        fig.update_yaxes(title_text=f"Price ({CUR})", row=1, col=1)
        st.plotly_chart(fig, width="stretch")
    st.caption("Prices are in USD (source currency). Indicators are computed on daily data." if CUR == "USD" else f"Prices converted from USD to {CUR} at the historical USD/{CUR} rate. Indicators are computed on daily {CUR} data.")

# ================================================================ TECHNICALS & BACKTEST
with t_tech:
    st.markdown("#### Signal dashboard")
    if sig.empty:
        st.info("Not enough history for technical signals. Choose a longer window.")
    else:
        st.dataframe(sig.drop(columns="Score"), width="stretch", hide_index=True)
        st.caption("Rule-based and mechanical (e.g. RSI<30 = bullish/oversold, %B>1 = bearish/stretched). "
                   "The composite is the mean of ±1 scores. It is a summary of indicator state, not a forecast.")
    if "ATR" in tech:
        a1, a2 = st.columns(2)
        a1.metric("ATR (14d)", dkk(tech["ATR"].iloc[-1]))
        a2.metric("ATR as % of price", pct(tech["ATR"].iloc[-1] / latest))

    st.markdown("#### SMA crossover backtest vs buy & hold")
    b1, b2, b3 = st.columns(3)
    fast = b1.slider("Fast SMA", 5, 100, 50)
    slow = b2.slider("Slow SMA", 50, 300, 200, step=10)
    cost = b3.slider("Transaction cost (bps per switch)", 0, 50, 5)
    if fast >= slow:
        st.warning("Fast SMA must be shorter than slow SMA.")
    elif len(close) < slow + 60:
        st.info("Window too short for this slow SMA. Choose a longer historical window.")
    else:
        bt = an.backtest_sma(close, fast, slow, cost)
        valid = bt["strat_eq"].index[slow:]
        se, be = bt["strat_ret"].loc[valid], bt["bh_ret"].loc[valid]
        ss, bs = an.series_stats(se, rf), an.series_stats(be, rf)
        f = go.Figure()
        f.add_trace(go.Scatter(x=valid, y=(1 + se).cumprod() * 100, name=f"SMA {fast}/{slow}", line=dict(color=GREEN, width=2)))
        f.add_trace(go.Scatter(x=valid, y=(1 + be).cumprod() * 100, name="Buy & hold", line=dict(color=GREY, width=1.8)))
        style(f, 340, yaxis_title=f"Growth of 100 {CUR}", hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(f, width="stretch")
        res = pd.DataFrame({"Strategy": ss, "Buy & hold": bs}).T
        res = res.rename(columns={"total_return": "Total return", "cagr": "CAGR", "volatility": "Volatility",
                                  "sharpe": "Sharpe", "max_drawdown": "Max drawdown"})
        for col in ["Total return", "CAGR", "Volatility", "Max drawdown"]:
            res[col] = res[col].map(pct)
        res["Sharpe"] = res["Sharpe"].map(num)
        st.dataframe(res, width="stretch")
        st.caption(f"Switches: {bt['trades']} · Time in market: {bt['time_in_market']:.0%} · Long/flat, signal lagged one day, "
                   "in-sample (no parameter optimisation guard). Past results do not imply future performance.")

# ================================================================ RISK
with t_risk:
    if not M:
        st.info("Not enough data to compute risk statistics.")
    else:
        st.markdown("#### Performance & risk")
        r1 = st.columns(4)
        r1[0].metric("CAGR", pct(M["cagr"]))
        r1[1].metric("ANNUALIZED VOLATILITY", pct(M["volatility"]))
        r1[2].metric("SHARPE", num(M["sharpe"]), help=f"Risk-free rate {rf:.2%}")
        r1[3].metric("SORTINO", num(M["sortino"]), help="Penalises downside deviation only")
        r2 = st.columns(4)
        r2[0].metric("MAX DRAWDOWN", pct(M["max_drawdown"]))
        r2[1].metric("CALMAR", num(M["calmar"]), help="CAGR / |max drawdown|")
        r2[2].metric(f"1-DAY VaR {conf:.0%}", pct(M["var_hist"]), help="Historical simulation")
        r2[3].metric(f"1-DAY CVaR {conf:.0%}", pct(M["cvar_hist"]), help="Average loss on days beyond VaR")

        st.markdown("#### Return distribution & tail risk")
        d1, d2 = st.columns([3, 2])
        with d1:
            x = returns.values
            hist = go.Figure(go.Histogram(x=x * 100, nbinsx=70, histnorm="probability density",
                                          marker_color=BLUE, opacity=.75, name="Daily returns"))
            grid = np.linspace(x.min(), x.max(), 300)
            hist.add_trace(go.Scatter(x=grid * 100, y=stats.norm.pdf(grid, x.mean(), x.std()) / 100,
                                      name="Normal fit", line=dict(color=AMBER, width=2)))
            hist.add_vline(x=-M["var_hist"] * 100, line_dash="dash", line_color=RED,
                           annotation_text=f"VaR {conf:.0%}")
            style(hist, 340, xaxis_title="Daily return (%)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(hist, width="stretch")
        with d2:
            tr = pd.DataFrame({"Method": ["Historical", "Gaussian", "Cornish-Fisher"],
                               f"VaR {conf:.0%}": [pct(M["var_hist"]), pct(M["var_norm"]), pct(M["var_cf"])]})
            st.dataframe(tr, width="stretch", hide_index=True)
            st.metric("Skewness", num(M["skew"]))
            st.metric("Excess kurtosis", num(M["kurt"]))
            verdict = "rejected" if M["jb_p"] < 0.05 else "not rejected"
            st.caption(f"Jarque-Bera normality {verdict} (p = {M['jb_p']:.3g}). "
                       + (f"A Gaussian model understates historical VaR by {(1 - M['var_norm'] / M['var_hist']):.0%}."
                          if M["var_hist"] > M["var_norm"] else "Gaussian VaR is not understating risk in this window."))
            st.caption(f"Win rate {M['win_rate']:.1%} · Best day {pct(M['best_day'], sign=True)} · Worst day {pct(M['worst_day'], sign=True)}")

        st.markdown("#### Drawdowns")
        dd = an.calculate_drawdown(close)
        dd_fig = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=RED, width=1.6), name="Drawdown"))
        style(dd_fig, 260, yaxis_title="Drawdown (%)", showlegend=False)
        st.plotly_chart(dd_fig, width="stretch")
        ep = an.drawdown_episodes(close)
        if not ep.empty:
            ep = ep.assign(Peak=ep["Peak"].dt.strftime("%Y-%m-%d"), Trough=ep["Trough"].dt.strftime("%Y-%m-%d"),
                           Recovered=ep["Recovered"].dt.strftime("%Y-%m-%d").fillna("ongoing"),
                           Depth=ep["Depth"].map(pct)).rename(columns={"Days_to_trough": "Days to trough", "Days_underwater": "Days underwater"})
            st.dataframe(ep, width="stretch", hide_index=True)

        st.markdown(f"#### Rolling risk ({roll}-day window)")
        rv, rs_ = an.rolling_vol(returns, roll), an.rolling_sharpe(returns, roll, rf)
        rf_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.08,
                               subplot_titles=("Annualised volatility", "Sharpe ratio"))
        rf_fig.add_trace(go.Scatter(x=rv.index, y=rv * 100, line=dict(color=AMBER), name="Vol %"), row=1, col=1)
        rf_fig.add_trace(go.Scatter(x=rs_.index, y=rs_, line=dict(color=GREEN), name="Sharpe"), row=2, col=1)
        rf_fig.add_hline(y=0, line_dash="dot", line_color=GREY, row=2, col=1)
        style(rf_fig, 420, showlegend=False)
        st.plotly_chart(rf_fig, width="stretch")

        st.markdown("#### Monthly returns")
        mt = an.monthly_returns(close)
        hm = px.imshow(mt * 100, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn",
                       color_continuous_midpoint=0, labels=dict(color="%"))
        style(hm, max(240, 50 * len(mt) + 80))
        st.plotly_chart(hm, width="stretch")

        if CUR == "USD":
            st.caption("Currency effect not applicable: prices are shown in the source currency (USD).")
        else:
            st.markdown(f"#### Currency effect (USD asset, {CUR} investor)")
            usd = raw["Close"].dropna()
            us, ds = an.series_stats(usd.pct_change().dropna(), rf), an.series_stats(returns, rf)
            k = st.columns(4)
            k[0].metric("Total return (USD)", pct(us["total_return"]))
            k[1].metric(f"Total return ({CUR})", pct(ds["total_return"]))
            k[2].metric("FX impact", f"{(ds['total_return'] - us['total_return']) * 100:+.1f} pp")
            k[3].metric(f"Volatility USD → {CUR}", f"{us['volatility']:.1%} → {ds['volatility']:.1%}")
            st.caption(f"A positive impact means USD strengthened against {CUR} over the window. A negative one means the {CUR} investor lost to FX.")

# ================================================================ MARKET MODEL
with t_mkt:
    st.markdown("#### CAPM regression")
    opts = {}
    if prices3 is not None:
        opts["S&P 500"] = prices3["S&P 500"].pct_change().dropna()
        opts["OMXC25"] = prices3["OMXC25"].pct_change().dropna()
    if peer_ret is not None:
        opts["Peer basket"] = peer_ret
    bench = None
    if opts:
        bname = st.radio("BENCHMARK", list(opts), horizontal=True)
        aligned = pd.concat([ret_n, opts[bname]], axis=1, join="inner").dropna()
        aligned.columns = ["asset", "bench"]
        bench = opts[bname] if len(aligned) >= 60 else None
    if bench is None:
        st.info(EXTRA_MSG if not load_extra else "Benchmark unavailable (index data could not be loaded or too little overlap).")
    else:
        mm = an.market_model(aligned["asset"], aligned["bench"], rf)
        st.caption(f"Benchmark: {bname}, in {CUR}. " + ("Equal-weighted average of the other stocks in the universe, a peer-group proxy." if bname == "Peer basket"
                   else "Daily returns on shared trading days only."))
        m1 = st.columns(4)
        m1[0].metric("BETA", num(mm["beta"]), help=f"Std. error {mm['beta_se']:.3f}")
        m1[1].metric("JENSEN'S ALPHA (ann.)", pct(mm["alpha_ann"], sign=True),
                     help=f"p-value {mm['alpha_p']:.3f}. Alpha is rarely statistically significant.")
        m1[2].metric("R²", num(mm["r2"]), help="Share of variance explained by the benchmark")
        m1[3].metric("CORRELATION", num(mm["corr"]))
        m2 = st.columns(4)
        m2[0].metric("TRACKING ERROR", pct(mm["tracking_error"]))
        m2[1].metric("INFORMATION RATIO", num(mm["info_ratio"]))
        m2[2].metric("UP CAPTURE", pct(mm["up_capture"], 0))
        m2[3].metric("DOWN CAPTURE", pct(mm["down_capture"], 0))
        s1, s2 = st.columns(2)
        with s1:
            sc = go.Figure(go.Scattergl(x=aligned["bench"] * 100, y=aligned["asset"] * 100, mode="markers",
                                        marker=dict(color=BLUE, size=4, opacity=.45), name="Days"))
            xs = np.linspace(aligned["bench"].min(), aligned["bench"].max(), 50)
            sc.add_trace(go.Scatter(x=xs * 100, y=(mm["intercept"] + mm["beta"] * xs) * 100 + rf / an.TD * 100 * (1 - mm["beta"]),
                                    line=dict(color=AMBER, width=2), name="Regression"))
            style(sc, 360, xaxis_title="Benchmark daily return (%)", yaxis_title=f"{selected} daily return (%)",
                  legend=dict(orientation="h", y=1.1))
            st.plotly_chart(sc, width="stretch")
        with s2:
            rb = an.rolling_beta(aligned["asset"], aligned["bench"], roll)
            rc = aligned["asset"].rolling(roll).corr(aligned["bench"])
            rbf = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.1,
                                subplot_titles=(f"Rolling beta ({roll}d)", f"Rolling correlation ({roll}d)"))
            rbf.add_trace(go.Scatter(x=rb.index, y=rb, line=dict(color=PURPLE)), row=1, col=1)
            rbf.add_hline(y=1, line_dash="dot", line_color=GREY, row=1, col=1)
            rbf.add_trace(go.Scatter(x=rc.index, y=rc, line=dict(color=GREEN)), row=2, col=1)
            style(rbf, 360, showlegend=False)
            st.plotly_chart(rbf, width="stretch")

# ================================================================ COMPARE
with t_cmp:
    st.markdown(f"#### {selected} vs S&P 500 and OMXC25 ({CUR})")
    if prices3 is None:
        st.warning(EXTRA_MSG if not load_extra else "Benchmark data could not be loaded. It needs ^GSPC and ^OMXC25 from Yahoo Finance plus the FX rate.")
    else:
        disp = prices3.rename(columns={"Selected stock": selected})
        cum = bm.cumulative_returns(disp)
        cf = px.line(cum, x=cum.index, y=cum.columns,
                     labels={"value": f"Indexed performance (start = 100, {CUR})", "variable": "", "x": "Date"})
        style(cf, 400, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(cf, width="stretch")
        st.caption(f"{prices3.index[0].date()} to {prices3.index[-1].date()} · {len(prices3)} shared trading days "
                   "(only dates when all three markets were open). S&P 500 is converted from USD, OMXC25 from DKK.")
        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown("##### Performance ranking")
            rk = bm.performance_ranking(disp).copy()
            rk["Total return"] = rk["Total return"].map(pct)
            rk["CAGR"] = rk["CAGR"].map(pct)
            st.dataframe(rk, width="stretch")
        with c_r:
            cm = px.imshow(bm.correlation_matrix(disp), text_auto=".2f", aspect="auto", color_continuous_scale="RdYlGn",
                           zmin=-1, zmax=1, title="Daily return correlation")
            style(cm, 260)
            st.plotly_chart(cm, width="stretch")
        st.markdown(f"##### Rolling Sharpe ({roll}-day window)")
        rsh = bm.rolling_sharpe(disp, window=max(roll, 2), risk_free_rate=rf)
        rsf = px.line(rsh, x=rsh.index, y=rsh.columns, labels={"value": "Sharpe", "variable": "", "x": "Date"})
        rsf.add_hline(y=0, line_dash="dot", line_color=GREY)
        style(rsf, 320, hovermode="x unified", legend=dict(orientation="h", y=1.1))
        st.plotly_chart(rsf, width="stretch")
        st.markdown("##### Regression vs each benchmark")
        rows = {}
        for b in ("S&P 500", "OMXC25"):
            r_ = bm.benchmark_regression(prices3["Selected stock"], prices3[b], rf)
            rows[b] = {"Beta": num(r_.get("beta")), "Alpha (ann.)": pct(r_.get("alpha_annualized"), sign=True),
                       "R²": num(r_.get("r_squared")), "Alpha p-value": num(r_.get("alpha_p_value"), 3),
                       "Observations": r_.get("observations")}
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
        st.caption("CAPM-style OLS on daily excess returns. Alpha is rarely statistically significant. Check the p-value.")

# ================================================================ PORTFOLIO
with t_port:
    st.markdown("#### Mean-variance optimisation across the universe")
    if uni.empty or uni.shape[1] < 3:
        st.info(EXTRA_MSG if not load_extra else "Needs at least three stocks with loaded price history.")
    else:
        prets = uni.pct_change().dropna(how="any")
        if len(prets) < 60:
            st.warning("Not enough overlapping history.")
        else:
            cap = st.slider("Max weight per stock", 0.15, 1.0, 0.40, 0.05)
            pf = an.optimize_portfolio(prets, rf, cap)
            ff = go.Figure()
            ff.add_trace(go.Scattergl(x=pf["rand_vol"], y=pf["rand_ret"], mode="markers", name="Random portfolios",
                                      marker=dict(size=4, color=pf["rand_sharpe"], colorscale="Viridis", opacity=.6,
                                                  colorbar=dict(title="Sharpe"))))
            ff.add_trace(go.Scatter(x=pf["assets"]["vol"], y=pf["assets"]["ret"], mode="markers+text", text=pf["tickers"],
                                    textposition="top center", marker=dict(size=9, color=GREY), name="Single stocks"))
            for key, lab, colr, sym in (("max_sharpe", "Max Sharpe", GREEN, "star"), ("min_vol", "Min volatility", BLUE, "diamond"),
                                        ("equal", "Equal weight", AMBER, "square")):
                ff.add_trace(go.Scatter(x=[pf[key]["vol"]], y=[pf[key]["ret"]], mode="markers", name=lab,
                                        marker=dict(size=16, color=colr, symbol=sym, line=dict(width=1, color="white"))))
            style(ff, 480, xaxis_title="Annualised volatility", yaxis_title="Annualised return",
                  legend=dict(orientation="h", y=1.08))
            ff.update_xaxes(tickformat=".0%")
            ff.update_yaxes(tickformat=".0%")
            st.plotly_chart(ff, width="stretch")
            w = pd.DataFrame({"Max Sharpe": pf["max_sharpe"]["w"], "Min volatility": pf["min_vol"]["w"], "Equal weight": pf["equal"]["w"]})
            wf = px.bar(w, barmode="group", labels={"value": "Weight", "index": "Ticker"})
            wf.update_yaxes(tickformat=".0%")
            style(wf, 320, legend=dict(orientation="h", y=1.15))
            st.plotly_chart(wf, width="stretch")
            sm = pd.DataFrame({k: {"Return": pct(v["ret"]), "Volatility": pct(v["vol"]), "Sharpe": num(v["sharpe"])}
                               for k, v in (("Max Sharpe", pf["max_sharpe"]), ("Min volatility", pf["min_vol"]), ("Equal weight", pf["equal"]))}).T
            st.dataframe(sm, width="stretch")
            st.caption("In-sample optimisation on historical means is highly sensitive to estimation error, so weights are indicative only. "
                       "Long-only, fully invested, weight cap applied.")

# ================================================================ SIMULATION
with t_sim:
    st.markdown(f"#### Monte Carlo price simulation · {selected}")
    s1, s2, s3, s4 = st.columns(4)
    hz = s1.slider("Horizon (trading days)", 21, 504, 252, step=21)
    npaths = s2.select_slider("Paths", [500, 1000, 2000, 5000], value=2000)
    demean = s3.checkbox("Neutralise historical drift", value=False,
                         help="Historical mean returns can be extreme. Zeroing drift isolates pure volatility risk.")
    seed = s4.number_input("Seed", 0, 9999, 42)
    if len(close) < 60:
        st.info("Not enough history.")
    else:
        paths = an.monte_carlo(close, hz, npaths, demean, int(seed))
        pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
        xs = list(range(hz + 1))
        mf = go.Figure()
        for j in range(min(npaths, 40)):
            mf.add_trace(go.Scatter(x=xs, y=paths[:, j], line=dict(color="rgba(143,161,181,.12)", width=1), showlegend=False, hoverinfo="skip"))
        for lo, hi, a in ((0, 4, .12), (1, 3, .22)):
            mf.add_trace(go.Scatter(x=xs, y=pcts[hi], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            mf.add_trace(go.Scatter(x=xs, y=pcts[lo], fill="tonexty", fillcolor=f"rgba(54,211,153,{a})", line=dict(width=0),
                                    name="5–95% band" if lo == 0 else "25–75% band"))
        mf.add_trace(go.Scatter(x=xs, y=pcts[2], line=dict(color=GREEN, width=2.5), name="Median"))
        mf.add_hline(y=latest, line_dash="dot", line_color=AMBER, annotation_text="Today")
        style(mf, 440, xaxis_title="Trading days ahead", yaxis_title=f"Price ({CUR})", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(mf, width="stretch")
        term = paths[-1] / latest - 1
        k = st.columns(5)
        k[0].metric("Median outcome", pct(np.median(term), sign=True))
        k[1].metric("5th percentile", pct(np.percentile(term, 5), sign=True))
        k[2].metric("95th percentile", pct(np.percentile(term, 95), sign=True))
        k[3].metric("P(loss)", pct((term < 0).mean(), 1))
        k[4].metric("P(loss > 20%)", pct((term < -0.2).mean(), 1))
        st.caption("Bootstrap of historical daily log returns (i.i.d.), which preserves fat tails but ignores volatility clustering and regime changes. "
                   "Illustrative distribution, not a forecast.")

# ================================================================ FUNDAMENTALS & PROFILE
with t_fund:
    st.markdown("#### Valuation & operating metrics")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Forward P/E", num(_f(info.get("forwardPE"))))
    f2.metric("Price / Book", num(_f(info.get("priceToBook"))))
    f3.metric("EPS (reported)", num(_f(info.get("trailingEps"))))
    dy = _f(info.get("dividendYield"))
    f4.metric("Dividend yield", f"{dy * 100:,.2f}%" if _ok(dy) else "—")
    f5, f6, f7, f8 = st.columns(4)
    f5.metric("Revenue (TTM)", fmt_big(_f(info.get("totalRevenue")) * fx_last))
    f6.metric("Net income (TTM)", fmt_big(_f(info.get("netIncomeToCommon")) * fx_last))
    pm = _f(info.get("profitMargins"))
    f7.metric("Profit margin", f"{pm * 100:,.2f}%" if _ok(pm) else "—")
    f8.metric("Beta (Yahoo)", num(_f(info.get("beta"))))

    peers = get_peers() if load_extra else pd.DataFrame(columns=["Ticker", "Name", "Mkt cap (USD bn)", "P/E", "Fwd P/E", "P/B",
                                                                 "Margin %", "ROE %", "Rev growth %", "Div yield %", "Beta"])
    pe = _f(info.get("trailingPE"))
    st.markdown("#### Relative valuation & earnings yield")
    v = st.columns(4)
    med = peers["P/E"].median()
    v[0].metric("P/E vs peer median", pct(pe / med - 1, 0, sign=True) if _ok(pe) and _ok(med) else "—", help=f"Peer median P/E: {num(med)}")
    ey = 1 / pe if _ok(pe) and pe > 0 else np.nan
    v[1].metric("Earnings yield", pct(ey))
    v[2].metric("Earnings yield − risk-free", pct(ey - rf, sign=True) if _ok(ey) else "—",
                help="Crude equity risk premium proxy (USD earnings vs. your chosen risk-free rate)")
    roe = _f(info.get("returnOnEquity"))
    v[3].metric("Return on equity", pct(roe))

    tgt = _f(info.get("targetMeanPrice"))
    usd_px = float(raw["Close"].dropna().iloc[-1])
    st.markdown("#### Analyst consensus")
    a = st.columns(4)
    a[0].metric("Mean target (USD)", num(tgt))
    a[1].metric("Implied upside", pct(tgt / usd_px - 1, 1, sign=True) if _ok(tgt) else "—")
    a[2].metric("Recommendation", str(info.get("recommendationKey") or "—").replace("_", " ").title())
    a[3].metric("# analysts", num(_f(info.get("numberOfAnalystOpinions")), 0))

    st.markdown("#### Peer table")
    pt = peers.copy()
    pt["Selected"] = np.where(pt["Ticker"] == selected, "◀", "")
    st.dataframe(pt.round(2), width="stretch", hide_index=True)
    pv = peers.dropna(subset=["P/E", "Margin %", "Mkt cap (USD bn)"]).copy()
    if len(pv) >= 3:
        pv["Group"] = np.where(pv["Ticker"] == selected, "Selected", "Peer")
        vf = px.scatter(pv, x="P/E", y="Margin %", size="Mkt cap (USD bn)", color="Group", text="Ticker",
                        color_discrete_map={"Selected": GREEN, "Peer": BLUE}, title="Valuation vs profitability (bubble = market cap)")
        vf.update_traces(textposition="top center")
        style(vf, 420, showlegend=False)
        st.plotly_chart(vf, width="stretch")
    st.caption("Fundamental fields depend on Yahoo Finance coverage. Monetary fundamentals converted to the display currency; peer table in source currency.")

    st.markdown("#### Business overview")
    st.write(info.get("longBusinessSummary") or "A company description is not available from the data provider.")
    p1, p2, p3 = st.columns(3)
    p1.metric("SECTOR", info.get("sector") or stock["sector"])
    p2.metric("INDUSTRY", info.get("industry") or "—")
    emp = info.get("fullTimeEmployees")
    p3.metric("EMPLOYEES", f"{emp:,}" if emp else "—")
    st.write(f"**Website:** {info.get('website') or '—'}")
    st.write(f"**Headquarters:** {info.get('city') or '—'}, {info.get('country') or '—'}")

st.divider()
st.caption("MARKETPULSE · Educational analytics project · Not investment advice. Data availability, freshness and accuracy are not guaranteed.")