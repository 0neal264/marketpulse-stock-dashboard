"""Quant analytics for MarketPulse: performance, tail risk, technicals, market model,
backtesting, Monte Carlo and portfolio optimisation. All functions are pure (no Streamlit)."""
import calendar

import numpy as np
import pandas as pd
from scipy import optimize, stats

TD = 252  # trading days per year


# ---------------------------------------------------------------- basics
def calculate_returns(close):
    return close.pct_change().dropna()


def calculate_drawdown(close):
    return close / close.cummax() - 1


def correlation_matrix(prices):
    return prices.pct_change().dropna().corr()


def series_stats(ret, rf=0.0):
    """Annualised stats from a daily return series."""
    ret = ret.dropna()
    if len(ret) < 2:
        return {}
    eq = (1 + ret).cumprod()
    cagr = eq.iloc[-1] ** (TD / len(ret)) - 1
    vol = ret.std() * np.sqrt(TD)
    return dict(
        total_return=eq.iloc[-1] - 1,
        cagr=cagr,
        volatility=vol,
        sharpe=(ret.mean() * TD - rf) / vol if vol > 0 else np.nan,
        max_drawdown=(eq / eq.cummax() - 1).min(),
    )


def tail_risk(r, conf=0.95):
    """Historical, Gaussian and Cornish-Fisher VaR plus historical CVaR (all as positive loss fractions)."""
    a = 1 - conf
    mu, sd = r.mean(), r.std()
    q = np.percentile(r, a * 100)
    s, k = stats.skew(r), stats.kurtosis(r)
    z = stats.norm.ppf(a)
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return dict(
        var_hist=-q,
        cvar_hist=-r[r <= q].mean(),
        var_norm=-(mu + z * sd),
        var_cf=-(mu + zcf * sd),
    )


def calculate_metrics(data, rf=0.0, conf=0.95):
    c = data["Close"].dropna()
    r = c.pct_change().dropna()
    if len(r) < 30:
        return {}
    m = series_stats(r, rf)
    dsd = np.sqrt(np.mean(np.minimum(r - rf / TD, 0) ** 2)) * np.sqrt(TD)
    m["downside_vol"] = dsd
    m["sortino"] = (r.mean() * TD - rf) / dsd if dsd > 0 else np.nan
    m["calmar"] = m["cagr"] / abs(m["max_drawdown"]) if m["max_drawdown"] < 0 else np.nan
    m["return_1y"] = c.iloc[-1] / c.iloc[-253] - 1 if len(c) > 252 else np.nan
    hi = data["High"] if "High" in data else c
    lo = data["Low"] if "Low" in data else c
    m["high_52w"], m["low_52w"] = hi.tail(TD).max(), lo.tail(TD).min()
    m["skew"], m["kurt"] = stats.skew(r), stats.kurtosis(r)
    m["jb_p"] = stats.jarque_bera(r).pvalue
    m["win_rate"], m["best_day"], m["worst_day"] = (r > 0).mean(), r.max(), r.min()
    m.update(tail_risk(r, conf))
    return m


def drawdown_episodes(close, top=5):
    dd = calculate_drawdown(close)
    grp = (dd >= 0).cumsum()
    last, rows = grp.max(), []
    for g, sub in dd.groupby(grp):
        if sub.min() >= 0:
            continue
        peak, trough = sub.index[0], sub.idxmin()
        rec = dd[grp == g + 1].index[0] if g < last else pd.NaT
        end = rec if pd.notna(rec) else dd.index[-1]
        rows.append(dict(Peak=peak, Trough=trough, Recovered=rec, Depth=sub.min(),
                         Days_to_trough=(trough - peak).days, Days_underwater=(end - peak).days))
    cols = ["Peak", "Trough", "Recovered", "Depth", "Days_to_trough", "Days_underwater"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("Depth").head(top).reset_index(drop=True)


def monthly_returns(close):
    m = close.groupby([close.index.year, close.index.month]).last().pct_change().unstack()
    m.columns = [calendar.month_abbr[i] for i in m.columns]
    m.index = m.index.astype(str)
    return m


# ---------------------------------------------------------------- rolling
def rolling_vol(r, w):
    return r.rolling(w).std() * np.sqrt(TD)


def rolling_sharpe(r, w, rf=0.0):
    return (r.rolling(w).mean() * TD - rf) / (r.rolling(w).std() * np.sqrt(TD))


def rolling_beta(r, b, w):
    return r.rolling(w).cov(b) / b.rolling(w).var()


# ---------------------------------------------------------------- technicals
def add_technicals(df):
    c = df["Close"]
    out = pd.DataFrame(index=df.index)
    for n in (20, 50, 200):
        out[f"SMA{n}"] = c.rolling(n).mean()
    sd = c.rolling(20).std()
    out["BB_mid"], out["BB_up"], out["BB_lo"] = out["SMA20"], out["SMA20"] + 2 * sd, out["SMA20"] - 2 * sd
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + rs)
    out["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["MACD_sig"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_sig"]
    if {"High", "Low"} <= set(df.columns):
        pc = c.shift()
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
        out["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return out


def signal_table(data, tech):
    """Rule-based signal dashboard. Scores: +1 bullish, 0 neutral, -1 bearish."""
    c, t = data["Close"], tech.iloc[-1]
    rows = []

    def add(name, reading, score):
        label = {1: "Bullish", 0: "Neutral", -1: "Bearish"}[score]
        rows.append({"Indicator": name, "Reading": reading, "Signal": label, "Score": score})

    def ok(*v):
        return all(pd.notna(x) for x in v)

    if ok(t["SMA200"]):
        add("Price vs 200d SMA", f"{c.iloc[-1] / t['SMA200'] - 1:+.1%}", 1 if c.iloc[-1] > t["SMA200"] else -1)
    if ok(t["SMA50"], t["SMA200"]):
        add("50d vs 200d (golden/death cross)", f"{t['SMA50'] / t['SMA200'] - 1:+.1%}", 1 if t["SMA50"] > t["SMA200"] else -1)
    if ok(t["RSI"]):
        add("RSI (14)", f"{t['RSI']:.1f}", 1 if t["RSI"] < 30 else -1 if t["RSI"] > 70 else 0)
    if ok(t["MACD"], t["MACD_sig"]):
        add("MACD vs signal", f"{t['MACD_hist']:+.3f}", 1 if t["MACD"] > t["MACD_sig"] else -1)
    if ok(t["BB_up"], t["BB_lo"]):
        pb = (c.iloc[-1] - t["BB_lo"]) / (t["BB_up"] - t["BB_lo"])
        add("Bollinger %B", f"{pb:.2f}", 1 if pb < 0 else -1 if pb > 1 else 0)
    if len(c) > TD:
        mom = c.iloc[-22] / c.iloc[-TD] - 1
        add("12-1 month momentum", f"{mom:+.1%}", 1 if mom > 0 else -1)
    rng = c.tail(TD)
    pos = (c.iloc[-1] - rng.min()) / (rng.max() - rng.min()) if rng.max() > rng.min() else np.nan
    if ok(pos):
        add("Position in 52w range", f"{pos:.0%}", 1 if pos > 0.66 else -1 if pos < 0.33 else 0)
    return pd.DataFrame(rows)


def backtest_sma(close, fast=50, slow=200, cost_bps=5):
    """Long/flat SMA crossover with one-day signal lag and proportional transaction costs."""
    r = close.pct_change().fillna(0)
    pos = (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float).shift(1).fillna(0)
    turn = pos.diff().abs().fillna(0)
    sr = pos * r - turn * cost_bps / 1e4
    return dict(strat_ret=sr, bh_ret=r, strat_eq=(1 + sr).cumprod(), bh_eq=(1 + r).cumprod(),
                position=pos, trades=int(turn.sum()), time_in_market=pos.mean())


# ---------------------------------------------------------------- market model
def market_model(asset, bench, rf=0.0):
    """CAPM regression on excess returns plus capture ratios. Inputs: aligned daily return series."""
    ea, eb = asset - rf / TD, bench - rf / TD
    res = stats.linregress(eb, ea)
    active = asset - bench
    up, dn = bench > 0, bench < 0
    return dict(
        beta=res.slope, beta_se=res.stderr, alpha_ann=res.intercept * TD, alpha_p=res.pvalue,
        r2=res.rvalue**2, corr=res.rvalue,
        tracking_error=active.std() * np.sqrt(TD),
        info_ratio=active.mean() * TD / (active.std() * np.sqrt(TD)) if active.std() > 0 else np.nan,
        up_capture=asset[up].mean() / bench[up].mean() if up.any() else np.nan,
        down_capture=asset[dn].mean() / bench[dn].mean() if dn.any() else np.nan,
        intercept=res.intercept,
    )


# ---------------------------------------------------------------- simulation
def monte_carlo(close, horizon=252, n_paths=2000, demean=False, seed=42):
    """Bootstrap Monte Carlo of log returns. Returns array (horizon+1, n_paths) of simulated prices."""
    lr = np.log(close / close.shift(1)).dropna().values
    if demean:
        lr = lr - lr.mean()
    rng = np.random.default_rng(seed)
    draws = lr[rng.integers(0, len(lr), size=(horizon, n_paths))]
    return close.iloc[-1] * np.exp(np.vstack([np.zeros(n_paths), np.cumsum(draws, axis=0)]))


# ---------------------------------------------------------------- portfolio
def optimize_portfolio(returns, rf=0.0, max_w=1.0, n_random=4000, seed=7):
    tick = list(returns.columns)
    mu, cov = returns.mean().values * TD, returns.cov().values * TD
    n = len(tick)
    max_w = max(max_w, 1 / n + 1e-6)

    def perf(w):
        return w @ mu, np.sqrt(w @ cov @ w)

    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    kw = dict(bounds=[(0, max_w)] * n, constraints=cons, method="SLSQP")
    w0 = np.repeat(1 / n, n)
    ms = optimize.minimize(lambda w: -(perf(w)[0] - rf) / perf(w)[1], w0, **kw).x
    mv = optimize.minimize(lambda w: perf(w)[1], w0, **kw).x
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(n), n_random)
    rr = W @ mu
    rv = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
    pack = lambda w: dict(w=pd.Series(w, index=tick), ret=perf(w)[0], vol=perf(w)[1],
                          sharpe=(perf(w)[0] - rf) / perf(w)[1])
    return dict(tickers=tick, rand_ret=rr, rand_vol=rv, rand_sharpe=(rr - rf) / rv,
                max_sharpe=pack(ms), min_vol=pack(mv), equal=pack(w0),
                assets=pd.DataFrame({"ret": mu, "vol": np.sqrt(np.diag(cov))}, index=tick))