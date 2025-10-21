import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tvDatafeed import TvDatafeed, Interval
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from math import sqrt, log, exp

tv = None
try:
    username = st.secrets.get("tv_username")
    password = st.secrets.get("tv_password")
    if username and password:
        tv = TvDatafeed(username=username, password=password)
except Exception:
    tv = None

st.set_page_config(
    page_title="Option Pricing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMetric {background-color: #131a24 !important; padding: 1rem; border-radius: 0.5rem;}
    .stPlotlyChart {border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)


def fetch_underlying_price(tv, symbol, exchange):
    if tv is not None:
        try:
            df = tv.get_hist(symbol=symbol, exchange=exchange,
                             interval=Interval.in_1_minute, n_bars=1)
            if isinstance(df, pd.DataFrame) and not df.empty:
                p = df['close'].iloc[-1]
                if p is not None and not np.isnan(p):
                    return float(p)
        except Exception:
            pass
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2d', interval='1m')
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            p = hist['Close'].dropna().iloc[-1]
            if p is not None and not np.isnan(p):
                return float(p)
    except Exception:
        pass
    try:
        ticker = yf.Ticker(symbol)
        fi = getattr(ticker, "fast_info", None)
        if fi:
            p = None
            if isinstance(fi, dict):
                p = fi.get("last_price") or fi.get("lastPrice") or fi.get("last_trade_price")
            else:
                p = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
            if p is not None and not np.isnan(p):
                return float(p)
    except Exception:
        pass
    return 0.0


def get_option_chain(ticker, expiry_date):
    try:
        oc = yf.Ticker(ticker).option_chain(expiry_date)
        return oc.calls, oc.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def binomial_american_option_price(S, K, r, sigma, T, steps=100, option_type='call'):
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    steps = max(50, min(200, int(T * 365 * 2)))
    steps = max(1, int(steps))
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    stock_prices = np.zeros((steps + 1, steps + 1))
    option_prices = np.zeros((steps + 1, steps + 1))
    stock_prices[0, 0] = S
    for i in range(1, steps + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * d
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * u
    if option_type == 'call':
        option_prices[steps, :] = np.maximum(stock_prices[steps, :] - K, 0)
    else:
        option_prices[steps, :] = np.maximum(K - stock_prices[steps, :], 0)
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = exp(-r * dt) * (p * option_prices[i + 1, j + 1] + (1 - p) * option_prices[i + 1, j])
            if option_type == 'call':
                exercise_value = max(stock_prices[i, j] - K, 0)
            else:
                exercise_value = max(K - stock_prices[i, j], 0)
            option_prices[i, j] = max(hold_value, exercise_value)
    return option_prices[0, 0]


class EuropeanPricer:
    def __init__(self, risk_free_rate=0.05):
        self.r = risk_free_rate

    def d(self, sigma, S, K, t):
        d1 = (log(S / K) + (self.r + (sigma ** 2) / 2) * t) / (sigma * sqrt(t))
        d2 = d1 - sigma * sqrt(t)
        return d1, d2

    def price(self, S, K, sigma, t, option_type='call'):
        if t <= 0 or sigma <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        d1, d2 = self.d(sigma, S, K, t)
        if option_type == 'call':
            return norm.cdf(d1) * S - norm.cdf(d2) * K * exp(-self.r * t)
        else:
            return norm.cdf(-d2) * K * exp(-self.r * t) - norm.cdf(-d1) * S

    def implied_vol(self, market_price, S, K, t, option_type='call'):
        tol = 1e-7
        try:
            if market_price is None or market_price <= 0 or S <= 0 or K <= 0 or t <= 0:
                return 0.0
        except Exception:
            return 0.0
        if option_type == 'call':
            lb = max(0.0, S - K * exp(-self.r * t))
        else:
            lb = max(0.0, K * exp(-self.r * t) - S)
        if market_price < lb - 1e-12:
            return 0.0
        def objective(sig):
            return self.price(S, K, sig, t, option_type) - market_price
        try:
            iv = brentq(objective, 1e-6, 5.0, maxiter=200, xtol=1e-8)
            return iv
        except Exception:
            sigma = 0.5
            max_iter = 100
            for _ in range(max_iter):
                if sigma <= 0:
                    sigma = 1e-6
                try:
                    d1, _ = self.d(sigma, S, K, t)
                except Exception:
                    break
                vega = S * norm.pdf(d1) * sqrt(t)
                if vega < 1e-12:
                    break
                model_price = self.price(S, K, sigma, t, option_type)
                diff = model_price - market_price
                if abs(diff) < tol:
                    break
                sigma -= diff / vega
                sigma = max(1e-6, min(sigma, 5.0))
            return sigma


def plot_iv_smile(strikes, ivs, current_strike=None):
    valid_mask = [(iv > 0.01 and iv < 5.0 and not np.isnan(iv)) for iv in ivs]
    if not any(valid_mask):
        return None
    filtered_strikes = [s for i, s in enumerate(strikes) if valid_mask[i]]
    filtered_ivs = [iv for i, iv in enumerate(ivs) if valid_mask[i]]
    if len(filtered_strikes) < 3:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_strikes,
        y=filtered_ivs,
        mode='lines+markers',
        name='IV Smile',
        line=dict(color='#636EFA', width=3)
    ))
    if current_strike:
        fig.add_vline(
            x=current_strike,
            line=dict(color='#FFA15A', dash='dash'),
            annotation_text='Selected Strike'
        )
    fig.update_layout(
        title='Implied Volatility Smile',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        template='plotly_dark',
        height=400
    )
    return fig


def plot_fit_curve(x, y, x_label, y_label):
    if len(x) < 4:
        return None
    x_arr = np.array(x)
    y_arr = np.array(y)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr) | np.isinf(x_arr) | np.isinf(y_arr))
    if np.sum(mask) < 4:
        return None
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    deg = min(3, len(x_clean) - 1)
    try:
        coeffs = np.polyfit(x_clean, y_clean, deg=deg)
        poly_eq = np.poly1d(coeffs)
        y_fit = poly_eq(x_clean)
        ss_res = np.sum((y_clean - y_fit) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    except Exception:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_clean, y=y_clean,
        mode='markers',
        name='Data',
        marker=dict(color='#00CC96')
    ))
    fig.add_trace(go.Scatter(
        x=x_clean, y=y_fit,
        mode='lines',
        name=f'Fit Curve (R² = {r_squared:.3f})',
        line=dict(color='#EF553B', width=3)
    ))
    fig.update_layout(
        title=f'{y_label} vs {x_label}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_dark',
        height=400
    )
    return fig


st.title("Options Pricing Project")

st.sidebar.header("⚙️ Pricing Parameters")
option_style = st.sidebar.radio("Option Style", ["European", "American"])
symbol = st.sidebar.text_input("Symbol", value="AAPL")
expiry_date = st.sidebar.date_input("Expiry Date", value=datetime.today() + timedelta(days=30))

try:
    ticker = yf.Ticker(symbol)
    expiries = ticker.options
    expiry_date = st.sidebar.selectbox("Select Expiry", expiries, index=min(3, len(expiries) - 1))
except Exception:
    st.sidebar.warning("Could not fetch expiries. Using default.")

calls, puts = get_option_chain(symbol, str(expiry_date))
strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
if not strikes:
    st.error("No option chain data available for selected symbol/expiry")
    st.stop()
spot_price = fetch_underlying_price(tv, symbol, "NASDAQ")
if spot_price == 0:
    try:
        ticker = yf.Ticker(symbol)
        spot_price = ticker.fast_info.last_price
    except Exception:
        spot_price = 0.0
atm_index = min(range(len(strikes)), key=lambda i: abs(strikes[i] - spot_price))
strike = st.sidebar.selectbox("Strike Price", strikes, index=atm_index)
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
try:
    expiry_dt = pd.to_datetime(str(expiry_date)).to_pydatetime()
except Exception:
    st.error("Could not parse expiry date.")
    st.stop()
time_to_exp = (expiry_dt - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
if time_to_exp <= 0:
    st.error("Selected expiry is today or in the past. Please choose a future expiry.")
    st.stop()

st.header("Information")
col1, col2, col3 = st.columns(3)
market_price = 0.0
chain = calls if option_type == "Call" else puts
if not chain.empty:
    try:
        sel = chain[chain["strike"] == strike].iloc[0]
        market_price = sel.get("lastPrice", np.nan)
        if pd.isna(market_price):
            bid = sel.get("bid", np.nan)
            ask = sel.get("ask", np.nan)
            if not pd.isna(bid) or not pd.isna(ask):
                market_price = np.nanmean([bid, ask])
            else:
                market_price = 0.0
    except Exception:
        market_price = 0.0

with col1:
    st.metric("Underlying Price", f"${spot_price:.2f}")
    st.metric("Market Price", f"${market_price:.2f}")

model_price = 0.0
implied_vol = 0.0
if option_style == "European":
    pricer = EuropeanPricer(risk_free_rate=risk_free_rate)
    try:
        if market_price > 0 and spot_price > 0:
            implied_vol = pricer.implied_vol(
                market_price,
                spot_price,
                strike,
                time_to_exp,
                option_type.lower()
            )
            model_price = pricer.price(
                spot_price,
                strike,
                implied_vol,
                time_to_exp,
                option_type.lower()
            )
    except Exception:
        pass
else:
    try:
        def price_diff(sigma):
            return binomial_american_option_price(
                spot_price, strike, risk_free_rate, sigma,
                time_to_exp, option_type=option_type.lower()
            ) - market_price
        try:
            implied_vol = brentq(price_diff, 0.01, 5.0, maxiter=100)
            model_price = binomial_american_option_price(
                spot_price, strike, risk_free_rate, implied_vol,
                time_to_exp, option_type=option_type.lower()
            )
        except Exception:
            for vol in np.linspace(0.1, 1.0, 50):
                price = binomial_american_option_price(
                    spot_price, strike, risk_free_rate, vol,
                    time_to_exp, option_type=option_type.lower()
                )
                if abs(price - market_price) < 0.1:
                    implied_vol = vol
                    model_price = price
                    break
    except Exception:
        pass

with col2:
    st.metric("Model Price", f"${model_price:.2f}",
              delta=f"{(model_price - market_price):.2f}" if model_price else None)
    st.metric("Implied Volatility", f"{implied_vol:.2%}" if implied_vol else "N/A")

with col3:
    st.metric("Time to Expiry", f"{time_to_exp * 365:.1f} days")
    st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

st.header("Implied Volatility Analytics")
iv_smile_fig = None
if not chain.empty and len(chain) > 5:
    atm_range = [s for s in strikes if 0.8 * spot_price <= s <= 1.2 * spot_price]
    ivs = []
    valid_strikes = []
    pricer = EuropeanPricer(risk_free_rate=risk_free_rate)
    for s in atm_range:
        try:
            opt = chain[chain["strike"] == s].iloc[0]
            mp = opt["lastPrice"]
            if pd.isna(mp) or mp <= 0:
                bid = opt.get("bid", np.nan)
                ask = opt.get("ask", np.nan)
                if not pd.isna(bid) and not pd.isna(ask):
                    mp = (bid + ask) / 2
                else:
                    continue
            if mp > 0:
                iv = pricer.implied_vol(
                    mp,
                    spot_price,
                    s,
                    time_to_exp,
                    option_type.lower()
                )
                if iv > 0.01 and iv < 5.0:
                    ivs.append(iv)
                    valid_strikes.append(s)
        except:
            continue
    if len(valid_strikes) > 3:
        iv_smile_fig = plot_iv_smile(valid_strikes, ivs, strike)
if iv_smile_fig:
    st.plotly_chart(iv_smile_fig, use_container_width=True)
else:
    st.warning("Could not generate IV smile. Insufficient data.")

st.header("Pricing Analytics")
analytics_type = st.selectbox("Analytics Type",
                              ["Strike vs IV", "Strike vs Price"])
if not chain.empty:
    df = chain.copy()
    df["strike"] = df["strike"].astype(float)
    df["lastPrice"] = df["lastPrice"].astype(float)
    if "iv" not in df.columns:
        df["iv"] = np.nan
        pricer = EuropeanPricer(risk_free_rate=risk_free_rate)
        for i, row in df.iterrows():
            try:
                mp = row["lastPrice"]
                if pd.isna(mp) or mp <= 0:
                    bid = row.get("bid", np.nan)
                    ask = row.get("ask", np.nan)
                    if not pd.isna(bid) and not pd.isna(ask):
                        mp = (bid + ask) / 2
                    else:
                        continue
                if mp > 0:
                    iv = pricer.implied_vol(
                        mp,
                        spot_price,
                        row["strike"],
                        time_to_exp,
                        option_type.lower()
                    )
                    df.at[i, "iv"] = iv
            except:
                pass
    df = df[(df["strike"] > spot_price * 0.7) & (df["strike"] < spot_price * 1.3)]
    if analytics_type == "Strike vs IV":
        fig = plot_fit_curve(
            df["strike"].tolist(),
            df["iv"].tolist(),
            "Strike Price",
            "Implied Volatility"
        )
    else:
        fig = plot_fit_curve(
            df["strike"].tolist(),
            df["lastPrice"].tolist(),
            "Strike Price",
            "Market Price"
        )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data to generate analytics plot")
else:
    st.warning("No option chain data available for analytics")
