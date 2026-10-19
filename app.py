from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Stock Market Analytics", page_icon="📈", layout="wide")
st.title("📈 Stock Market Data Analysis")
st.caption("Historical market analytics and an educational next-day closing-price baseline. Not investment advice.")

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    # yfinance end date is exclusive; add one day to include the selected end date.
    raw = yf.download(
        ticker, start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False, progress=False, threads=False
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    # Newer yfinance versions can return MultiIndex columns even for one ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in raw.columns for c in needed):
        return pd.DataFrame()
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"
    raw = raw.dropna(subset=["Close"]).sort_index()
    return raw

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["Daily_Return_%"] = x["Close"].pct_change() * 100
    x["MA_20"] = x["Close"].rolling(20).mean()
    x["MA_50"] = x["Close"].rolling(50).mean()
    x["Volatility_20D_%"] = x["Daily_Return_%"].rolling(20).std() * np.sqrt(252)
    x["Volume_MA_20"] = x["Volume"].rolling(20).mean()
    x["Rolling_Max"] = x["Close"].cummax()
    x["Drawdown_%"] = (x["Close"] / x["Rolling_Max"] - 1) * 100
    return x

def model_evaluation(df: pd.DataFrame):
    # Features at day t predict Close at day t+1. Shift target to prevent look-ahead.
    d = df.copy()
    d["Target_Next_Close"] = d["Close"].shift(-1)
    d["Target_Date"] = d.index.to_series().shift(-1)
    d["Close_Lag_1"] = d["Close"]
    d["Close_Lag_2"] = d["Close"].shift(1)
    d["Close_Lag_5"] = d["Close"].shift(4)
    d["Return_Lag_1"] = d["Close"].pct_change()
    d["MA_5"] = d["Close"].rolling(5).mean()
    d["MA_20_model"] = d["Close"].rolling(20).mean()
    d["Volume_Change"] = d["Volume"].pct_change()
    features = ["Close_Lag_1", "Close_Lag_2", "Close_Lag_5",
                "Return_Lag_1", "MA_5", "MA_20_model", "Volume_Change"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["Target_Next_Close"])
    if len(d) < 80:
        return None
    split = int(len(d) * 0.8)
    train, test = d.iloc[:split], d.iloc[split:]
    X_train, y_train = train[features], train["Target_Next_Close"]
    X_test, y_test = test[features], test["Target_Next_Close"]

    # Simple baseline: tomorrow's close equals today's close.
    baseline = DummyRegressor(strategy="constant", constant=0.0)
    # Use today's close as baseline predictions directly (not a fitted constant).
    baseline_pred = X_test["Close_Lag_1"].to_numpy()

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=3,
                              random_state=42, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results = pd.DataFrame({
        "Actual_Next_Close": y_test.to_numpy(),
        "RandomForest_Prediction": pred,
        "Naive_Baseline_Today_Close": baseline_pred,
    }, index=test["Target_Date"].to_numpy())
    results.index = pd.to_datetime(results.index)
    results.index.name = "Target_Date"

    def metrics(y, p):
        return {
            "MAE": mean_absolute_error(y, p),
            "RMSE": np.sqrt(mean_squared_error(y, p)),
            "R²": r2_score(y, p),
        }
    metric_table = pd.DataFrame({
        "Random Forest": metrics(y_test, pred),
        "Naive baseline": metrics(y_test, baseline_pred),
    }).T
    # Estimate the next trading day's close from the most recent available row.
    latest = d.iloc[[-1]][features]
    next_pred = float(model.predict(latest)[0])
    latest_close = float(d["Close"].iloc[-1])
    return results, metric_table, next_pred, latest_close

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    today = date.today()
    start_default = today - timedelta(days=365*3)
    start = st.date_input("Start date", value=start_default, max_value=today)
    end = st.date_input("End date", value=today, max_value=today)
    show_ma20 = st.checkbox("Show 20-day moving average", True)
    show_ma50 = st.checkbox("Show 50-day moving average", True)
    st.caption("Examples: AAPL, MSFT, RELIANCE.NS, TCS.NS, ^NSEI")

if not ticker:
    st.warning("Enter a ticker symbol.")
    st.stop()
if start >= end:
    st.error("Start date must be earlier than end date.")
    st.stop()

with st.spinner(f"Downloading {ticker} historical data..."):
    raw = load_prices(ticker, start, end)

if raw.empty:
    st.error("No data was returned. Check the ticker symbol, date range, internet connection, and Yahoo Finance availability.")
    st.stop()

df = add_features(raw)
latest_close = float(df["Close"].iloc[-1])
first_close = float(df["Close"].iloc[0])
change_pct = (latest_close / first_close - 1) * 100
latest_return = df["Daily_Return_%"].iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{latest_close:,.2f}")
c2.metric("Period return", f"{change_pct:.2f}%")
c3.metric("Latest daily return", "—" if pd.isna(latest_return) else f"{latest_return:.2f}%")
c4.metric("Trading sessions", f"{len(df):,}")

tab_overview, tab_returns, tab_model, tab_data = st.tabs(
    ["Price overview", "Returns & risk", "Model evaluation", "Data & download"]
)

with tab_overview:
    st.subheader(f"{ticker}: historical closing price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", mode="lines"))
    if show_ma20:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_20"], name="20-day MA", mode="lines"))
    if show_ma50:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA_50"], name="50-day MA", mode="lines"))
    fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    vol_fig = go.Figure(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    vol_fig.update_layout(height=280, xaxis_title="Date", yaxis_title="Volume")
    st.plotly_chart(vol_fig, use_container_width=True)

with tab_returns:
    left, right = st.columns(2)
    with left:
        st.subheader("Daily returns")
        returns_fig = go.Figure(go.Histogram(x=df["Daily_Return_%"].dropna(), nbinsx=50))
        returns_fig.update_layout(height=360, xaxis_title="Daily return (%)", yaxis_title="Count")
        st.plotly_chart(returns_fig, use_container_width=True)
    with right:
        st.subheader("20-day annualized volatility")
        vol_fig2 = go.Figure(go.Scatter(x=df.index, y=df["Volatility_20D_%"], mode="lines", name="Volatility"))
        vol_fig2.update_layout(height=360, xaxis_title="Date", yaxis_title="Annualized volatility (%)")
        st.plotly_chart(vol_fig2, use_container_width=True)
    st.subheader("Drawdown from running peak")
    dd_fig = go.Figure(go.Scatter(x=df.index, y=df["Drawdown_%"], fill="tozeroy", mode="lines", name="Drawdown"))
    dd_fig.update_layout(height=320, xaxis_title="Date", yaxis_title="Drawdown (%)")
    st.plotly_chart(dd_fig, use_container_width=True)
    st.caption("Volatility is a historical measure; it does not predict future risk with certainty.")

with tab_model:
    st.subheader("Next-trading-day close: baseline comparison")
    st.write("Features use information available on each historical date. The target is the next observed trading session's close. The split is chronological (first 80% train, last 20% test).")
    with st.spinner("Training model and evaluating the holdout period..."):
        evaluated = model_evaluation(df)
    if evaluated is None:
        st.warning("Not enough data to evaluate the model. Choose a longer date range (at least about 6 months of trading data).")
    else:
        results, metric_table, next_pred, model_last_close = evaluated
        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest MAE", f"{metric_table.loc['Random Forest','MAE']:.3f}")
        m2.metric("Naive baseline MAE", f"{metric_table.loc['Naive baseline','MAE']:.3f}")
        m3.metric("Next-session estimate", f"{next_pred:,.2f}", delta=f"{(next_pred/model_last_close-1)*100:.2f}% vs latest close")
        st.dataframe(metric_table.style.format("{:.4f}"), use_container_width=True)
        st.caption("MAE and RMSE are in price units. R² can be negative. A model that fails to beat the naive baseline is not useful for forecasting in this test.")
        model_fig = go.Figure()
        model_fig.add_trace(go.Scatter(x=results.index, y=results["Actual_Next_Close"], name="Actual"))
        model_fig.add_trace(go.Scatter(x=results.index, y=results["RandomForest_Prediction"], name="Random Forest"))
        model_fig.add_trace(go.Scatter(x=results.index, y=results["Naive_Baseline_Today_Close"], name="Naive baseline"))
        model_fig.update_layout(height=430, xaxis_title="Target date", yaxis_title="Close", hovermode="x unified")
        st.plotly_chart(model_fig, use_container_width=True)
        st.download_button("Download model test predictions", results.to_csv().encode("utf-8"),
                           file_name=f"{ticker.lower().replace('^','')}_model_predictions.csv", mime="text/csv")

with tab_data:
    st.subheader("Historical data with indicators")
    st.dataframe(df.tail(100), use_container_width=True)
    st.download_button("Download full analyzed data (CSV)", df.to_csv().encode("utf-8"),
                       file_name=f"{ticker.lower().replace('^','')}_analyzed_data.csv", mime="text/csv")
    st.caption("Yahoo Finance data may be delayed, adjusted, revised, incomplete, or unavailable for some tickers.")
