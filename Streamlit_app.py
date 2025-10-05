# ===============================
# Live Stock & Crypto Dashboard
# With ML Forecast + Portfolio Simulation (Enhanced Version + Theme + CSV Download)
# ===============================

import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from prophet import Prophet

# -------------------------------
# Database Connection
# -------------------------------
engine = create_engine('postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db')

st.set_page_config(page_title="Live Stock & Crypto Dashboard", layout="wide")
st.title("📈 Live Stock & Crypto Dashboard with Forecast & Portfolio Simulation")
st.sidebar.header("Select Options")

# -------------------------------
# Sidebar Theme Selector
# -------------------------------
theme_choice = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"], index=0)

if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stMetric {
            background: #1e2228;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 2px 5px rgba(255,255,255,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f7fa;
        }
        .stMetric {
            background: #ffffff;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# User Inputs
# -------------------------------
asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])

stocks_list = ["AAPL", "AMZN", "GOOGLE", "META", "MSFT", "NFLX"]

if asset_type == "Stock":
    selected_stocks = st.sidebar.multiselect("Select Stocks", stocks_list, default=["AAPL"])
else:
    crypto_list = ["BTC"]  # column is 'symbol' in crypto table
    selected_crypto = st.sidebar.selectbox("Select Crypto Coin", crypto_list, index=0)

forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=7)

invest_amount = st.sidebar.number_input("Portfolio Investment Amount per Asset ($)", min_value=100, value=1000)

# -------------------------------
# Fetch Data from PostgreSQL
# -------------------------------
def get_stock_data(symbol):
    query = f"SELECT * FROM cleaned_stock_data WHERE symbol='{symbol}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    return df

def get_crypto_data(symbol):
    query = f"SELECT * FROM cleaned_crypto_data WHERE symbol='{symbol}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)
    return df

# -------------------------------
# Forecast Function
# -------------------------------
def forecast_df(df, days=7):
    df_prophet = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
    m = Prophet(daily_seasonality=True)
    with st.spinner("🔮 Running forecast model..."):
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# -------------------------------
# Display Data & Charts
# -------------------------------
if asset_type == "Stock":
    st.subheader("📊 Selected Stocks Data + Forecast")
    combined_df = pd.DataFrame()
    portfolio_df = pd.DataFrame()

    for symbol in selected_stocks:
        df = get_stock_data(symbol)
        df['symbol'] = symbol
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Forecast
        fc = forecast_df(df, forecast_days)
        fc['symbol'] = symbol

        # Portfolio Simulation
        start_price = df['close'].iloc[-1]
        fc['investment_value'] = invest_amount * (fc['yhat'] / start_price)
        portfolio_df = pd.concat([portfolio_df, fc], ignore_index=True)

        # KPIs for each stock
        current_price = start_price
        forecasted_price = fc['yhat'].iloc[-1]
        return_pct = ((forecasted_price - current_price) / current_price) * 100
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{symbol} Current Price", f"${current_price:,.2f}")
        col2.metric(f"{forecast_days}-Day Forecast", f"${forecasted_price:,.2f}")
        col3.metric("Expected Return", f"{return_pct:.2f}%")

    st.dataframe(combined_df)

    # Plot Historical + Forecast
    fig = px.line(combined_df, x='date', y='close', color='symbol', title="Stock Prices (Historical + Forecast)")
    for symbol in selected_stocks:
        fc = portfolio_df[portfolio_df['symbol'] == symbol]
        fig.add_scatter(x=fc['ds'], y=fc['yhat'], mode='lines', name=f"{symbol} Forecast", line=dict(dash='dot'))
        fig.add_scatter(x=fc['ds'], y=fc['yhat_lower'], mode='lines', name=f"{symbol} Lower Bound", line=dict(dash='dot'))
        fig.add_scatter(x=fc['ds'], y=fc['yhat_upper'], mode='lines', name=f"{symbol} Upper Bound", line=dict(dash='dot'))
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Simulation Plot
    st.subheader("💰 Portfolio Simulation Value Over Forecast Period")
    fig_port = px.line(portfolio_df, x='ds', y='investment_value', color='symbol', title="Simulated Portfolio Value")
    st.plotly_chart(fig_port, use_container_width=True)

    # -------------------------------
    # CSV Downloads
    # -------------------------------
    st.subheader("📥 Download Data")
    st.download_button("⬇ Download Historical Data (All Stocks)", combined_df.to_csv(index=False).encode("utf-8"), "stocks_historical.csv", "text/csv")
    st.download_button("⬇ Download Forecast Data (All Stocks)", portfolio_df.to_csv(index=False).encode("utf-8"), "stocks_forecast.csv", "text/csv")

else:
    st.subheader(f"📊 {selected_crypto} Crypto Data + Forecast")
    df = get_crypto_data(selected_crypto)
    st.dataframe(df)

    # Forecast
    fc = forecast_df(df, forecast_days)
    fc['symbol'] = selected_crypto

    # Portfolio Simulation
    start_price = df['close'].iloc[-1]
    fc['investment_value'] = invest_amount * (fc['yhat'] / start_price)

    # KPIs
    current_price = start_price
    forecasted_price = fc['yhat'].iloc[-1]
    return_pct = ((forecasted_price - current_price) / current_price) * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:,.2f}")
    col2.metric(f"{forecast_days}-Day Forecast", f"${forecasted_price:,.2f}")
    col3.metric("Expected Return", f"{return_pct:.2f}%")

    st.success(f"💰 If you invest ${invest_amount}, estimated value after {forecast_days} days: **${fc['investment_value'].iloc[-1]:,.2f}**")

    # Historical + Forecast Plot
    fig = px.line(df, x='date', y='close', title=f"{selected_crypto} Price (Historical + Forecast)")
    fig.add_scatter(x=fc['ds'], y=fc['yhat'], mode='lines', name=f"{selected_crypto} Forecast", line=dict(dash='dot'))
    fig.add_scatter(x=fc['ds'], y=fc['yhat_lower'], mode='lines', name="Lower Bound", line=dict(dash='dot'))
    fig.add_scatter(x=fc['ds'], y=fc['yhat_upper'], mode='lines', name="Upper Bound", line=dict(dash='dot'))
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio Simulation Plot
    st.subheader("💰 Portfolio Simulation Value Over Forecast Period")
    fig_port = px.line(fc, x='ds', y='investment_value', title=f"{selected_crypto} Portfolio Value")
    st.plotly_chart(fig_port, use_container_width=True)

    # -------------------------------
    # CSV Downloads
    # -------------------------------
    st.subheader("📥 Download Data")
    st.download_button("⬇ Download Historical Data (Crypto)", df.to_csv(index=False).encode("utf-8"), f"{selected_crypto}_historical.csv", "text/csv")
    st.download_button("⬇ Download Forecast Data (Crypto)", fc.to_csv(index=False).encode("utf-8"), f"{selected_crypto}_forecast.csv", "text/csv")

# -------------------------------
# Optional Raw Data View
# -------------------------------
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data View")
    if asset_type == "Stock":
        raw_df_list = []
        for symbol in selected_stocks:
            raw_df = pd.read_sql(f"SELECT * FROM cleaned_stock_data WHERE symbol='{symbol}'", engine)
            raw_df_list.append(raw_df)
        st.dataframe(pd.concat(raw_df_list, ignore_index=True))
    else:
        raw_df = pd.read_sql(f"SELECT * FROM cleaned_crypto_data WHERE symbol='{selected_crypto}'", engine)
        st.dataframe(raw_df)