# ===============================
# Live Stock & Crypto Dashboard
# With ML Forecast + Portfolio Simulation + Comparison Filters
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Database Connection
# -------------------------------
engine = create_engine('postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db')

st.set_page_config(page_title="Live Stock & Crypto Dashboard", layout="wide")
st.title("📈 Live Stock & Crypto Dashboard with Forecast, Portfolio & Comparison")
st.sidebar.header("Select Options")

# -------------------------------
# Sidebar Theme Selector
# -------------------------------
theme_choice = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"], index=0)
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        .stMetric { background: #1e2228; padding: 10px; border-radius: 10px; box-shadow: 0px 2px 5px rgba(255,255,255,0.1); }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        .stApp { background-color: #f5f7fa; }
        .stMetric { background: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); }
        </style>
        """, unsafe_allow_html=True)

# -------------------------------
# Sidebar Inputs
# -------------------------------
asset_type = st.sidebar.selectbox("Select Asset Type", ["Stock", "Crypto"])
stocks_list = ["AAPL", "AMZN", "GOOGLE", "META", "MSFT", "NFLX", "TSLA", "NVDA"]
crypto_list = ["BTC"]

if asset_type == "Stock":
    selected_assets = st.sidebar.multiselect("Select Stocks", stocks_list, default=["AAPL"])
else:
    selected_assets = st.sidebar.multiselect("Select Crypto", crypto_list, default=["BTC"])

forecast_days = st.sidebar.slider("Forecast Days", min_value=7, max_value=30, value=7)
invest_amount = st.sidebar.number_input("Portfolio Investment Amount per Asset ($)", min_value=100, value=1000)

# -------------------------------
# Comparison Filters
# -------------------------------
st.sidebar.header("📊 Comparison Filters")
all_assets = stocks_list + crypto_list
comparison_assets = st.sidebar.multiselect("Select Assets for Comparison", all_assets, default=selected_assets)
timeframes = {"7D":7, "14D":14, "30D":30, "90D":90, "180D":180, "1Y":365, "All":0}
selected_timeframe = st.sidebar.selectbox("Select Timeframe", list(timeframes.keys()), index=3)
indicators = ["Close Price","Daily Return","MA7","MA30","Volatility"]
selected_indicators = st.sidebar.multiselect("Select Indicators", indicators, default=["Close Price"])

# -------------------------------
# Fetch Data Functions
# -------------------------------
def get_stock_data(symbol):
    query = f"SELECT * FROM cleaned_stock_data WHERE symbol='{symbol}' ORDER BY date ASC"
    return pd.read_sql(query, engine)

def get_crypto_data(symbol):
    query = f"SELECT * FROM cleaned_crypto_data WHERE symbol='{symbol}' ORDER BY date ASC"
    return pd.read_sql(query, engine)

def filter_timeframe(df, days):
    if days > 0:
        return df.tail(days)
    return df

# -------------------------------
# Forecast Functions
# -------------------------------
def forecast_prophet(df, days=7):
    df_prophet = df[['date','close']].rename(columns={'date':'ds','close':'y'})
    m = Prophet(daily_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

def forecast_arima(df, days=7):
    ts = df['close']
    model = ARIMA(ts, order=(5,1,0)).fit()
    fc = model.get_forecast(steps=days)
    forecast = pd.DataFrame({
        'ds': pd.date_range(start=df['date'].iloc[-1]+pd.Timedelta(days=1), periods=days),
        'yhat': fc.predicted_mean,
        'yhat_lower': fc.conf_int().iloc[:,0],
        'yhat_upper': fc.conf_int().iloc[:,1]
    })
    return forecast

def forecast_lstm(df, days=7):
    from sklearn.preprocessing import MinMaxScaler
    data = df['close'].values.reshape(-1,1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X = []
    time_step = 60
    for i in range(time_step, len(scaled)):
        X.append(scaled[i-time_step:i,0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1],1)
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, scaled[time_step:], epochs=5, batch_size=32, verbose=0)
    last_seq = scaled[-time_step:]
    preds = []
    for _ in range(days):
        x_input = last_seq.reshape(1,time_step,1)
        pred = model.predict(x_input, verbose=0)
        preds.append(pred[0,0])
        last_seq = np.append(last_seq[1:], pred)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    forecast = pd.DataFrame({
        'ds': pd.date_range(start=df['date'].iloc[-1]+pd.Timedelta(days=1), periods=days),
        'yhat': preds.flatten()
    })
    forecast['yhat_lower'] = forecast['yhat']*0.98
    forecast['yhat_upper'] = forecast['yhat']*1.02
    return forecast

# -------------------------------
# Display & Forecast
# -------------------------------
st.subheader("📊 Selected Assets Data + Forecast")
combined_df = pd.DataFrame()
portfolio_df = pd.DataFrame()

for symbol in selected_assets:
    df = get_crypto_data(symbol) if symbol=="BTC" else get_stock_data(symbol)
    df['symbol'] = symbol
    combined_df = pd.concat([combined_df, df], ignore_index=True)

    fc_prophet = forecast_prophet(df, forecast_days)
    fc_arima = forecast_arima(df, forecast_days)
    fc_lstm = forecast_lstm(df, forecast_days)

    fc = fc_prophet  # default display Prophet
    fc['symbol'] = symbol
    start_price = df['close'].iloc[-1]
    fc['investment_value'] = invest_amount*(fc['yhat']/start_price)
    portfolio_df = pd.concat([portfolio_df, fc], ignore_index=True)

    # KPIs
    col1,col2,col3 = st.columns(3)
    col1.metric(f"{symbol} Current Price", f"${start_price:,.2f}")
    col2.metric(f"{forecast_days}-Day Forecast (Prophet)", f"${fc['yhat'].iloc[-1]:,.2f}")
    col3.metric("Expected Return", f"{((fc['yhat'].iloc[-1]-start_price)/start_price*100):.2f}%")

# Plot
fig = px.line(combined_df, x='date', y='close', color='symbol', title="Historical Prices + Forecast")
for symbol in selected_assets:
    fc = portfolio_df[portfolio_df['symbol']==symbol]
    fig.add_scatter(x=fc['ds'], y=fc['yhat'], mode='lines', name=f"{symbol} Forecast", line=dict(dash='dot'))
st.plotly_chart(fig, use_container_width=True)

# Portfolio Simulation
st.subheader("💰 Portfolio Simulation")
fig_port = px.line(portfolio_df, x='ds', y='investment_value', color='symbol', title="Portfolio Value Over Forecast")
st.plotly_chart(fig_port, use_container_width=True)

# -------------------------------
# Comparison Charts
# -------------------------------
st.subheader("📈 Comparison Charts")
comparison_df = pd.DataFrame()
for asset in comparison_assets:
    df = get_crypto_data(asset) if asset=="BTC" else get_stock_data(asset)
    df = filter_timeframe(df, timeframes[selected_timeframe])
    if "Daily Return" in selected_indicators:
        df['Daily Return'] = df['close'].pct_change()
    if "MA7" in selected_indicators:
        df['MA7'] = df['close'].rolling(7).mean()
    if "MA30" in selected_indicators:
        df['MA30'] = df['close'].rolling(30).mean()
    if "Volatility" in selected_indicators:
        df['Volatility'] = df['close'].pct_change().rolling(7).std()
    df['Asset'] = asset
    comparison_df = pd.concat([comparison_df, df], ignore_index=True)

for indicator in selected_indicators:
    if indicator in comparison_df.columns:
        fig = px.line(comparison_df, x='date', y=indicator, color='Asset', title=f"{indicator} Comparison")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# CSV Download
# -------------------------------
st.subheader("📥 Download Data")
st.download_button("⬇ Download Historical Data", combined_df.to_csv(index=False).encode("utf-8"), "historical.csv", "text/csv")
st.download_button("⬇ Download Forecast", portfolio_df.to_csv(index=False).encode(), "forecast.csv")
