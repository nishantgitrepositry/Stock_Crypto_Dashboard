import pandas as pd
import sqlalchemy
import psycopg2
from sqlalchemy import create_engine

# ---------------------------
# 1. Database Connection
# ---------------------------
engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://postgres:Nishant%40123@localhost:5432/stock_crypto_db"
)

# ---------------------------
# 2. Helper Function for Metrics
# ---------------------------
def calculate_metrics(df, id_col, price_col):
    df = df.sort_values(["date"])
    
    # Daily Returns %
    df["daily_ret_pct"] = df[price_col].pct_change() * 100
    
    # Moving Averages
    df["ma_7"] = df[price_col].rolling(window=7).mean()
    df["ma_30"] = df[price_col].rolling(window=30).mean()
    df["ma_90"] = df[price_col].rolling(window=90).mean()
    
    # Volatility (30-day rolling, annualized %)
    df["vol_30_ann_pct"] = df["daily_ret_pct"].rolling(window=30).std() * (252**0.5)
    
    return df

# ---------------------------
# 3. Process Stock Prices
# ---------------------------
stock_query = "SELECT date, symbol, close FROM stocks;"
stock_df = pd.read_sql(stock_query, engine)

if not stock_df.empty:
    stock_metrics = (
        stock_df.groupby("symbol", group_keys=False)[["date", "symbol", "close"]]
        .apply(lambda x: calculate_metrics(x, "symbol", "close"))
    )
    
    # Save to stock_metrics table
    stock_metrics.to_sql("stock_metrics", engine, if_exists="replace", index=False)
    print("✅ Stock metrics updated!")

# ---------------------------
# 4. Process Crypto Prices
# ---------------------------
crypto_query = "SELECT date, symbol, close FROM crypto;"
crypto_df = pd.read_sql(crypto_query, engine)

if not crypto_df.empty:
    crypto_metrics = (
        crypto_df.groupby("symbol", group_keys=False)[["date", "symbol", "close"]]
        .apply(lambda x: calculate_metrics(x, "symbol", "close"))
    )
    
    # Save to crypto_metrics table
    crypto_metrics.to_sql("crypto_metrics", engine, if_exists="replace", index=False)
    print("✅ Crypto metrics updated!")
