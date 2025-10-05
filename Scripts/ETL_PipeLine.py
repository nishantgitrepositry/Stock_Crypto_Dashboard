import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, time
import sqlalchemy
from datetime import datetime
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# ---------------------------
# 1. Database Connection
# ---------------------------
engine = sqlalchemy.create_engine(
    "postgresql+psycopg2://postgres:Nishant%40123@localhost:5432/stock_crypto_db"
)

# ---------------------------
# Step 1: Fetch raw data (dummy placeholder)
# ---------------------------

# 1. Crypto Data Fetch
# ---------------------------

def fetch_crypto(symbol="BTC", currency="USD", limit=2000, to_ts=None):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol, "tsym": currency, "limit": limit}
    if to_ts:
        params["toTs"] = to_ts
    r = requests.get(url, params=params)
    data = r.json()["Data"]["Data"]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df["symbol"] = symbol

    df.rename(columns={
        "volumefrom": "volume_crypto",
        "volumeto": "volume_usd"
    }, inplace=True)

    return df[["symbol", "date", "open", "high", "low", "close", "volume_crypto", "volume_usd"]]

def update_crypto(symbol="BTC"):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date) FROM crypto WHERE symbol=:sym"), {"sym": symbol})
        last_date = result.scalar()

    if last_date:
        last_datetime = datetime.combine(last_date, time.min)
        crypto_df = fetch_crypto(symbol, to_ts=int(datetime.now().timestamp()))
        crypto_df = crypto_df[crypto_df["date"] > last_datetime]
    else:
        crypto_df = fetch_crypto(symbol)

    if not crypto_df.empty:
        crypto_df.to_sql("crypto", engine, if_exists="append", index=False)
        print(f"✅ [Crypto-{symbol}] Inserted {len(crypto_df)} new rows.")
    else:
        print(f"⚠️ [Crypto-{symbol}] No new data.")

# ---------------------------
# 2. Stock Data Fetch
# ---------------------------
def _normalize_yf_df(df, symbol):
    if df.empty:
        return pd.DataFrame(columns=["symbol","date","open","high","low","close","volume"])
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower() for c in df.columns]
    rename_map = {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "adj close": "adj_close", "volume": "volume"}
    df = df.rename(columns=rename_map)
    out = df[["date","open","high","low","close","volume"]].copy()
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def fetch_stock(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    return _normalize_yf_df(df, symbol)

def update_stock(symbol):
    with engine.connect() as conn:
        last_date = conn.execute(text("SELECT MAX(date) FROM stocks WHERE symbol = :sym"), {"sym": symbol}).scalar()

    if last_date is not None:
        last_date = pd.to_datetime(last_date).date()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[{symbol}] fetching from {start_date} to {end_date} ...")
    df = fetch_stock(symbol, start=start_date, end=end_date)

    if df.empty:
        print(f"⚠️ [Stock-{symbol}] No rows fetched.")
        return

    insert_query = """
    INSERT INTO stocks (symbol, date, open, high, low, close, volume)
    VALUES (:symbol, :date, :open, :high, :low, :close, :volume)
    ON CONFLICT (symbol, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume;
    """

    rows = df.to_dict(orient="records")
    with engine.begin() as conn:
        conn.execute(text(insert_query), rows)

    print(f"✅ [Stock-{symbol}] Upserted {len(df)} rows.")

# ---------------------------
# 3. Combined Raw Fetch Function
# ---------------------------
def fetch_raw_data():
    print("📥 Fetching raw data for Stocks & Crypto...")

    # Stocks
    stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
    for s in stock_list:
        try:
            update_stock(s)
        except Exception as e:
            print(f"❌ [Stock-{s}] error: {e}")

    # Crypto
    crypto_list = ["BTC"]
    for c in crypto_list:
        try:
            update_crypto(c)
        except Exception as e:
            print(f"❌ [Crypto-{c}] error: {e}")

    print("✅ Raw data fetched & stored in 'stocks' and 'crypto' tables")





#  Steps-2 -- Clean Raw Data

def clean_data():
   
    # -----------------------------
    # 2. CLEANING CRYPTO
    # -----------------------------
    # Fetch Raw Data
    query = "SELECT * FROM crypto;"
    df = pd.read_sql(query, engine)

    print("Before Cleaning (Crypto):")
    print(df.info())
    print(df.head())

    # Handle Missing Values
    df = df.sort_values(by="date")
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    missing_after = df.isnull().sum().sum()
    missing_filled = missing_before - missing_after

    # Handle Outliers
    z_scores = np.abs(stats.zscore(df['close']))
    outliers_removed = (z_scores >= 3).sum()
    df = df[(z_scores < 3)]

    # Insert Clean Data
    with engine.connect() as conn:
        table_check = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='cleaned_crypto_data');")
        )
        exists = table_check.scalar()

        last_date = None
        if exists:
            result = conn.execute(text("SELECT MAX(date) FROM cleaned_crypto_data;"))
            last_date = result.scalar()

    if last_date:
        df_new = df[df['date'] > last_date]
    else:
        df_new = df.copy()

    df_new.to_sql("cleaned_crypto_data", engine, if_exists="append", index=False)

    # Save Logs
    log_data = pd.DataFrame([{
        "run_time": datetime.now(),
        "records_inserted": len(df_new),
        "missing_filled": int(missing_filled),
        "outliers_removed": int(outliers_removed)
    }])
    log_data.to_sql("logs_cleaning", engine, if_exists="append", index=False)

    print("After Cleaning (Crypto):")
    print(df.info())
    print(f"Inserted {len(df_new)} new records into cleaned_crypto_data")
    print(f"Missing values filled: {missing_filled}, Outliers removed: {outliers_removed}")

    # -----------------------------
    # 3. CLEANING STOCKS
    # -----------------------------
    # Fetch Raw Data
    query = "SELECT * FROM stocks;"
    df = pd.read_sql(query, engine)

    print("Before Cleaning (Stock):")
    print(df.info())
    print(df.head())

    # Handle Missing Values
    df = df.sort_values(by="date")
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    missing_after = df.isnull().sum().sum()
    missing_filled = missing_before - missing_after

    # Handle Outliers
    z_scores = np.abs(stats.zscore(df['close']))
    outliers_removed = (z_scores >= 3).sum()
    df = df[(z_scores < 3)]

    # Insert Clean Data
    with engine.connect() as conn:
        table_check = conn.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='cleaned_stock_data');")
        )
        exists = table_check.scalar()

        last_date = None
        if exists:
            result = conn.execute(text("SELECT MAX(date) FROM cleaned_stock_data;"))
            last_date = result.scalar()

    if last_date:
        df_new = df[df['date'] > last_date]
    else:
        df_new = df.copy()

    df_new.to_sql("cleaned_stock_data", engine, if_exists="append", index=False)

    # Save Logs
    log_data = pd.DataFrame([{
        "run_time": datetime.now(),
        "records_inserted": len(df_new),
        "missing_filled": int(missing_filled),
        "outliers_removed": int(outliers_removed)
    }])
    log_data.to_sql("logs_cleaning", engine, if_exists="append", index=False)

    print("After Cleaning (Stock):")
    print(df.info())
    print(f"Inserted {len(df_new)} new records into cleaned_stock_data")
    print(f"Missing values filled: {missing_filled}, Outliers removed: {outliers_removed}")







# Step-3  Date-time format conversion, time-series alignment



def align_and_merge():
    print("\n🔄 Aligning and merging Stocks + Crypto data...")

    # 1. Load Stock Data
    stock_df = pd.read_sql("SELECT date, symbol, close FROM cleaned_stock_data ORDER BY date", engine)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_pivot = stock_df.pivot(index="date", columns="symbol", values="close")

    # 2. Load Crypto Data
    crypto_df = pd.read_sql("SELECT date, symbol, close FROM cleaned_crypto_data ORDER BY date", engine)
    crypto_df['date'] = pd.to_datetime(crypto_df['date'])
    crypto_pivot = crypto_df.pivot(index="date", columns="symbol", values="close")

    # 3. Merge (outer join)
    merged_df = stock_pivot.join(crypto_pivot, how="outer").sort_index()

    # 4. Fill missing values
    merged_df = merged_df.ffill().bfill()

    # 5. Debug Info
    print("✅ Merged DF Shape:", merged_df.shape)
    print(merged_df.head())

    # 6. Reset Index so 'date' bhi column ban jaye
    merged_df = merged_df.reset_index()

    # 7. Save Final Merged Data
    merged_df.to_sql("merged_stocks_crypto", engine, if_exists="replace", index=False)

    print("✅ All Stocks + Crypto data aligned and saved in table: merged_stocks_crypto")




# Step 4: Calculate Metrics


def calculate_metrics():
    print("\n📊 Calculating metrics for Stocks & Crypto...")

    # ---------------------------
    # 1. Helper Function
    # ---------------------------
    def _calc(df, price_col):
        df = df.sort_values("date")

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
    # 2. Process Stock Prices
    # ---------------------------
    stock_query = "SELECT date, symbol, close FROM cleaned_stock_data;"
    stock_df = pd.read_sql(stock_query, engine)

    if not stock_df.empty:
        stock_metrics = (
            stock_df.groupby("symbol", group_keys=False)[["date", "symbol", "close"]]
            .apply(lambda x: _calc(x, "close"))
        )

        # Save to stock_metrics table
        stock_metrics.to_sql("stock_metrics", engine, if_exists="replace", index=False)

        print(f"✅ Stock metrics updated! Records: {len(stock_metrics)}")
        print(stock_metrics.head())
    else:
        print("⚠️ No stock data found!")

    # ---------------------------
    # 3. Process Crypto Prices
    # ---------------------------
    crypto_query = "SELECT date, symbol, close FROM cleaned_crypto_data;"
    crypto_df = pd.read_sql(crypto_query, engine)

    if not crypto_df.empty:
        crypto_metrics = (
            crypto_df.groupby("symbol", group_keys=False)[["date", "symbol", "close"]]
            .apply(lambda x: _calc(x, "close"))
        )

        # Save to crypto_metrics table
        crypto_metrics.to_sql("crypto_metrics", engine, if_exists="replace", index=False)

        print(f"✅ Crypto metrics updated! Records: {len(crypto_metrics)}")
        print(crypto_metrics.head())
    else:
        print("⚠️ No crypto data found!")

    print("\n📊 Metrics calculation completed successfully!")





# ---------------------------
# Master Pipeline Runner
# ---------------------------
def run_pipeline():
    print("\n🚀 Starting ETL Pipeline...\n")
    fetch_raw_data()
    clean_data()
    align_and_merge()
    calculate_metrics()
    print("\n✅ ETL Pipeline finished successfully!\n")

# ---------------------------
# Run if executed directly
# ---------------------------
if __name__ == "__main__":
    run_pipeline()
