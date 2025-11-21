import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, time

# PostgreSQL Connection
engine=create_engine('postgresql://postgres.szcndefhegrwzhuzgozm:Nishant%407879692581@aws-1-ap-south-1.pooler.supabase.com:5432/postgres')

# -----------------------------
# Function: Fetch Crypto Data
# -----------------------------
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


# -----------------------------------------
# Crypto List (You can add more here)
# -----------------------------------------
crypto_list = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE",
    "DOT", "TRX", "AVAX", "MATIC", "LTC", "BCH", "XLM", "LINK"
]


# -----------------------------------------
# Loop through each crypto and update DB
# -----------------------------------------
for symbol in crypto_list:
    print(f"\n⏳ Fetching new data for {symbol}...")

    # Step 1: Check last saved date
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT MAX(date) FROM crypto WHERE symbol='{symbol}'"))
        last_date = result.scalar()

    # Step 2: Fetch only new data
    if last_date:
        last_datetime = datetime.combine(last_date, time.min)
        last_timestamp = int(last_datetime.timestamp())

        df = fetch_crypto(symbol, to_ts=int(datetime.now().timestamp()))
        df = df[df["date"] > last_datetime]
    else:
        df = fetch_crypto(symbol)

    # Step 3: Insert new rows
    if not df.empty:
        df.to_sql("crypto", engine, if_exists="append", index=False)
        print(f"✅ Inserted {len(df)} new rows for {symbol}.")
    else:
        print(f"✔ {symbol} is already up to date.")
