import requests
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db')

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

# Step 1: Check last saved date in DB

from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("SELECT MAX(date) FROM crypto WHERE symbol='BTC'"))
    last_date = result.scalar()


# Step 2: Fetch only new data

from datetime import datetime, time

if last_date:
    # Convert last_date (date) -> datetime
    last_datetime = datetime.combine(last_date, time.min)
    last_timestamp = int(last_datetime.timestamp())

    # Fetch till current time
    crypto_df = fetch_crypto("BTC", to_ts=int(datetime.now().timestamp()))

    # Only new rows
    crypto_df = crypto_df[crypto_df["date"] > last_datetime]
else:
    # If no data in DB, fetch full history
    crypto_df = fetch_crypto("BTC")


# ✅ Step 3: Save only new rows

if not crypto_df.empty:
    crypto_df.to_sql("crypto", engine, if_exists="append", index=False)
    print(f"Inserted {len(crypto_df)} new rows.")
else:
    print("No new data to insert.")