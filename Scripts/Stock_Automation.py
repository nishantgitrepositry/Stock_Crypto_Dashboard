# stock_updater.py
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# ---------- CONFIG ----------
ENGINE_STR = "postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db"
engine = create_engine(ENGINE_STR)

# ---------- helpers ----------
def _normalize_yf_df(df, symbol):
    if df.empty:
        return pd.DataFrame(columns=["symbol","date","open","high","low","close","volume"])
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # flatten
    df.columns = [str(c).lower() for c in df.columns]
    rename_map = {
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "volume": "volume"
    }
    df = df.rename(columns=rename_map)
    out = df[["date","open","high","low","close","volume"]].copy()
    out["symbol"] = symbol
    out["date"] = pd.to_datetime(out["date"]).dt.date
    return out

def fetch_stock(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    return _normalize_yf_df(df, symbol)

# ---------- main update function ----------
def update_stock(symbol):
    with engine.connect() as conn:
        last_date = conn.execute(
            text("SELECT MAX(date) FROM stocks WHERE symbol = :sym"),
            {"sym": symbol}
        ).scalar()

    if last_date is not None:
        last_date = pd.to_datetime(last_date).date()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[{symbol}] fetching from {start_date} to {end_date} ...")
    df = fetch_stock(symbol, start=start_date, end=end_date)

    if df.empty:
        print(f"[{symbol}] no rows fetched.")
        return

    # ✅ Insert with ON CONFLICT
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

    print(f"[{symbol}] upserted {len(df)} rows.")

# ---------- run for a list ----------
if __name__ == "__main__":
    stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"]
    for s in stock_list:
        try:
            update_stock(s)
        except Exception as e:
            print(f"[{s}] error:", e)
 