import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
from datetime import datetime
from scipy import stats
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, time

# -----------------------------
# 1. Database Connection
# -----------------------------
engine = create_engine('postgresql://postgres.szcndefhegrwzhuzgozm:Nishant%407879692581@aws-1-ap-south-1.pooler.supabase.com:5432/postgres') 

# -----------------------------
# 2. Fetch Raw Data
# -----------------------------
query = "SELECT * FROM crypto;"
df = pd.read_sql(query, engine)

print("Before Cleaning:")
print(df.info())
print(df.head())

# -----------------------------
# 3. Clean Per-Coin
# -----------------------------
df = df.sort_values(by=["symbol", "date"])

cleaned_list = []

for symbol, d in df.groupby("symbol"):
    
    # Missing values fill for each coin
    d = d.ffill().bfill()

    # Outlier removal — per coin
    if d['close'].std() != 0:  # avoid division by zero
        z_scores = np.abs(stats.zscore(d['close']))
        d = d[(z_scores < 3)]

    cleaned_list.append(d)

df_clean = pd.concat(cleaned_list)

missing_filled = df.isnull().sum().sum() - df_clean.isnull().sum().sum()
outliers_removed = len(df) - len(df_clean)

# -----------------------------
# 4. Insert Clean Data (Per Coin Date Logic)
# -----------------------------
with engine.connect() as conn:
    table_check = conn.execute(
        text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='cleaned_crypto_data');")
    )
    exists = table_check.scalar()

    last_dates = {}
    
    if exists:
        result = conn.execute(
            text("SELECT symbol, MAX(date) FROM cleaned_crypto_data GROUP BY symbol;")
        ).fetchall()

        last_dates = {row[0]: row[1] for row in result}

# Filter only new rows per coin
filtered_list = []

for symbol, d in df_clean.groupby("symbol"):

    if symbol in last_dates:
        d_new = d[d["date"] > last_dates[symbol]]
        filtered_list.append(d_new)
    else:
        filtered_list.append(d)  # first time data insert

df_new = pd.concat(filtered_list)

# Insert
df_new.to_sql("cleaned_crypto_data", engine, if_exists="append", index=False)

# -----------------------------
# 6. Save Logs
# -----------------------------
log_data = pd.DataFrame([{
    "run_time": datetime.now(),
    "records_inserted": len(df_new),
    "missing_filled": int(missing_filled),
    "outliers_removed": int(outliers_removed)
}])

log_data.to_sql("logs_cleaning", engine, if_exists="append", index=False)

# -----------------------------
# 7. Summary
# -----------------------------
print("After Cleaning:")
print(df_clean.info())
print(f"Inserted {len(df_new)} new records into cleaned_crypto_data")
print(f"Missing values filled: {missing_filled}, Outliers removed: {outliers_removed}")
