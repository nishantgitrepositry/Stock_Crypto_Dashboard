import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import text
from datetime import datetime
from scipy import stats

# -----------------------------
# 1. Database Connection
# -----------------------------
engine = sqlalchemy.create_engine("postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db")

# -----------------------------
# 2. Fetch Raw Data
# -----------------------------
query = "SELECT * FROM crypto;"
df = pd.read_sql(query, engine)

print("Before Cleaning:")
print(df.info())
print(df.head())

# -----------------------------
# 3. Handle Missing Values
# -----------------------------
df = df.sort_values(by="date")
missing_before = df.isnull().sum().sum()

df = df.ffill().bfill()
missing_after = df.isnull().sum().sum()
missing_filled = missing_before - missing_after

# -----------------------------
# 4. Handle Outliers (Z-score)
# -----------------------------
z_scores = np.abs(stats.zscore(df['close']))
outliers_removed = (z_scores >= 3).sum()

df = df[(z_scores < 3)]

# -----------------------------
# 5. Insert Clean Data Safely
# -----------------------------
with engine.connect() as conn:
    # Table exist karta hai ya nahi check karo
    table_check = conn.execute(
        text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='cleaned_crypto_data');")
    )
    exists = table_check.scalar()

    last_date = None
    if exists:
        result = conn.execute(text("SELECT MAX(date) FROM cleaned_crypto_data;"))
        last_date = result.scalar()

# Sirf naya data lo agar table already hai
if last_date:
    df_new = df[df['date'] > last_date]
else:
    df_new = df.copy()

# Insert into cleaned table
df_new.to_sql("cleaned_crypto_data", engine, if_exists="append", index=False)

# -----------------------------
# 6. Save Logs Safely
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
print(df.info())
print(f"Inserted {len(df_new)} new records into cleaned_crypto_data")
print(f"Missing values filled: {missing_filled}, Outliers removed: {outliers_removed}")
