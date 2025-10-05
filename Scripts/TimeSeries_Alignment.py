# Day 9 (Improved): Multi-Stock + Multi-Crypto Time-Series Alignment & Visualization

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# 1. Connect to PostgreSQL Database
engine = create_engine("postgresql://postgres:Nishant%40123@localhost:5432/stock_crypto_db")

# 2. Load Cleaned Stock Data
stock_df = pd.read_sql("SELECT date, symbol, close FROM cleaned_stock_data ORDER BY date", engine)
stock_df['date'] = pd.to_datetime(stock_df['date'])
stock_pivot = stock_df.pivot(index="date", columns="symbol", values="close")

# 3. Load Cleaned Crypto Data (support multiple cryptos)
crypto_df = pd.read_sql("SELECT date, symbol, close FROM cleaned_crypto_data ORDER BY date", engine)
crypto_df['date'] = pd.to_datetime(crypto_df['date'])
crypto_pivot = crypto_df.pivot(index="date", columns="symbol", values="close")

# 4. Align Data (outer join so all dates included)
merged_df = stock_pivot.join(crypto_pivot, how="outer").sort_index()

# 5. Handle Missing Values
merged_df = merged_df.fillna(method="ffill").fillna(method="bfill")

# 6. Plot Function
def plot_prices(df, normalize=False):
    plt.figure(figsize=(14, 7))
    
    if normalize:
        df = df / df.iloc[0] * 100  # Normalize to 100
    
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    
    plt.title("Stocks vs Crypto Prices (Aligned Timeline)" + (" - Normalized" if normalize else ""))
    plt.xlabel("Date")
    plt.ylabel("Normalized Price" if normalize else "Close Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot Raw Prices
plot_prices(merged_df, normalize=False)

# Plot Normalized Trends
plot_prices(merged_df, normalize=True)

# 7. Save Final Merged Data
merged_df.to_sql("merged_stocks_crypto", engine, if_exists="replace", index=True)

print("✅ All Stocks + Crypto data aligned and saved in table: merged_stocks_crypto")
