
import pandas as pd

df = pd.read_csv('runnable/btcusd_1-min_data.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.dropna()
df = df.reset_index(drop=True)

# Select every 70th row to reduce ~7M rows to ~100K
df_downsampled = df.iloc[::70]

# Save to CSV
df_downsampled.to_csv('bitcoin_downsampled_time.csv', index=False)

