import pandas as pd

# Load the full dataset
df = pd.read_csv("transactions.csv")

# Sample 30,000 random rows without replacement
sample_df = df.sample(n=30000, random_state=42)

# Optionally save the sampled data to a new CSV
sample_df.to_csv("transactions_sampled_30000.csv", index=False)

print("Sample of 30,000 rows saved to transactions_sampled_30000.csv")
