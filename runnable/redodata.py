import pandas as pd

# Load the CSV file
df = pd.read_csv("runnable/preddata.csv")

# Convert price to Lakhs
df["price_lakhs"] = df["price_numeric"] / 1e5  # or 100000

df.drop("price_numeric", axis=1, inplace=True)

# Save the modified dataframe (optional)
df.to_csv("runnable/preddata2.csv", index=False)

# Display updated DataFrame
print(df.head())
