import pandas as pd
import numpy as np

# Step 1: Raw dirty data (real shop data)
data = {
    "Product":  ["Rice", "Beans", "Rice", "Garri", "Beans", "Rice", "Garri", "Beans"],
    "Price":    ["20000", "16000", None, "15000", "16000", "20000", "$15,000", "17000"],
    "Quantity": [5, 3, 5, 8, None, 5, 8, 2],
    "Date":     ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]
}

df = pd.DataFrame(data)

# Step 2: Remove duplicates
df = df.drop_duplicates(keep="first")

# Step 3: Clean Price column
df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
df["Price"] = df["Price"].replace(["nan", "None", "NaN", "NAT"], np.nan)
df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
df["Price"] = df["Price"].fillna(df["Price"].mean())

# Step 4: Clean Quantity column
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].mean())

# Step 5: Convert Date to real date
df["Date"] = pd.to_datetime(df["Date"])

# Step 6: Final validation
print("CLEANED DATASET:")
print(df)
print("\n--- dtypes (ML-ready) ---")
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Step 7: Business Insights
print("\n--- Total Quantity per Product ---")
print(df.groupby("Product")["Quantity"].sum())

print("\n--- Average Price per Product ---")
print(df.groupby("Product")["Price"].mean().sort_values(ascending=False))

# Step 8: Save cleaned CSV for next project
df.to_csv("cleaned_sales.csv", index=False)
print("\n✅ Saved as cleaned_sales.csv — ready for ML!")