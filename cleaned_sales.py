import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Raw dirty data - real shop sales with errors
data = {
    "Product":  ["Rice", "Beans", "Rice", "Garri", "Beans", "Rice", "Garri", "Beans"],
    "Price":    ["20000", "16000", None, "15000", "16000", "20000", "$15,000", "17000"],
    "Quantity": [5, 3, 5, 8, None, 5, 8, 2],
    "Date":     ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02", "2024-01-03"]
}
df = pd.DataFrame(data)

# Step 2: Remove duplicate rows (keep first occurrence)
df = df.drop_duplicates(keep="first")

# Step 3: Clean Price column - remove $ and , then convert to numeric
# Example: "$15,000" -> "15000" -> 15000.0
df["Price"] = df["Price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
df["Price"] = df["Price"].replace(["nan", "None", "NaN", "NAT"], np.nan)
df["Price"] = pd.to_numeric(df["Price"], errors='coerce')
df["Price"] = df["Price"].fillna(df["Price"].mean())  # Fill missing with average price

# Step 4: Clean Quantity column - fill missing with mean
df["Quantity"] = df["Quantity"].fillna(df["Quantity"].mean())

# Step 5: Convert Date string to datetime object for feature engineering
df["Date"] = pd.to_datetime(df["Date"])

print("CLEANED DATASET:")
print(df)
print("\n--- dtypes (ML-ready) ---")
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())

# Step 6: Feature Engineering - Calendar features from Date
print("\n--- Feature Engineering: Calendar ---")
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.day_of_week  # 0=Mon, 6=Sun (fixed warning)
df["IsWeekend"] = df["Weekday"] >= 5
print(df[["Date", "Day", "Weekday", "IsWeekend"]].head())

# Step 7: Feature Engineering - PriceChange per product
print("\n--- Feature Engineering: PriceChange ---")
df = df.sort_values(by="Date")
df["PrevPrice"] = df.groupby("Product")["Price"].shift(1)  # Previous price of same product
df["PriceChange"] = df["Price"] - df["PrevPrice"]  # Price difference
# First occurrence has no previous price, fill with 0 (no change)
df["PriceChange"] = df["PriceChange"].fillna(0)
df["PrevPrice"] = df["PrevPrice"].fillna(df["Price"])
print(df[["Product", "Date", "Price", "PrevPrice", "PriceChange"]].head(10))

# Step 8: Business Insights
print("\n--- Total Quantity per Product ---")
print(df.groupby("Product")["Quantity"].sum())
print("\n--- Average Price per Product ---")
print(df.groupby("Product")["Price"].mean().sort_values(ascending=False))

# Step 9: Save cleaned data for ML
df.to_csv("cleaned_sales.csv", index=False)
print("\n cleaned_sales.csv — ready for ML!")

# Step 10: Encoding - Turn text Product into numbers (One-Hot)
print("\n--- Encoding: Product -> Numbers ---")
print("Before encoding:")
print(df[["Product", "Price"]].head())
df_encoded = pd.get_dummies(df, columns=["Product"])
print("\nAfter One-Hot encoding:")
print(df_encoded[["Price", "Product_Beans", "Product_Garri", "Product_Rice"]].head())

# Step 11: Train/Test Split - 75% train, 25% test
print("\n--- Train/Test Split ---")
X = df_encoded[["Price", "Product_Beans", "Product_Garri", "Product_Rice"]]
y = df_encoded["Quantity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"Total: {len(X)}, Train: {len(X_train)}, Test: {len(X_test)}")

# Step 12: Train first AI model - Linear Regression
print("\n--- Training First AI Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Real Quantity: {y_test.values}")
print(f"Predicted: {y_pred}")
print(f"Mean error: {mean_absolute_error(y_test, y_pred):.2f} units")
print("\n MY FIRST AI MODEL TO PREDICT SALES!")