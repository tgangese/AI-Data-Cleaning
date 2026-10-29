# AI Data Cleaning & Sales Prediction

My first AI/Data Science projects — from dirty data to AI model.

## Project 1: Data Cleaning (Completed )
Raw sales data had:
- Missing values (NaN)
- Wrong formats ($15,000)
- Duplicate rows
- String prices instead of numbers

**What I did:**
1. Removed duplicates with `drop_duplicates()`
2. Cleaned Price: removed `$` and `,` , converted to float
3. Filled missing Price & Quantity with mean
4. Converted Date to datetime
5. Result: 0 missing, 0 duplicates, ML-ready — saved as `cleaned_sales.csv`

## Project 2: Feature Engineering + First AI Model (Completed )
**Features created:**
- Calendar: Day, Month, Weekday, IsWeekend from Date
- PriceChange: difference from previous price per product
- Encoding: One-Hot encoded Product -> Product_Beans, Product_Garri, Product_Rice (True/False -> 1/0)

**ML Pipeline:**
- Train/Test Split: 6 rows train (75%), 2 rows test (25%)
- Model: LinearRegression to predict Quantity from Price + Product
- Result: Mean error 0.60 units (Real: [3, 5] vs Predicted: [3.72, 4.52])

**Tech:** Python, Pandas, NumPy, Scikit-Learn

**How to run:**
```bash
source .venv/bin/activate
python3 cleaned_sales.py