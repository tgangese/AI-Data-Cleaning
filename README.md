# AI Data Cleaning - Sales Log

My first AI/Data Science project. 
Learned how to clean dirty shop sales data for Machine Learning.

## Problem
Raw sales data had:
- Missing values (NaN)
- Wrong formats ($15,000)
- Duplicate rows
- String prices instead of numbers

## What I Did (Pandas)
1. Checked data with `df.dtypes` and `isnull().sum()`
2. Removed duplicates with `drop_duplicates()`
3. Cleaned Price: removed `$` and `,` , converted to float with `pd.to_numeric`
4. Filled missing Price & Quantity with mean `fillna(mean)`
5. Converted Date to datetime
6. Validated: 0 missing, 0 duplicates

## Result
Clean dataset: 8 rows, Price float64, Quantity float64, Date datetime64
Saved as `cleaned_sales.csv` - ready for ML

## Tech
Python, Pandas, NumPy