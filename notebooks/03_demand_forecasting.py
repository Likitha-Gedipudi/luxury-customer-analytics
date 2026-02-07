# %% [markdown]
# # Demand Forecasting
# 
# **Objective:** Forecast future sales using time series models (SARIMAX & Prophet) for inventory planning.

# %% [markdown]
# ## 1. Setup & Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
print("📦 Libraries loaded!")

# %%
# Load data
df = pd.read_excel('../data/raw/online_retail_II.xlsx', sheet_name='Year 2010-2011')

# Clean data
df_clean = df.dropna(subset=['Customer ID'])
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

print(f"📊 Loaded {len(df_clean):,} transactions")

# %% [markdown]
# ## 2. Create Time Series Data

# %%
# Aggregate to daily sales
daily_sales = df_clean.groupby(df_clean['InvoiceDate'].dt.date).agg({
    'TotalAmount': 'sum',
    'Invoice': 'nunique',
    'Customer ID': 'nunique'
}).reset_index()

daily_sales.columns = ['Date', 'Revenue', 'Transactions', 'UniqueCustomers']
daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
daily_sales = daily_sales.set_index('Date').sort_index()

# Handle missing dates (weekends/holidays)
daily_sales = daily_sales.asfreq('D', fill_value=0)

print(f"📅 Time Series Range: {daily_sales.index.min()} to {daily_sales.index.max()}")
print(f"📊 Total Days: {len(daily_sales)}")

# %%
# Weekly aggregation (smoother for forecasting)
weekly_sales = daily_sales.resample('W').sum()
weekly_sales = weekly_sales[weekly_sales['Revenue'] > 0]

print(f"📅 Weekly Data Points: {len(weekly_sales)}")
weekly_sales.head()

# %% [markdown]
# ## 3. Time Series Visualization

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Weekly Revenue
axes[0].plot(weekly_sales.index, weekly_sales['Revenue'], color='steelblue', linewidth=1.5)
axes[0].fill_between(weekly_sales.index, weekly_sales['Revenue'], alpha=0.3)
axes[0].set_title('Weekly Revenue Over Time', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Revenue (£)')

# Weekly Transactions
axes[1].plot(weekly_sales.index, weekly_sales['Transactions'], color='green', linewidth=1.5)
axes[1].fill_between(weekly_sales.index, weekly_sales['Transactions'], alpha=0.3, color='green')
axes[1].set_title('Weekly Transactions Over Time', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Transactions')

# Weekly Unique Customers
axes[2].plot(weekly_sales.index, weekly_sales['UniqueCustomers'], color='purple', linewidth=1.5)
axes[2].fill_between(weekly_sales.index, weekly_sales['UniqueCustomers'], alpha=0.3, color='purple')
axes[2].set_title('Weekly Unique Customers Over Time', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Customers')
axes[2].set_xlabel('Date')

plt.tight_layout()
plt.savefig('../reports/sales_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 4. Seasonal Decomposition

# %%
# Decompose the time series
decomposition = seasonal_decompose(weekly_sales['Revenue'], model='additive', period=4)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))

decomposition.observed.plot(ax=axes[0], color='steelblue')
axes[0].set_title('Observed', fontweight='bold')

decomposition.trend.plot(ax=axes[1], color='green')
axes[1].set_title('Trend', fontweight='bold')

decomposition.seasonal.plot(ax=axes[2], color='orange')
axes[2].set_title('Seasonal', fontweight='bold')

decomposition.resid.plot(ax=axes[3], color='red')
axes[3].set_title('Residual', fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Train/Test Split

# %%
# Use last 8 weeks for testing
train_size = len(weekly_sales) - 8
train = weekly_sales['Revenue'].iloc[:train_size]
test = weekly_sales['Revenue'].iloc[train_size:]

print(f"📊 Training: {len(train)} weeks | Testing: {len(test)} weeks")

# %% [markdown]
# ## 6. SARIMAX Model

# %%
# Fit SARIMAX model
# SARIMAX(p,d,q)(P,D,Q,s) - using simple parameters
print("🏋️ Training SARIMAX model...")

model = SARIMAX(train, 
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False)

sarimax_fit = model.fit(disp=False)

# Forecast
sarimax_forecast = sarimax_fit.get_forecast(steps=len(test))
sarimax_pred = sarimax_forecast.predicted_mean
sarimax_ci = sarimax_forecast.conf_int()

# Metrics
sarimax_mae = mean_absolute_error(test, sarimax_pred)
sarimax_rmse = np.sqrt(mean_squared_error(test, sarimax_pred))
sarimax_mape = np.mean(np.abs((test - sarimax_pred) / test)) * 100

print(f"\n📈 SARIMAX PERFORMANCE")
print("="*50)
print(f"MAE:  £{sarimax_mae:,.2f}")
print(f"RMSE: £{sarimax_rmse:,.2f}")
print(f"MAPE: {sarimax_mape:.2f}%")

# %% [markdown]
# ## 7. Forecast Visualization

# %%
fig, ax = plt.subplots(figsize=(14, 6))

# Plot training data
ax.plot(train.index, train.values, label='Training Data', color='steelblue')

# Plot test data
ax.plot(test.index, test.values, label='Actual', color='green', linewidth=2)

# Plot forecast
ax.plot(test.index, sarimax_pred.values, label='SARIMAX Forecast', 
        color='red', linestyle='--', linewidth=2)

# Confidence interval
ax.fill_between(test.index, 
                sarimax_ci.iloc[:, 0], 
                sarimax_ci.iloc[:, 1], 
                alpha=0.2, color='red', label='95% CI')

ax.set_title('Weekly Revenue Forecast - SARIMAX', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue (£)')
ax.legend()

plt.tight_layout()
plt.savefig('../reports/demand_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Future Forecast (Next 12 Weeks)

# %%
# Retrain on full data
full_model = SARIMAX(weekly_sales['Revenue'], 
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 4),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

full_fit = full_model.fit(disp=False)

# Forecast next 12 weeks
future_steps = 12
future_forecast = full_fit.get_forecast(steps=future_steps)
future_pred = future_forecast.predicted_mean
future_ci = future_forecast.conf_int()

# Create future dates
last_date = weekly_sales.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), 
                              periods=future_steps, freq='W')

# %%
# Visualization
fig, ax = plt.subplots(figsize=(14, 6))

# Historical
ax.plot(weekly_sales.index, weekly_sales['Revenue'], 
        label='Historical', color='steelblue')

# Forecast
ax.plot(future_dates, future_pred.values, 
        label='12-Week Forecast', color='red', linewidth=2, linestyle='--')
ax.fill_between(future_dates, 
                future_ci.iloc[:, 0].values, 
                future_ci.iloc[:, 1].values, 
                alpha=0.2, color='red', label='95% CI')

ax.axvline(x=weekly_sales.index[-1], color='gray', linestyle=':', label='Forecast Start')
ax.set_title('12-Week Revenue Forecast', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue (£)')
ax.legend()

plt.tight_layout()
plt.savefig('../reports/future_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Forecast summary
forecast_summary = pd.DataFrame({
    'Week': range(1, future_steps + 1),
    'Date': future_dates,
    'Forecast': future_pred.values,
    'Lower_CI': future_ci.iloc[:, 0].values,
    'Upper_CI': future_ci.iloc[:, 1].values
})

print("\n📈 12-WEEK FORECAST SUMMARY")
print("="*60)
print(f"Total Forecasted Revenue: £{future_pred.sum():,.2f}")
print(f"Weekly Average: £{future_pred.mean():,.2f}")
print(f"Range: £{future_pred.min():,.2f} - £{future_pred.max():,.2f}")

forecast_summary.to_csv('../data/processed/revenue_forecast.csv', index=False)
print("\n✅ Saved forecast to data/processed/revenue_forecast.csv")

# %% [markdown]
# ## 9. Export for Power BI

# %%
# Export time series data
weekly_export = weekly_sales.reset_index()
weekly_export.columns = ['Date', 'Revenue', 'Transactions', 'UniqueCustomers']
weekly_export.to_csv('../data/processed/weekly_sales.csv', index=False)

print("✅ Exported weekly sales to data/processed/weekly_sales.csv")

# %%
# Summary for resume
print("\n" + "="*60)
print("📝 RESUME BULLET POINT")
print("="*60)
print(f"""
Developed demand forecasting pipeline using SARIMAX with seasonal decomposition,
achieving MAPE of {sarimax_mape:.1f}% on weekly revenue predictions. Generated
12-week forecasts totaling £{future_pred.sum():,.0f} to support inventory
planning and early risk identification.
""")
