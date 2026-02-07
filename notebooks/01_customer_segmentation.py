# %% [markdown]
# # Customer Segmentation & RFM Analysis
# 
# **Objective:** Segment customers into VVIC, VIC, Prestige, and One-time tiers using RFM (Recency, Frequency, Monetary) analysis.
# 
# This mirrors the customer analytics work done at luxury brands like Louis Vuitton.

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("📦 Libraries loaded successfully!")

# %%
# Load the dataset
df = pd.read_excel('../data/raw/online_retail_II.xlsx', sheet_name='Year 2010-2011')

print(f"📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\n📋 Columns: {list(df.columns)}")
df.head()

# %% [markdown]
# ## 2. Data Exploration & Cleaning

# %%
# Basic info
print("📊 DATA OVERVIEW")
print("="*50)
print(f"Total Transactions: {len(df):,}")
print(f"Unique Customers: {df['Customer ID'].nunique():,}")
print(f"Unique Products: {df['StockCode'].nunique():,}")
print(f"Countries: {df['Country'].nunique()}")
print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

# %%
# Check missing values
print("\n🔍 MISSING VALUES")
print("="*50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({'Missing': missing, 'Percentage': missing_pct}))

# %%
# Data Cleaning
print("\n🧹 CLEANING DATA...")

# Remove rows without Customer ID (can't segment anonymous customers)
df_clean = df.dropna(subset=['Customer ID'])
print(f"✓ Removed {len(df) - len(df_clean):,} rows without Customer ID")

# Remove cancelled orders (Invoice starts with 'C')
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
print(f"✓ Removed cancelled orders")

# Remove negative quantities and prices
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
print(f"✓ Removed negative quantities/prices")

# Calculate total amount per transaction
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']

print(f"\n📊 Final Dataset: {len(df_clean):,} transactions")

# %% [markdown]
# ## 3. RFM Analysis
# 
# - **Recency:** Days since last purchase
# - **Frequency:** Total number of transactions
# - **Monetary:** Total spend

# %%
# Set reference date (day after last transaction)
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"📅 Reference Date: {reference_date}")

# Calculate RFM metrics per customer
rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'Invoice': 'nunique',  # Frequency
    'TotalAmount': 'sum'  # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"\n📊 RFM Table Shape: {rfm.shape}")
rfm.head(10)

# %%
# RFM Statistics
print("📈 RFM STATISTICS")
print("="*50)
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))

# %% [markdown]
# ## 4. RFM Scoring (1-5 Scale)

# %%
# Create RFM scores using quintiles
# For Recency: lower is better (more recent), so we reverse the labels
rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')

# For Frequency and Monetary: higher is better
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Combine to create RFM Score
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

rfm.head(10)

# %% [markdown]
# ## 5. Customer Tier Classification
# 
# Mapping to luxury brand tiers (similar to Louis Vuitton):
# - **VVIC (Very Very Important Client):** Top 5% - RFM Score >= 444
# - **VIC (Very Important Client):** Top 15% - RFM Score >= 333
# - **Prestige:** Regular customers - RFM Score >= 222
# - **One-time:** Infrequent/lapsed customers

# %%
def assign_customer_tier(row):
    """Assign customer tier based on RFM scores"""
    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
    
    # VVIC: Best customers (R>=4, F>=4, M>=4)
    if r >= 4 and f >= 4 and m >= 4:
        return 'VVIC'
    
    # VIC: High-value customers (R>=3, F>=3, M>=3)
    elif r >= 3 and f >= 3 and m >= 3:
        return 'VIC'
    
    # Prestige: Regular customers (any score >= 2 and not one-time)
    elif (r >= 2 and f >= 2) or m >= 3:
        return 'Prestige'
    
    # One-time: Low engagement
    else:
        return 'One-time'

rfm['CustomerTier'] = rfm.apply(assign_customer_tier, axis=1)

# View distribution
tier_dist = rfm['CustomerTier'].value_counts()
tier_pct = (tier_dist / len(rfm) * 100).round(2)

print("🏆 CUSTOMER TIER DISTRIBUTION")
print("="*50)
for tier in ['VVIC', 'VIC', 'Prestige', 'One-time']:
    if tier in tier_dist.index:
        print(f"{tier:12}: {tier_dist[tier]:>6,} customers ({tier_pct[tier]:>5.1f}%)")

# %%
# Visualize tier distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pie chart
colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#808080']
tier_order = ['VVIC', 'VIC', 'Prestige', 'One-time']
tier_counts = [tier_dist.get(t, 0) for t in tier_order]

axes[0].pie(tier_counts, labels=tier_order, autopct='%1.1f%%', colors=colors, explode=(0.05, 0, 0, 0))
axes[0].set_title('Customer Tier Distribution', fontsize=14, fontweight='bold')

# Bar chart with revenue
tier_revenue = rfm.groupby('CustomerTier')['Monetary'].sum().reindex(tier_order)
bars = axes[1].bar(tier_order, tier_revenue / 1000, color=colors, edgecolor='black')
axes[1].set_ylabel('Total Revenue (£ thousands)', fontsize=12)
axes[1].set_xlabel('Customer Tier', fontsize=12)
axes[1].set_title('Revenue by Customer Tier', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar, val in zip(bars, tier_revenue / 1000):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                 f'£{val:,.0f}K', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/customer_tier_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Chart saved to reports/customer_tier_distribution.png")

# %% [markdown]
# ## 6. Customer Lifetime Value (CLV) Estimation

# %%
# Simple CLV calculation
# CLV = Average Order Value × Purchase Frequency × Average Customer Lifespan

# Calculate metrics
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']

# Estimate customer lifespan (in years) based on activity
max_recency = rfm['Recency'].max()
rfm['ActiveDays'] = max_recency - rfm['Recency']
rfm['EstimatedLifespan'] = (rfm['ActiveDays'] / 365).clip(lower=0.1)  # Minimum 0.1 years

# Historical CLV
rfm['CLV'] = rfm['Monetary']  # Already represents total value

# Predicted Annual CLV (if customer continues at same rate)
rfm['AnnualCLV'] = rfm['AvgOrderValue'] * (rfm['Frequency'] / rfm['EstimatedLifespan'].clip(lower=0.5))

print("💰 CLV BY CUSTOMER TIER")
print("="*50)
clv_summary = rfm.groupby('CustomerTier').agg({
    'CLV': ['mean', 'sum'],
    'AnnualCLV': 'mean',
    'CustomerID': 'count'
}).round(2)
clv_summary.columns = ['Avg CLV', 'Total CLV', 'Avg Annual CLV', 'Customer Count']
print(clv_summary.reindex(['VVIC', 'VIC', 'Prestige', 'One-time']))

# %% [markdown]
# ## 7. Key Insights & Business Recommendations

# %%
print("="*60)
print("📊 KEY INSIGHTS")
print("="*60)

# Calculate key metrics
total_customers = len(rfm)
total_revenue = rfm['Monetary'].sum()

vvic_pct = len(rfm[rfm['CustomerTier'] == 'VVIC']) / total_customers * 100
vvic_revenue_pct = rfm[rfm['CustomerTier'] == 'VVIC']['Monetary'].sum() / total_revenue * 100

vic_pct = len(rfm[rfm['CustomerTier'] == 'VIC']) / total_customers * 100
vic_revenue_pct = rfm[rfm['CustomerTier'] == 'VIC']['Monetary'].sum() / total_revenue * 100

print(f"""
1. CUSTOMER CONCENTRATION
   • VVIC customers ({vvic_pct:.1f}% of base) generate {vvic_revenue_pct:.1f}% of revenue
   • VIC customers ({vic_pct:.1f}% of base) generate {vic_revenue_pct:.1f}% of revenue
   • Top 2 tiers = {vvic_pct + vic_pct:.1f}% of customers → {vvic_revenue_pct + vic_revenue_pct:.1f}% of revenue

2. BUSINESS RECOMMENDATIONS
   • 🎯 VVIC: Personal outreach, exclusive previews, VIP events
   • 💎 VIC: Upgrade program to VVIC, loyalty rewards
   • ⭐ Prestige: Email campaigns, seasonal promotions
   • 🔄 One-time: Re-engagement campaigns, win-back offers

3. CHURN RISK
   • Customers with Recency > 90 days: {len(rfm[rfm['Recency'] > 90]):,} ({len(rfm[rfm['Recency'] > 90])/total_customers*100:.1f}%)
   • Recommended: Proactive retention for high-value at-risk customers
""")

# %% [markdown]
# ## 8. Export Data for Power BI Dashboard

# %%
# Export RFM data for Power BI
export_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
               'R_Score', 'F_Score', 'M_Score', 'RFM_Score', 
               'CustomerTier', 'CLV', 'AvgOrderValue']

rfm_export = rfm[export_cols].copy()
rfm_export.to_csv('../data/processed/customer_segmentation.csv', index=False)

print(f"✅ Exported {len(rfm_export):,} customer records to data/processed/customer_segmentation.csv")
print("\n📊 Ready for Power BI Dashboard!")

# %%
# Summary statistics for dashboard
summary = pd.DataFrame({
    'Metric': ['Total Customers', 'Total Revenue', 'Avg Order Value', 'VVIC Count', 'VIC Count'],
    'Value': [
        total_customers,
        f"£{total_revenue:,.2f}",
        f"£{rfm['AvgOrderValue'].mean():,.2f}",
        len(rfm[rfm['CustomerTier'] == 'VVIC']),
        len(rfm[rfm['CustomerTier'] == 'VIC'])
    ]
})
print("\n📈 DASHBOARD SUMMARY METRICS")
print(summary.to_string(index=False))
