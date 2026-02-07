# %% [markdown]
# # Churn Prediction Model
# 
# **Objective:** Predict customer churn risk using machine learning, enabling proactive retention strategies.
# 
# Churn Definition: Customer inactive for 90+ days

# %% [markdown]
# ## 1. Setup & Load Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
print("📦 Libraries loaded!")

# %%
# Load cleaned data from segmentation notebook
df = pd.read_excel('../data/raw/online_retail_II.xlsx', sheet_name='Year 2010-2011')

# Basic cleaning
df_clean = df.dropna(subset=['Customer ID'])
df_clean = df_clean[~df_clean['Invoice'].astype(str).str.startswith('C')]
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']

print(f"📊 Loaded {len(df_clean):,} transactions from {df_clean['Customer ID'].nunique():,} customers")

# %% [markdown]
# ## 2. Define Churn & Create Features

# %%
# Reference date
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)

# Create customer-level features
customer_features = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': ['min', 'max', lambda x: (reference_date - x.max()).days],
    'Invoice': 'nunique',
    'TotalAmount': ['sum', 'mean', 'std'],
    'Quantity': ['sum', 'mean'],
    'StockCode': 'nunique',
    'Country': 'first'
}).reset_index()

# Flatten column names
customer_features.columns = [
    'CustomerID', 'FirstPurchase', 'LastPurchase', 'Recency',
    'Frequency', 'TotalSpend', 'AvgOrderValue', 'StdOrderValue',
    'TotalQuantity', 'AvgQuantity', 'UniqueProducts', 'Country'
]

# Calculate additional features
customer_features['CustomerAge'] = (customer_features['LastPurchase'] - customer_features['FirstPurchase']).dt.days
customer_features['AvgDaysBetweenPurchases'] = customer_features['CustomerAge'] / customer_features['Frequency'].clip(lower=1)
customer_features['IsUK'] = (customer_features['Country'] == 'United Kingdom').astype(int)

# Fill NaN in std with 0 (single purchase customers)
customer_features['StdOrderValue'] = customer_features['StdOrderValue'].fillna(0)

print(f"📊 Created {len(customer_features.columns)} features for {len(customer_features):,} customers")

# %%
# Define CHURN: No purchase in last 90 days
CHURN_THRESHOLD = 90

customer_features['Churned'] = (customer_features['Recency'] > CHURN_THRESHOLD).astype(int)

print(f"\n🔄 CHURN DISTRIBUTION (>{CHURN_THRESHOLD} days inactive)")
print("="*50)
churn_counts = customer_features['Churned'].value_counts()
print(f"Active (0):  {churn_counts[0]:,} ({churn_counts[0]/len(customer_features)*100:.1f}%)")
print(f"Churned (1): {churn_counts[1]:,} ({churn_counts[1]/len(customer_features)*100:.1f}%)")

# %% [markdown]
# ## 3. Prepare Data for Modeling

# %%
# Select features for modeling
feature_cols = [
    'Frequency', 'TotalSpend', 'AvgOrderValue', 'StdOrderValue',
    'TotalQuantity', 'AvgQuantity', 'UniqueProducts', 'CustomerAge',
    'AvgDaysBetweenPurchases', 'IsUK'
]

X = customer_features[feature_cols].copy()
y = customer_features['Churned']

# Handle any remaining NaN
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Training set: {len(X_train):,} | Test set: {len(X_test):,}")

# %% [markdown]
# ## 4. Train Multiple Models

# %%
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}

print("🏋️ TRAINING MODELS...")
print("="*60)

for name, model in models.items():
    # Use scaled data for Logistic Regression
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train_scaled if name == 'Logistic Regression' else X_train, 
                                 y_train, cv=5, scoring='roc_auc')
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  • Test AUC: {auc:.4f}")
    print(f"  • CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %% [markdown]
# ## 5. Model Evaluation

# %%
# Select best model (XGBoost typically performs best)
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_result = results[best_model_name]

print(f"🏆 BEST MODEL: {best_model_name}")
print("="*60)
print(f"\nClassification Report:")
print(classification_report(y_test, best_result['y_pred'], target_names=['Active', 'Churned']))

# %%
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")

axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('../reports/churn_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Saved to reports/churn_model_evaluation.png")

# %% [markdown]
# ## 6. Feature Importance

# %%
# Get feature importance from XGBoost
if best_model_name == 'XGBoost':
    importance = best_result['model'].feature_importances_
else:
    importance = results['XGBoost']['model'].feature_importances_

feat_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importance
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feat_importance['Feature'], feat_importance['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.title('Feature Importance for Churn Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/churn_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 TOP 5 CHURN PREDICTORS:")
for _, row in feat_importance.tail(5).iloc[::-1].iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.3f}")

# %% [markdown]
# ## 7. Generate Churn Risk Scores

# %%
# Score all customers
X_all = customer_features[feature_cols].fillna(0)
churn_probs = results['XGBoost']['model'].predict_proba(X_all)[:, 1]

customer_features['ChurnProbability'] = churn_probs
customer_features['ChurnRisk'] = pd.cut(churn_probs, 
                                         bins=[0, 0.3, 0.6, 1.0],
                                         labels=['Low', 'Medium', 'High'])

print("🎯 CHURN RISK DISTRIBUTION")
print("="*50)
risk_dist = customer_features['ChurnRisk'].value_counts()
for risk in ['High', 'Medium', 'Low']:
    pct = risk_dist[risk] / len(customer_features) * 100
    print(f"{risk:8}: {risk_dist[risk]:>6,} customers ({pct:>5.1f}%)")

# %%
# High-value customers at risk (Priority for retention)
# Merge with segmentation data
rfm = pd.read_csv('../data/processed/customer_segmentation.csv')
customer_features_merged = customer_features.merge(rfm[['CustomerID', 'CustomerTier', 'CLV']], 
                                                    on='CustomerID', how='left')

priority_retention = customer_features_merged[
    (customer_features_merged['ChurnRisk'] == 'High') & 
    (customer_features_merged['CustomerTier'].isin(['VVIC', 'VIC']))
]

print(f"\n🚨 PRIORITY RETENTION LIST")
print("="*50)
print(f"High-value customers at high churn risk: {len(priority_retention):,}")
print(f"\nTotal CLV at risk: £{priority_retention['CLV'].sum():,.2f}")

# %% [markdown]
# ## 8. Export Results

# %%
# Export churn predictions
export_cols = ['CustomerID', 'Recency', 'Frequency', 'TotalSpend', 
               'ChurnProbability', 'ChurnRisk', 'Churned']
export_df = customer_features[export_cols].copy()
export_df.to_csv('../data/processed/churn_predictions.csv', index=False)

print(f"✅ Exported churn predictions to data/processed/churn_predictions.csv")

# Save model
import joblib
joblib.dump(results['XGBoost']['model'], '../models/churn_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Saved XGBoost model to models/churn_model.pkl")

# %%
# Summary for resume
print("\n" + "="*60)
print("📝 RESUME BULLET POINT")
print("="*60)
print(f"""
Built churn prediction model using XGBoost to identify at-risk customers,
achieving {results['XGBoost']['auc']:.2f} AUC on {len(customer_features):,} customers. 
Identified {len(priority_retention):,} high-value clients at risk, representing 
£{priority_retention['CLV'].sum():,.0f} in CLV, enabling proactive retention strategies.
""")
