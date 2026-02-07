# Luxury Customer Analytics Platform

> An end-to-end data analytics project featuring customer segmentation, churn prediction, demand forecasting, and executive dashboards.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project demonstrates key data analytics skills valued by luxury fashion brands:

| Module | Business Application | Methods |
|--------|---------------------|---------|
| **Customer Segmentation** | Identify VIC/VVIC clients | RFM Analysis, K-Means |
| **Churn Prediction** | Predict at-risk customers | XGBoost, Logistic Regression |
| **Demand Forecasting** | Revenue & inventory planning | SARIMAX, Prophet |
| **Executive Dashboard** | Stakeholder reporting | Power BI |

---

## 🗂️ Project Structure

```
luxury-customer-analytics/
├── data/
│   ├── raw/                    # UCI Online Retail II dataset
│   └── processed/              # Cleaned data for analysis
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_customer_segmentation.ipynb
│   ├── 03_churn_prediction.ipynb
│   └── 04_demand_forecasting.ipynb
├── sql/
│   └── feature_queries.sql     # SQL for feature engineering
├── models/
│   └── churn_model.pkl         # Trained models
├── dashboards/
│   └── executive_dashboard.pbix
└── reports/
    └── insights_summary.md
```

---

## 📊 Dataset

**UCI Online Retail II Dataset**
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **Records:** 1,067,371 transactions
- **Period:** Dec 2009 – Dec 2011
- **Type:** Real transaction data from UK retailer

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn xgboost statsmodels prophet plotly openpyxl

# 2. Run notebooks in order
jupyter notebook notebooks/
```

---

## 📈 Key Deliverables

1. **Customer Tier Classification** — VVIC, VIC, Prestige, One-time segments
2. **Churn Risk Scores** — Probability of customer attrition
3. **6-Month Sales Forecast** — Revenue predictions with confidence intervals
4. **Power BI Dashboard** — Real-time KPIs for stakeholders

---

## 🎯 Skills Demonstrated

- SQL & Python for data manipulation
- RFM Analysis & Customer Lifetime Value
- Machine Learning (XGBoost, Logistic Regression)
- Time Series Forecasting (SARIMAX, Prophet)
- Data Visualization (Power BI, Plotly)
- Business storytelling for non-technical audiences

---

## 👩‍💻 Author

**Likitha Gedipudi**  
MS Data Science | 2X Kaggle Expert
