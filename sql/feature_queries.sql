-- Customer Analytics SQL Queries
-- Used for feature engineering and data extraction

-- =================================================
-- 1. RFM METRICS
-- =================================================

WITH customer_rfm AS (
    SELECT 
        "Customer ID" AS customer_id,
        DATEDIFF('day', MAX("InvoiceDate"), CURRENT_DATE) AS recency,
        COUNT(DISTINCT "Invoice") AS frequency,
        SUM("Quantity" * "Price") AS monetary
    FROM online_retail
    WHERE "Customer ID" IS NOT NULL
      AND "Quantity" > 0
      AND "Price" > 0
      AND NOT STARTSWITH("Invoice", 'C')
    GROUP BY "Customer ID"
)
SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
    NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
    NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
FROM customer_rfm;


-- =================================================
-- 2. CUSTOMER TIER CLASSIFICATION
-- =================================================

SELECT 
    customer_id,
    recency,
    frequency,
    monetary,
    CASE 
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'VVIC'
        WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'VIC'
        WHEN (r_score >= 2 AND f_score >= 2) OR m_score >= 3 THEN 'Prestige'
        ELSE 'One-time'
    END AS customer_tier
FROM customer_rfm_scored;


-- =================================================
-- 3. CHURN INDICATORS
-- =================================================

SELECT 
    "Customer ID",
    MAX("InvoiceDate") AS last_purchase,
    DATEDIFF('day', MAX("InvoiceDate"), CURRENT_DATE) AS days_since_purchase,
    CASE 
        WHEN DATEDIFF('day', MAX("InvoiceDate"), CURRENT_DATE) > 90 THEN 1 
        ELSE 0 
    END AS is_churned,
    COUNT(DISTINCT "Invoice") AS total_orders,
    SUM("Quantity" * "Price") AS lifetime_value
FROM online_retail
WHERE "Customer ID" IS NOT NULL
GROUP BY "Customer ID";


-- =================================================
-- 4. WEEKLY SALES AGGREGATION
-- =================================================

SELECT 
    DATE_TRUNC('week', "InvoiceDate") AS week_start,
    SUM("Quantity" * "Price") AS revenue,
    COUNT(DISTINCT "Invoice") AS transactions,
    COUNT(DISTINCT "Customer ID") AS unique_customers,
    AVG("Quantity" * "Price") AS avg_order_value
FROM online_retail
WHERE "Customer ID" IS NOT NULL
  AND "Quantity" > 0
  AND "Price" > 0
GROUP BY DATE_TRUNC('week', "InvoiceDate")
ORDER BY week_start;


-- =================================================
-- 5. PRODUCT PERFORMANCE
-- =================================================

SELECT 
    "StockCode",
    "Description",
    COUNT(DISTINCT "Invoice") AS order_count,
    SUM("Quantity") AS units_sold,
    SUM("Quantity" * "Price") AS revenue,
    COUNT(DISTINCT "Customer ID") AS unique_buyers
FROM online_retail
WHERE "Quantity" > 0 AND "Price" > 0
GROUP BY "StockCode", "Description"
ORDER BY revenue DESC
LIMIT 20;


-- =================================================
-- 6. GEOGRAPHIC ANALYSIS
-- =================================================

SELECT 
    "Country",
    COUNT(DISTINCT "Customer ID") AS customers,
    COUNT(DISTINCT "Invoice") AS orders,
    SUM("Quantity" * "Price") AS revenue,
    SUM("Quantity" * "Price") / COUNT(DISTINCT "Customer ID") AS avg_customer_value
FROM online_retail
WHERE "Customer ID" IS NOT NULL
  AND "Quantity" > 0
GROUP BY "Country"
ORDER BY revenue DESC;
