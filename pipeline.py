"""
Data Cleaning & Visualization Project
Generates a messy synthetic sales dataset, cleans it, and produces visual insights + HTML report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

BASE = Path("/mnt/documents/data-cleaning-project")
DATA, CHARTS, REPORT = BASE/"data", BASE/"charts", BASE/"report"
for p in (DATA, CHARTS, REPORT): p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
rng = np.random.default_rng(42)

# ---------- 1. GENERATE MESSY DATASET ----------
N = 2000
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Home", "Books", "Sports"]
dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

df = pd.DataFrame({
    "order_id": [f"ORD-{i:05d}" for i in range(N)],
    "order_date": rng.choice(dates, N),
    "region": rng.choice(regions, N, p=[0.3, 0.25, 0.2, 0.25]),
    "category": rng.choice(categories, N),
    "units": rng.integers(1, 12, N),
    "unit_price": np.round(rng.gamma(2.5, 30, N), 2),
    "customer_age": rng.normal(38, 12, N).round(),
    "discount": np.round(rng.beta(2, 8, N), 2),
})

# Inject mess
miss_idx = rng.choice(N, 180, replace=False)
df.loc[miss_idx[:60], "unit_price"] = np.nan
df.loc[miss_idx[60:120], "customer_age"] = np.nan
df.loc[miss_idx[120:], "region"] = np.nan

out_idx = rng.choice(N, 25, replace=False)
df.loc[out_idx, "unit_price"] *= rng.uniform(15, 40, 25)  # extreme outliers
df.loc[rng.choice(N, 10, replace=False), "customer_age"] = rng.choice([-5, 200, 350], 10)

# Duplicates
dup = df.sample(80, random_state=1)
df = pd.concat([df, dup], ignore_index=True)

# Inconsistent casing in region
df["region"] = df["region"].apply(lambda x: rng.choice([x, str(x).lower(), str(x).upper()]) if pd.notna(x) else x)

raw_path = DATA/"raw_sales.csv"
df.to_csv(raw_path, index=False)
print(f"Raw dataset: {df.shape}  ->  {raw_path}")

# ---------- 2. CLEANING ----------
report = {"raw_rows": len(df)}

clean = df.copy()

# Normalize region casing
clean["region"] = clean["region"].astype(str).str.title().replace("Nan", np.nan)

# Drop duplicates
before = len(clean)
clean = clean.drop_duplicates(subset="order_id", keep="first")
report["duplicates_removed"] = before - len(clean)

# Missing values
report["missing_before"] = clean.isna().sum().to_dict()
clean["unit_price"] = clean["unit_price"].fillna(clean["unit_price"].median())
clean["customer_age"] = clean["customer_age"].fillna(clean["customer_age"].median())
clean["region"] = clean["region"].fillna(clean["region"].mode().iloc[0])

# Fix invalid ages
clean.loc[(clean["customer_age"] < 14) | (clean["customer_age"] > 95), "customer_age"] = clean["customer_age"].median()

# Outliers via IQR on unit_price
q1, q3 = clean["unit_price"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
outliers = ((clean["unit_price"] < lo) | (clean["unit_price"] > hi)).sum()
clean["unit_price"] = clean["unit_price"].clip(lo, hi)
report["outliers_capped"] = int(outliers)
report["iqr_bounds"] = {"low": round(lo, 2), "high": round(hi, 2)}

# Feature engineering
clean["revenue"] = (clean["units"] * clean["unit_price"] * (1 - clean["discount"])).round(2)
clean["order_date"] = pd.to_datetime(clean["order_date"])
clean["month"] = clean["order_date"].dt.to_period("M").astype(str)

report["clean_rows"] = len(clean)
report["missing_after"] = clean.isna().sum().to_dict()
report["total_revenue"] = float(clean["revenue"].sum().round(2))
report["avg_order_value"] = float(clean["revenue"].mean().round(2))

clean_path = DATA/"clean_sales.csv"
clean.to_csv(clean_path, index=False)

# ---------- 3. VISUALIZATIONS ----------
def save(fig, name):
    path = CHARTS/name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path

# 3.1 Before/after unit_price distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["unit_price"].dropna(), bins=60, ax=axes[0], color="#ef4444")
axes[0].set_title("Unit Price â€” Before Cleaning (outliers visible)")
sns.histplot(clean["unit_price"], bins=60, ax=axes[1], color="#10b981")
axes[1].set_title("Unit Price â€” After IQR Capping")
save(fig, "01_price_before_after.png")

# 3.2 Monthly revenue trend
monthly = clean.groupby("month")["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(12, 4.5))
sns.lineplot(data=monthly, x="month", y="revenue", marker="o", ax=ax, color="#6366f1", linewidth=2.4)
ax.set_title("Monthly Revenue Trend")
ax.set_ylabel("Revenue ($)"); ax.set_xlabel("")
plt.xticks(rotation=45, ha="right")
save(fig, "02_monthly_revenue.png")

# 3.3 Revenue by category
cat = clean.groupby("category")["revenue"].sum().sort_values(ascending=True).reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(data=cat, y="category", x="revenue", ax=ax, palette="viridis", hue="category", legend=False)
ax.set_title("Total Revenue by Category"); ax.set_xlabel("Revenue ($)")
save(fig, "03_revenue_category.png")

# 3.4 Revenue by region
reg = clean.groupby("region")["revenue"].sum().reset_index()
fig, ax = plt.subplots(figsize=(7, 5))
ax.pie(reg["revenue"], labels=reg["region"], autopct="%1.1f%%", startangle=90,
       colors=sns.color_palette("Set2"), wedgeprops={"edgecolor": "white", "linewidth": 2})
ax.set_title("Revenue Share by Region")
save(fig, "04_revenue_region.png")

# 3.5 Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5))
corr = clean[["units", "unit_price", "discount", "customer_age", "revenue"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax, linewidths=.5)
ax.set_title("Feature Correlation")
save(fig, "05_correlation.png")

# 3.6 Age vs revenue
fig, ax = plt.subplots(figsize=(10, 4.5))
sns.scatterplot(data=clean.sample(800, random_state=2), x="customer_age", y="revenue",
                hue="category", alpha=0.7, ax=ax)
ax.set_title("Customer Age vs Order Revenue")
save(fig, "06_age_revenue.png")

# ---------- 4. HTML REPORT ----------
top_cat = clean.groupby("category")["revenue"].sum().idxmax()
top_reg = clean.groupby("region")["revenue"].sum().idxmax()
best_month = monthly.loc[monthly["revenue"].idxmax(), "month"]

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Sales Data â€” Cleaning & Insights Report</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#0f172a;background:#f8fafc}}
  h1{{font-size:34px;margin:0 0 6px}} h2{{margin-top:42px;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
  .sub{{color:#64748b;margin-bottom:30px}}
  .kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}}
  .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:18px}}
  .kpi .v{{font-size:24px;font-weight:700;color:#6366f1}} .kpi .l{{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
  img{{width:100%;border-radius:14px;border:1px solid #e2e8f0;background:#fff;margin:10px 0 24px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden}}
  th,td{{padding:10px 14px;border-bottom:1px solid #e2e8f0;text-align:left;font-size:14px}}
  th{{background:#f1f5f9}}
  code{{background:#e2e8f0;padding:2px 6px;border-radius:4px;font-size:13px}}
</style></head><body>
<h1>ðŸ“Š Sales Data â€” Cleaning & Insights</h1>
<div class="sub">Automated pipeline: raw â†’ cleaned â†’ visualized. Generated with Pandas, Matplotlib & Seaborn.</div>

<div class="kpis">
  <div class="kpi"><div class="l">Raw rows</div><div class="v">{report['raw_rows']:,}</div></div>
  <div class="kpi"><div class="l">Clean rows</div><div class="v">{report['clean_rows']:,}</div></div>
  <div class="kpi"><div class="l">Duplicates removed</div><div class="v">{report['duplicates_removed']}</div></div>
  <div class="kpi"><div class="l">Outliers capped</div><div class="v">{report['outliers_capped']}</div></div>
  <div class="kpi"><div class="l">Total revenue</div><div class="v">${report['total_revenue']:,.0f}</div></div>
  <div class="kpi"><div class="l">Avg order value</div><div class="v">${report['avg_order_value']:,.2f}</div></div>
</div>

<h2>1. Cleaning Steps Applied</h2>
<ol>
  <li>Normalized inconsistent text casing in <code>region</code> (e.g. "north", "NORTH" â†’ "North").</li>
  <li>Removed <b>{report['duplicates_removed']}</b> duplicate orders by <code>order_id</code>.</li>
  <li>Filled missing <code>unit_price</code> and <code>customer_age</code> with column median; <code>region</code> with mode.</li>
  <li>Replaced impossible ages (&lt;14 or &gt;95) with the median age.</li>
  <li>Capped <b>{report['outliers_capped']}</b> price outliers using the IQR rule
      (bounds ${report['iqr_bounds']['low']} â€“ ${report['iqr_bounds']['high']}).</li>
  <li>Engineered <code>revenue = units Ã— unit_price Ã— (1 âˆ’ discount)</code>.</li>
</ol>

<h2>2. Missing Values â€” Before vs After</h2>
<table><tr><th>Column</th><th>Missing (raw)</th><th>Missing (clean)</th></tr>
{''.join(f"<tr><td>{c}</td><td>{report['missing_before'].get(c,0)}</td><td>{report['missing_after'].get(c,0)}</td></tr>" for c in report['missing_before'])}
</table>

<h2>3. Outlier Treatment â€” Unit Price</h2>
<img src="../charts/01_price_before_after.png" alt="Price before vs after">

<h2>4. Monthly Revenue Trend</h2>
<img src="../charts/02_monthly_revenue.png" alt="Monthly revenue">

<h2>5. Revenue by Category</h2>
<img src="../charts/03_revenue_category.png" alt="Revenue by category">

<h2>6. Revenue Share by Region</h2>
<img src="../charts/04_revenue_region.png" alt="Revenue by region">

<h2>7. Feature Correlation</h2>
<img src="../charts/05_correlation.png" alt="Correlation heatmap">

<h2>8. Customer Age vs Revenue</h2>
<img src="../charts/06_age_revenue.png" alt="Age vs revenue">

<h2>ðŸ”‘ Key Findings</h2>
<ul>
  <li><b>{top_cat}</b> is the top-selling category by total revenue.</li>
  <li><b>{top_reg}</b> region contributes the largest revenue share.</li>
  <li>Best-performing month: <b>{best_month}</b>.</li>
  <li>Cleaning recovered the dataset from {report['raw_rows']:,} noisy rows to {report['clean_rows']:,} analysis-ready rows.</li>
  <li>Discount and revenue show the expected negative correlation; age is weakly correlated with spend.</li>
</ul>
</body></html>"""

(REPORT/"report.html").write_text(html)
(REPORT/"summary.json").write_text(json.dumps(report, indent=2, default=str))
print("Done. Report at:", REPORT/"report.html")