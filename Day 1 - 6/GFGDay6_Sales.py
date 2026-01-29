"""
Day 6: Sales Analysis and Forecasting
This project analyzes sales data, identifies trends, and creates forecasts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Synthetic Sales Data
# ========================================
np.random.seed(42)
months = pd.date_range(start='2022-01-01', periods=24, freq='ME')
base_sales = 50000
trend = np.linspace(0, 15000, 24)
seasonality = 10000 * np.sin(np.arange(24) * 2 * np.pi / 12)
noise = np.random.normal(0, 2000, 24)

sales = base_sales + trend + seasonality + noise

sales_df = pd.DataFrame({
    'Date': months,
    'Sales': sales,
    'Month': months.month,
    'Year': months.year
})

sales_df['Sales'] = sales_df['Sales'].clip(lower=30000)

print("Sales Data Summary:")
print(sales_df.head(10))
print(f"\nTotal Sales: ${sales_df['Sales'].sum():,.0f}")
print(f"Average Monthly Sales: ${sales_df['Sales'].mean():,.0f}")
print(f"Standard Deviation: ${sales_df['Sales'].std():,.0f}")

# ========================================
# Step 2: Trend Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Time series plot
axes[0, 0].plot(sales_df['Date'], sales_df['Sales'], 'b-o', linewidth=2, markersize=6)
axes[0, 0].set_title('Sales Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Monthly boxplot
sales_df.boxplot(column='Sales', by='Month', ax=axes[0, 1])
axes[0, 1].set_title('Sales Distribution by Month', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Sales ($)')
axes[0, 1].get_figure().suptitle('')

# Year-over-year comparison
sales_by_year_month = sales_df.groupby(['Year', 'Month'])['Sales'].sum().unstack()
sales_by_year_month.T.plot(ax=axes[1, 0], marker='o')
axes[1, 0].set_title('Year-over-Year Sales Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].legend(title='Year')
axes[1, 0].grid(True, alpha=0.3)

# Growth rate
sales_df['Growth_Rate'] = sales_df['Sales'].pct_change() * 100
axes[1, 1].bar(range(len(sales_df)), sales_df['Growth_Rate'], color='green', alpha=0.7)
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Month-over-Month Growth Rate', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Growth Rate (%)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/sales_trend_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nTrend analysis plot saved!")

# ========================================
# Step 3: Linear Regression Forecast
# ========================================
X = np.arange(len(sales_df)).reshape(-1, 1)
y = sales_df['Sales'].values

# Linear model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Polynomial model (degree 2)
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

plt.figure(figsize=(14, 8))
plt.plot(sales_df['Date'], sales_df['Sales'], 'o-', label='Actual Sales', 
        linewidth=2, markersize=8, color='blue')
plt.plot(sales_df['Date'], y_linear_pred, '--', label='Linear Trend', 
        linewidth=2, color='red', alpha=0.7)
plt.plot(sales_df['Date'], y_poly_pred, '-.', label='Polynomial Trend (degree 2)', 
        linewidth=2, color='green', alpha=0.7)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.title('Sales Forecasting - Linear vs Polynomial Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/sales_forecast_models.png', dpi=300, bbox_inches='tight')
plt.close()

print("Forecast models plot saved!")

# ========================================
# Step 4: Future Forecast
# ========================================
future_months = 6
X_future = np.arange(len(sales_df), len(sales_df) + future_months).reshape(-1, 1)
future_dates = pd.date_range(start=sales_df['Date'].iloc[-1], periods=future_months+1, freq='ME')[1:]

y_linear_future = linear_model.predict(X_future)
X_poly_future = poly_features.transform(X_future)
y_poly_future = poly_model.predict(X_poly_future)

plt.figure(figsize=(14, 8))
plt.plot(sales_df['Date'], sales_df['Sales'], 'o-', label='Historical Sales', 
        linewidth=2, markersize=8, color='blue')
plt.plot(future_dates, y_linear_future, 's--', label='Linear Forecast', 
        linewidth=2, markersize=8, color='red', alpha=0.7)
plt.plot(future_dates, y_poly_future, '^-.', label='Polynomial Forecast (degree 2)', 
        linewidth=2, markersize=8, color='green', alpha=0.7)

# Fill forecast region
plt.axvline(x=sales_df['Date'].iloc[-1], color='gray', linestyle=':', linewidth=2, alpha=0.5)
plt.text(sales_df['Date'].iloc[-1], plt.ylim()[1]*0.95, 'Forecast Period →', 
        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.title('Sales Forecast for Next 6 Months', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/sales_future_forecast.png', dpi=300, bbox_inches='tight')
plt.close()

print("Future forecast plot saved!")

# ========================================
# Step 5: Summary Statistics
# ========================================
print("\n=== SALES SUMMARY ===")
print(f"Forecast for next {future_months} months:")
for i, date in enumerate(future_dates):
    linear_forecast = y_linear_future[i]
    poly_forecast = y_poly_future[i]
    print(f"{date.strftime('%Y-%m')}: Linear=${linear_forecast:,.0f}, Polynomial=${poly_forecast:,.0f}")

# ========================================
# Step 6: Category Analysis (Simulated)
# ========================================
categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
category_sales = np.random.dirichlet(np.ones(len(categories))) * sales_df['Sales'].sum()

plt.figure(figsize=(12, 8))
colors = sns.color_palette('husl', len(categories))
wedges, texts, autotexts = plt.pie(category_sales, labels=categories, autopct='%1.1f%%',
                                    colors=colors, startangle=90, textprops={'fontsize': 11})
plt.title('Sales Distribution by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/sales_by_category.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCategory sales plot saved!")

print("\n✅ Sales Analysis Complete!")
print("Generated outputs:")
print("  - outputs/sales_trend_analysis.png")
print("  - outputs/sales_forecast_models.png")
print("  - outputs/sales_future_forecast.png")
print("  - outputs/sales_by_category.png")
