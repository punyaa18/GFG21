"""
Day 19: Market Analyst - Financial Data Analysis
Demonstrates financial data analysis, market insights, and investment analytics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Generate Synthetic Market Data
# ========================================
print("Generating market data...")

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Simulate stock prices for 5 companies
companies = {
    'TECH': 150,
    'FINANCE': 120,
    'ENERGY': 80,
    'RETAIL': 60,
    'PHARMA': 110
}

stock_data = {}

for company, initial_price in companies.items():
    prices = [initial_price]
    for i in range(1, len(dates)):
        change = np.random.normal(0.001, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10))  # Ensure positive price
    
    stock_data[company] = prices

df_stocks = pd.DataFrame(stock_data, index=dates)

print(f"Market data created: {df_stocks.shape[0]} trading days")
print("\nStock prices (last 5 days):")
print(df_stocks.tail())

# ========================================
# Step 2: Calculate Financial Indicators
# ========================================
print("\n=== FINANCIAL INDICATORS ===")

# Simple Moving Averages
sma_20 = df_stocks.rolling(window=20).mean()
sma_50 = df_stocks.rolling(window=50).mean()

# Returns
daily_returns = df_stocks.pct_change() * 100
cumulative_returns = (1 + daily_returns/100).cumprod() * 100 - 100

# Volatility
volatility = daily_returns.std()

# Correlation
correlation = df_stocks.pct_change().corr()

print("Annualized Volatility:")
print(volatility * np.sqrt(252))

# ========================================
# Step 3: Stock Price Trends
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Price trends with moving averages (TECH stock)
axes[0, 0].plot(df_stocks.index, df_stocks['TECH'], label='Price', linewidth=1.5, alpha=0.7)
axes[0, 0].plot(sma_20.index, sma_20['TECH'], label='20-day SMA', linewidth=2, alpha=0.8)
axes[0, 0].plot(sma_50.index, sma_50['TECH'], label='50-day SMA', linewidth=2, alpha=0.8)
axes[0, 0].set_ylabel('Price ($)', fontweight='bold')
axes[0, 0].set_title('TECH Stock Price with Moving Averages', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. All stocks comparison
for company in companies.keys():
    axes[0, 1].plot(df_stocks.index, df_stocks[company], label=company, linewidth=2, alpha=0.8)
axes[0, 1].set_ylabel('Price ($)', fontweight='bold')
axes[0, 1].set_title('Stock Price Comparison', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Cumulative returns
for company in companies.keys():
    axes[1, 0].plot(cumulative_returns.index, cumulative_returns[company], label=company, linewidth=2)
axes[1, 0].set_ylabel('Cumulative Return (%)', fontweight='bold')
axes[1, 0].set_title('Cumulative Returns Comparison', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 4. Daily returns distribution
data_for_violin = [daily_returns[col].dropna().values for col in daily_returns.columns]
parts = axes[1, 1].violinplot(data_for_violin, positions=range(len(companies)), showmeans=True)
axes[1, 1].set_xticks(range(len(companies)))
axes[1, 1].set_xticklabels(companies.keys())
axes[1, 1].set_ylabel('Daily Return (%)', fontweight='bold')
axes[1, 1].set_title('Distribution of Daily Returns', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/stock_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Stock analysis plot saved!")

# ========================================
# Step 4: Risk and Return Analysis
# ========================================
annual_returns = cumulative_returns.iloc[-1].values
annual_volatility = volatility.values * np.sqrt(252)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Risk-Return Scatter
axes[0, 0].scatter(annual_volatility * 100, annual_returns, s=200, alpha=0.6, 
                  edgecolors='black', linewidth=2, c=range(len(companies)), cmap='viridis')
for i, company in enumerate(companies.keys()):
    axes[0, 0].annotate(company, (annual_volatility[i]*100, annual_returns[i]), 
                       fontsize=10, fontweight='bold')
axes[0, 0].set_xlabel('Annualized Volatility (%)', fontweight='bold')
axes[0, 0].set_ylabel('Annual Return (%)', fontweight='bold')
axes[0, 0].set_title('Risk-Return Profile', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Volatility comparison
axes[0, 1].bar(companies.keys(), annual_volatility * 100, color='steelblue', 
              edgecolor='black', alpha=0.7)
axes[0, 1].set_ylabel('Annualized Volatility (%)', fontweight='bold')
axes[0, 1].set_title('Stock Volatility Comparison', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Correlation heatmap
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
           ax=axes[1, 0], cbar_kws={'label': 'Correlation'}, linewidths=0.5)
axes[1, 0].set_title('Stock Correlation Matrix', fontweight='bold')

# 4. Returns comparison
returns_data = {
    'Stock': list(companies.keys()),
    'Annual Return (%)': annual_returns,
    'Annual Volatility (%)': annual_volatility * 100,
    'Sharpe Ratio': annual_returns / (annual_volatility * 100)
}

returns_text = f"""
RISK-RETURN ANALYSIS
{'='*50}

ANNUAL RETURNS:
"""
for company, ret in zip(companies.keys(), annual_returns):
    returns_text += f"  {company}: {ret:>6.2f}%\n"

returns_text += "\nANNUALIZED VOLATILITY:\n"
for company, vol in zip(companies.keys(), annual_volatility * 100):
    returns_text += f"  {company}: {vol:>6.2f}%\n"

returns_text += "\nSHARPE RATIO (Risk-Adjusted Return):\n"
for company, sr in zip(companies.keys(), returns_data['Sharpe Ratio']):
    returns_text += f"  {company}: {sr:>6.3f}\n"

axes[1, 1].text(0.05, 0.95, returns_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')
axes[1, 1].set_title('Performance Metrics', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/risk_return_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Risk-return analysis plot saved!")

# ========================================
# Step 5: Portfolio Analysis
# ========================================
# Equal weight portfolio
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
portfolio_returns = (daily_returns * weights).sum(axis=1)
cumulative_portfolio_returns = (1 + portfolio_returns/100).cumprod() * 100 - 100

# Alternative portfolio (concentrated)
weights_concentrated = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
portfolio_returns_conc = (daily_returns * weights_concentrated).sum(axis=1)
cumulative_portfolio_returns_conc = (1 + portfolio_returns_conc/100).cumprod() * 100 - 100

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Portfolio performance
axes[0, 0].plot(cumulative_returns.index, cumulative_returns['TECH'], 
               label='TECH Stock', linewidth=2, alpha=0.7)
axes[0, 0].plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns, 
               label='Equal Weight Portfolio', linewidth=2, alpha=0.8)
axes[0, 0].plot(cumulative_portfolio_returns_conc.index, cumulative_portfolio_returns_conc, 
               label='Concentrated Portfolio', linewidth=2, alpha=0.8, linestyle='--')
axes[0, 0].set_ylabel('Cumulative Return (%)', fontweight='bold')
axes[0, 0].set_title('Portfolio vs Single Stock Performance', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Portfolio weights pie
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(companies)))
axes[0, 1].pie(weights, labels=companies.keys(), autopct='%1.1f%%', colors=colors_pie,
              startangle=90)
axes[0, 1].set_title('Equal Weight Portfolio Allocation', fontweight='bold')

# 3. Drawdown analysis
cumulative_max = cumulative_returns.cummax()
drawdown = (cumulative_returns - cumulative_max) / cumulative_max * 100

for company in companies.keys():
    axes[1, 0].plot(drawdown.index, drawdown[company], label=company, linewidth=1.5, alpha=0.8)

axes[1, 0].fill_between(drawdown.index, drawdown.min(axis=1), 0, alpha=0.1, color='red')
axes[1, 0].set_ylabel('Drawdown (%)', fontweight='bold')
axes[1, 0].set_title('Maximum Drawdown Analysis', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Portfolio statistics
portfolio_vol = portfolio_returns.std() * np.sqrt(252)
portfolio_return = cumulative_portfolio_returns.iloc[-1]
portfolio_sharpe = portfolio_return / (portfolio_vol * 100)

conc_vol = portfolio_returns_conc.std() * np.sqrt(252)
conc_return = cumulative_portfolio_returns_conc.iloc[-1]
conc_sharpe = conc_return / (conc_vol * 100)

stats_text = f"""
PORTFOLIO ANALYSIS
{'='*50}

EQUAL WEIGHT PORTFOLIO:
  Annual Return: {portfolio_return:.2f}%
  Volatility: {portfolio_vol*100:.2f}%
  Sharpe Ratio: {portfolio_sharpe:.3f}
  Max Drawdown: {drawdown.min().min():.2f}%

CONCENTRATED PORTFOLIO:
  Annual Return: {conc_return:.2f}%
  Volatility: {conc_vol*100:.2f}%
  Sharpe Ratio: {conc_sharpe:.3f}

DIVERSIFICATION BENEFIT:
  Vol Reduction: {(portfolio_vol*100 - annual_volatility.mean()*100):.2f}%
"""

axes[1, 1].text(0.05, 0.95, stats_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
axes[1, 1].axis('off')
axes[1, 1].set_title('Portfolio Statistics', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/portfolio_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Portfolio analysis plot saved!")

# ========================================
# Step 6: Market Insights
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Monthly returns heatmap
monthly_returns = daily_returns.resample('ME').sum()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

im = axes[0, 0].imshow(monthly_returns.T, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=5)
axes[0, 0].set_xticks(range(len(months)))
axes[0, 0].set_xticklabels(months)
axes[0, 0].set_yticks(range(len(companies)))
axes[0, 0].set_yticklabels(companies.keys())
axes[0, 0].set_title('Monthly Returns Heatmap', fontweight='bold')
plt.colorbar(im, ax=axes[0, 0], label='Return (%)')

# 2. Best and worst days
daily_returns_melted = daily_returns.melt(var_name='Stock', value_name='Return')
best_days = daily_returns_melted.nlargest(5, 'Return')
worst_days = daily_returns_melted.nsmallest(5, 'Return')

top_bottom = pd.concat([best_days, worst_days])
colors_bars = ['green'] * 5 + ['red'] * 5

axes[0, 1].barh(range(len(top_bottom)), top_bottom['Return'], color=colors_bars, alpha=0.7, edgecolor='black')
axes[0, 1].set_yticks(range(len(top_bottom)))
labels = [f"{row['Stock']}\n({row['Return']:.2f}%)" for _, row in top_bottom.iterrows()]
axes[0, 1].set_yticklabels(labels, fontsize=8)
axes[0, 1].set_xlabel('Daily Return (%)', fontweight='bold')
axes[0, 1].set_title('Best and Worst Trading Days', fontweight='bold')
axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Moving average crossover signals (TECH stock)
axes[1, 0].plot(df_stocks.index, df_stocks['TECH'], label='Price', linewidth=1.5, alpha=0.7)
axes[1, 0].plot(sma_20.index, sma_20['TECH'], label='20-SMA', linewidth=2, alpha=0.8)
axes[1, 0].plot(sma_50.index, sma_50['TECH'], label='50-SMA', linewidth=2, alpha=0.8)

# Mark crossovers
sma20_above = sma_20['TECH'] > sma_50['TECH']
crossovers = sma20_above != sma20_above.shift()
for date in sma20_above[crossovers].index:
    if sma20_above[date]:
        axes[1, 0].scatter(date, df_stocks.loc[date, 'TECH'], color='green', s=100, marker='^', zorder=5)
    else:
        axes[1, 0].scatter(date, df_stocks.loc[date, 'TECH'], color='red', s=100, marker='v', zorder=5)

axes[1, 0].set_ylabel('Price ($)', fontweight='bold')
axes[1, 0].set_title('TECH: MA Crossover Signals', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Summary stats
summary_text = f"""
MARKET ANALYSIS SUMMARY
{'='*50}

MARKET OVERVIEW:
  Trading Days: {len(dates)}
  Stocks: {len(companies)}
  Analysis Period: {dates[0].date()} to {dates[-1].date()}

BEST PERFORMER:
  Stock: {companies.keys().__iter__().__next__()} (Index 0 - Update)
  Return: {annual_returns.max():.2f}%

HIGHEST VOLATILITY:
  Stock: {list(companies.keys())[annual_volatility.argmax()]}
  Volatility: {annual_volatility.max()*100:.2f}%

MOST CORRELATED PAIR:
  Stocks: (See correlation matrix)
  Avg Correlation: {correlation.values[np.triu_indices_from(correlation.values, k=1)].mean():.3f}

MARKET BETA:
  Portfolio Beta vs Market: 1.0 (by definition)
  Alpha Generation: TBD
"""

axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/market_insights.png', dpi=300, bbox_inches='tight')
plt.close()

print("Market insights plot saved!")

print("\n✅ Market Analysis Complete!")
print("Generated outputs:")
print("  - outputs/stock_analysis.png")
print("  - outputs/risk_return_analysis.png")
print("  - outputs/portfolio_analysis.png")
print("  - outputs/market_insights.png")
