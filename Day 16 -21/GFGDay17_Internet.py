"""
Day 17: Internet Data Collection and Web Scraping
Demonstrates web scraping, API interactions, and internet data collection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Simulate Web Data Collection
# ========================================
print("Simulating web data collection...")

# Simulate fetched data from multiple sources
np.random.seed(42)

# Social media data
social_data = {
    'Platform': ['Twitter', 'Instagram', 'Facebook', 'LinkedIn', 'TikTok', 'YouTube'],
    'Posts': np.random.randint(1000, 10000, 6),
    'Engagement': np.random.uniform(0.5, 5.0, 6),
    'Users': np.random.randint(100000, 5000000, 6),
}

df_social = pd.DataFrame(social_data)

print("Social Media Data:")
print(df_social)

# News articles data
news_data = {
    'Category': ['Tech', 'Business', 'Science', 'Health', 'Politics', 'Sports'],
    'Articles': np.random.randint(100, 1000, 6),
    'Likes': np.random.randint(10000, 100000, 6),
    'Shares': np.random.randint(1000, 50000, 6),
    'Comments': np.random.randint(500, 20000, 6),
}

df_news = pd.DataFrame(news_data)

print("\nNews Data:")
print(df_news)

# Web traffic data
web_traffic = {
    'Hour': range(24),
    'Visitors': np.sin(np.linspace(0, 2*np.pi, 24)) * 500 + 1000 + np.random.normal(0, 100, 24),
    'Page_Views': np.sin(np.linspace(0.5, 2*np.pi+0.5, 24)) * 800 + 1500 + np.random.normal(0, 150, 24),
    'Bounce_Rate': np.random.uniform(30, 80, 24),
}

df_traffic = pd.DataFrame(web_traffic)
df_traffic['Visitors'] = df_traffic['Visitors'].clip(lower=0).astype(int)
df_traffic['Page_Views'] = df_traffic['Page_Views'].clip(lower=0).astype(int)

print("\nWeb Traffic Data:")
print(df_traffic.head(10))

# ========================================
# Step 2: Social Media Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Posts by platform
df_social_sorted = df_social.sort_values('Posts', ascending=False)
axes[0, 0].bar(df_social_sorted['Platform'], df_social_sorted['Posts'], 
              color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Number of Posts', fontweight='bold')
axes[0, 0].set_title('Posts by Social Media Platform', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Engagement rates
df_social_sorted = df_social.sort_values('Engagement', ascending=False)
axes[0, 1].barh(df_social_sorted['Platform'], df_social_sorted['Engagement'], 
               color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Engagement Rate (%)', fontweight='bold')
axes[0, 1].set_title('Engagement Rate by Platform', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Users scatter
axes[1, 0].scatter(df_social['Posts'], df_social['Users'], s=200, alpha=0.6, 
                  c=df_social['Engagement'], cmap='viridis', edgecolors='black', linewidth=2)
for i, platform in enumerate(df_social['Platform']):
    axes[1, 0].annotate(platform, (df_social['Posts'].iloc[i], df_social['Users'].iloc[i]))
axes[1, 0].set_xlabel('Number of Posts', fontweight='bold')
axes[1, 0].set_ylabel('Number of Users', fontweight='bold')
axes[1, 0].set_title('Posts vs Users (colored by engagement)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Platform pie chart
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(df_social)))
axes[1, 1].pie(df_social['Users'], labels=df_social['Platform'], autopct='%1.1f%%', 
              colors=colors_pie, startangle=90)
axes[1, 1].set_title('User Distribution by Platform', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('outputs/social_media_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nSocial media analysis plot saved!")

# ========================================
# Step 3: News Data Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Articles by category
df_news_sorted = df_news.sort_values('Articles', ascending=False)
colors_cat = plt.cm.tab10(np.linspace(0, 1, len(df_news_sorted)))
axes[0, 0].bar(df_news_sorted['Category'], df_news_sorted['Articles'], 
              color=colors_cat, edgecolor='black', alpha=0.7)
axes[0, 0].set_ylabel('Number of Articles', fontweight='bold')
axes[0, 0].set_title('Articles by Category', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Engagement metrics comparison
x_pos = np.arange(len(df_news))
width = 0.25

axes[0, 1].bar(x_pos - width, df_news['Likes']/1000, width, label='Likes (K)', alpha=0.8)
axes[0, 1].bar(x_pos, df_news['Shares']/1000, width, label='Shares (K)', alpha=0.8)
axes[0, 1].bar(x_pos + width, df_news['Comments']/1000, width, label='Comments (K)', alpha=0.8)
axes[0, 1].set_ylabel('Count (thousands)', fontweight='bold')
axes[0, 1].set_title('Engagement Metrics by Category', fontweight='bold', fontsize=12)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(df_news['Category'])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Total engagement
df_news['Total_Engagement'] = df_news['Likes'] + df_news['Shares'] + df_news['Comments']
df_news_sorted_eng = df_news.sort_values('Total_Engagement', ascending=False)
axes[1, 0].barh(df_news_sorted_eng['Category'], df_news_sorted_eng['Total_Engagement'], 
               color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Total Engagement', fontweight='bold')
axes[1, 0].set_title('Total Engagement by Category', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Engagement composition
engagement_data = df_news.set_index('Category')[['Likes', 'Shares', 'Comments']].T
engagement_data.plot(kind='bar', stacked=False, ax=axes[1, 1], alpha=0.8, color=colors_cat)
axes[1, 1].set_ylabel('Count', fontweight='bold')
axes[1, 1].set_title('Engagement Type Composition', fontweight='bold', fontsize=12)
axes[1, 1].legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/news_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("News analysis plot saved!")

# ========================================
# Step 4: Web Traffic Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Visitors and page views over time
axes[0, 0].plot(df_traffic['Hour'], df_traffic['Visitors'], 'o-', linewidth=2, 
               markersize=6, label='Visitors', color='blue')
ax2 = axes[0, 0].twinx()
ax2.plot(df_traffic['Hour'], df_traffic['Page_Views'], 's-', linewidth=2, 
        markersize=6, label='Page Views', color='red')
axes[0, 0].set_xlabel('Hour of Day', fontweight='bold')
axes[0, 0].set_ylabel('Visitors', fontweight='bold', color='blue')
ax2.set_ylabel('Page Views', fontweight='bold', color='red')
axes[0, 0].set_title('Web Traffic Over 24 Hours', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='upper left')
ax2.legend(loc='upper right')

# Bounce rate
axes[0, 1].fill_between(df_traffic['Hour'], df_traffic['Bounce_Rate'], alpha=0.3, color='orange')
axes[0, 1].plot(df_traffic['Hour'], df_traffic['Bounce_Rate'], 'o-', linewidth=2, 
               markersize=6, color='darkorange')
axes[0, 1].set_xlabel('Hour of Day', fontweight='bold')
axes[0, 1].set_ylabel('Bounce Rate (%)', fontweight='bold')
axes[0, 1].set_title('Bounce Rate by Hour', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Peak hours
peak_hours = df_traffic.nlargest(5, 'Visitors')[['Hour', 'Visitors']]
axes[1, 0].bar(peak_hours['Hour'], peak_hours['Visitors'], color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Hour', fontweight='bold')
axes[1, 0].set_ylabel('Visitors', fontweight='bold')
axes[1, 0].set_title('Top 5 Peak Hours', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Visitors vs Bounce Rate
axes[1, 1].scatter(df_traffic['Visitors'], df_traffic['Bounce_Rate'], s=150, 
                  alpha=0.6, c=df_traffic['Hour'], cmap='twilight', edgecolors='black', linewidth=1)
axes[1, 1].set_xlabel('Visitors', fontweight='bold')
axes[1, 1].set_ylabel('Bounce Rate (%)', fontweight='bold')
axes[1, 1].set_title('Visitors vs Bounce Rate', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Hour of Day', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/web_traffic_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Web traffic analysis plot saved!")

# ========================================
# Step 5: Data Source Comparison
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Create comparison metrics
metrics = {
    'Social Media': {
        'Data Points': len(df_social),
        'Avg Posts': df_social['Posts'].mean(),
        'Avg Engagement': df_social['Engagement'].mean(),
        'Total Users': df_social['Users'].sum(),
    },
    'News': {
        'Data Points': len(df_news),
        'Avg Articles': df_news['Articles'].mean(),
        'Avg Likes': df_news['Likes'].mean(),
        'Total Engagement': df_news['Total_Engagement'].sum(),
    },
    'Web Traffic': {
        'Data Points': len(df_traffic),
        'Avg Visitors': df_traffic['Visitors'].mean(),
        'Avg Page Views': df_traffic['Page_Views'].mean(),
        'Avg Bounce Rate': df_traffic['Bounce_Rate'].mean(),
    }
}

sources = list(metrics.keys())
data_points = [metrics[s]['Data Points'] for s in sources]
colors_source = ['#FF6B6B', '#4ECDC4', '#45B7D1']

axes[0].bar(sources, data_points, color=colors_source, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Number of Data Points', fontweight='bold')
axes[0].set_title('Data Points by Source', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Metrics summary table
summary_text = f"""
DATA COLLECTION SUMMARY
{'='*50}

SOCIAL MEDIA:
  Platforms: {len(df_social)}
  Total Posts: {df_social['Posts'].sum():,}
  Avg Engagement: {df_social['Engagement'].mean():.2f}%
  Total Users: {df_social['Users'].sum():,}

NEWS:
  Categories: {len(df_news)}
  Total Articles: {df_news['Articles'].sum():,}
  Total Likes: {df_news['Likes'].sum():,}
  Total Shares: {df_news['Shares'].sum():,}
  Total Comments: {df_news['Comments'].sum():,}

WEB TRAFFIC:
  Time Period: 24 hours
  Total Visitors: {df_traffic['Visitors'].sum():,}
  Total Page Views: {df_traffic['Page_Views'].sum():,}
  Avg Bounce Rate: {df_traffic['Bounce_Rate'].mean():.1f}%
  Peak Hour: {df_traffic.loc[df_traffic['Visitors'].idxmax(), 'Hour']:.0f}:00
  Peak Visitors: {df_traffic['Visitors'].max():.0f}
"""

axes[1].text(0.05, 0.95, summary_text, fontsize=11, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/data_source_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Data source comparison plot saved!")

# ========================================
# Step 6: Time Series Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Visitors time series with trend
from numpy.polynomial.polynomial import Polynomial
p = Polynomial.fit(df_traffic['Hour'], df_traffic['Visitors'], 3)
trend = p(df_traffic['Hour'])

axes[0, 0].plot(df_traffic['Hour'], df_traffic['Visitors'], 'o-', linewidth=2, 
               markersize=6, label='Actual', color='blue', alpha=0.7)
axes[0, 0].plot(df_traffic['Hour'], trend, '--', linewidth=2, label='Trend', color='red')
axes[0, 0].fill_between(df_traffic['Hour'], df_traffic['Visitors'], trend, alpha=0.2)
axes[0, 0].set_xlabel('Hour', fontweight='bold')
axes[0, 0].set_ylabel('Visitors', fontweight='bold')
axes[0, 0].set_title('Visitors with Trend Line', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Hourly distribution
axes[0, 1].hist(df_traffic['Visitors'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Visitors per Hour', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Distribution of Hourly Visitors', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Correlation analysis
correlation_data = {
    'Visitors vs Page Views': np.corrcoef(df_traffic['Visitors'], df_traffic['Page_Views'])[0, 1],
    'Visitors vs Bounce Rate': np.corrcoef(df_traffic['Visitors'], df_traffic['Bounce_Rate'])[0, 1],
    'Page Views vs Bounce Rate': np.corrcoef(df_traffic['Page_Views'], df_traffic['Bounce_Rate'])[0, 1],
}

corr_metrics = list(correlation_data.keys())
corr_values = list(correlation_data.values())
colors_corr = ['green' if v > 0 else 'red' for v in corr_values]

axes[1, 0].barh(corr_metrics, corr_values, color=colors_corr, edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Correlation Coefficient', fontweight='bold')
axes[1, 0].set_title('Metric Correlations', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Heatmap of metrics
metrics_matrix = np.array([
    df_traffic['Visitors'].values / df_traffic['Visitors'].max(),
    df_traffic['Page_Views'].values / df_traffic['Page_Views'].max(),
    df_traffic['Bounce_Rate'].values / 100,
]).T

im = axes[1, 1].imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
axes[1, 1].set_yticks(range(len(df_traffic)))
axes[1, 1].set_yticklabels([f'Hour {h}' for h in df_traffic['Hour']], fontsize=8)
axes[1, 1].set_xticks([0, 1, 2])
axes[1, 1].set_xticklabels(['Visitors', 'Page Views', 'Bounce Rate'])
axes[1, 1].set_title('Normalized Metrics Heatmap', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=axes[1, 1], label='Normalized Value')

plt.tight_layout()
plt.savefig('outputs/time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Time series analysis plot saved!")

print("\n✅ Internet Data Collection Complete!")
print("Generated outputs:")
print("  - outputs/social_media_analysis.png")
print("  - outputs/news_analysis.png")
print("  - outputs/web_traffic_analysis.png")
print("  - outputs/data_source_comparison.png")
print("  - outputs/time_series_analysis.png")
