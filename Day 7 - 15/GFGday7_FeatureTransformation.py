"""
Day 7: Feature Transformation and Engineering
Demonstrates various feature transformation techniques for improving model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   PolynomialFeatures, PowerTransformer)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Sample Data
# ========================================
np.random.seed(42)
n_samples = 1000

# Create features with different distributions
X = pd.DataFrame({
    'Feature1': np.random.exponential(scale=2, size=n_samples),
    'Feature2': np.random.normal(100, 30, n_samples),
    'Feature3': np.random.uniform(0, 50, n_samples),
    'Feature4': np.random.poisson(5, n_samples),
})

print("Original Data Statistics:")
print(X.describe())

# ========================================
# Step 2: Visualization of Original Features
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(X.columns):
    axes[idx].hist(X[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Original {col} Distribution', fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/original_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nOriginal features distribution saved!")

# ========================================
# Step 3: Scaling Transformations
# ========================================
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

scaler_robust = RobustScaler()
X_robust = scaler_robust.fit_transform(X)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].boxplot([X['Feature1'], X['Feature2'], X['Feature3'], X['Feature4']], 
                    labels=X.columns)
axes[0, 0].set_title('Original Data', fontweight='bold')
axes[0, 0].set_ylabel('Value')
axes[0, 0].grid(True, alpha=0.3)

# StandardScaler
axes[0, 1].boxplot([X_standard[:, 0], X_standard[:, 1], X_standard[:, 2], X_standard[:, 3]], 
                    labels=X.columns)
axes[0, 1].set_title('StandardScaler', fontweight='bold')
axes[0, 1].set_ylabel('Scaled Value')
axes[0, 1].grid(True, alpha=0.3)

# MinMaxScaler
axes[1, 0].boxplot([X_minmax[:, 0], X_minmax[:, 1], X_minmax[:, 2], X_minmax[:, 3]], 
                    labels=X.columns)
axes[1, 0].set_title('MinMaxScaler', fontweight='bold')
axes[1, 0].set_ylabel('Scaled Value')
axes[1, 0].grid(True, alpha=0.3)

# RobustScaler
axes[1, 1].boxplot([X_robust[:, 0], X_robust[:, 1], X_robust[:, 2], X_robust[:, 3]], 
                    labels=X.columns)
axes[1, 1].set_title('RobustScaler', fontweight='bold')
axes[1, 1].set_ylabel('Scaled Value')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/scaling_transformations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Scaling transformations plot saved!")

# ========================================
# Step 4: Power Transformation
# ========================================
power_transformer = PowerTransformer(method='yeo-johnson')
X_power = power_transformer.fit_transform(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(X.columns):
    axes[idx].hist(X_power[:, idx], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Power Transformed {col}', fontweight='bold')
    axes[idx].set_xlabel('Transformed Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/power_transformation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Power transformation plot saved!")

# ========================================
# Step 5: Polynomial Features
# ========================================
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print(f"\nOriginal features: {X.shape[1]}")
print(f"Polynomial features (degree 2): {X_poly.shape[1]}")
print(f"Feature names: {poly_features.get_feature_names_out()}")

# ========================================
# Step 6: PCA Dimensionality Reduction
# ========================================
pca = PCA()
X_pca = pca.fit_transform(X_standard)

# Explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
           pca.explained_variance_ratio_, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Principal Component', fontweight='bold')
axes[0].set_ylabel('Explained Variance Ratio', fontweight='bold')
axes[0].set_title('Scree Plot', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Cumulative variance
axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2, markersize=8)
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[1].set_xlabel('Number of Components', fontweight='bold')
axes[1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("PCA analysis plot saved!")

# ========================================
# Step 7: Feature Correlation Heatmap
# ========================================
X_df = pd.DataFrame(X_standard, columns=X.columns)
correlation_matrix = X_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap (Standardized)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Correlation heatmap saved!")

# ========================================
# Step 8: Before/After Comparison
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Feature1
axes[0, 0].hist(X['Feature1'], bins=30, color='red', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Original Feature1', fontweight='bold')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(X_standard[:, 0], bins=30, color='orange', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Standardized Feature1', fontweight='bold')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(X_power[:, 0], bins=30, color='yellow', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Power Transformed Feature1', fontweight='bold')
axes[0, 2].set_ylabel('Frequency')

# Feature2
axes[1, 0].hist(X['Feature2'], bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Original Feature2', fontweight='bold')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(X_standard[:, 1], bins=30, color='cyan', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Standardized Feature2', fontweight='bold')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].hist(X_power[:, 1], bins=30, color='lightblue', alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Power Transformed Feature2', fontweight='bold')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/transformation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Transformation comparison plot saved!")

print("\n✅ Feature Transformation Complete!")
print("Generated outputs:")
print("  - outputs/original_features_distribution.png")
print("  - outputs/scaling_transformations.png")
print("  - outputs/power_transformation.png")
print("  - outputs/pca_analysis.png")
print("  - outputs/feature_correlation.png")
print("  - outputs/transformation_comparison.png")
