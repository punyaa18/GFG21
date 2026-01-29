"""
Day 11: ML Pipelines and Orchestration
Demonstrates building and orchestrating machine learning pipelines using scikit-learn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Synthetic Dataset
# ========================================
np.random.seed(42)
n_samples = 500

X = pd.DataFrame({
    'Feature1': np.random.normal(100, 30, n_samples),
    'Feature2': np.random.normal(50, 15, n_samples),
    'Feature3': np.random.uniform(0, 100, n_samples),
    'Feature4': np.random.exponential(20, n_samples),
})

# Target variable with non-linear relationship
y = (0.5 * X['Feature1'] + 0.3 * X['Feature2'] - 0.2 * X['Feature3'] + 
     0.1 * X['Feature4'] + 0.1 * X['Feature1']**2 + np.random.normal(0, 50, n_samples))

print("Dataset Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nFeature Statistics:")
print(X.describe())

# ========================================
# Step 2: Train-Test Split
# ========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ========================================
# Step 3: Simple Linear Regression Pipeline
# ========================================
simple_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

simple_pipeline.fit(X_train, y_train)
y_pred_simple = simple_pipeline.predict(X_test)

simple_r2 = r2_score(y_test, y_pred_simple)
simple_rmse = np.sqrt(mean_squared_error(y_test, y_pred_simple))
simple_mae = mean_absolute_error(y_test, y_pred_simple)

print(f"\n=== Simple Pipeline Results ===")
print(f"R² Score: {simple_r2:.4f}")
print(f"RMSE: {simple_rmse:.4f}")
print(f"MAE: {simple_mae:.4f}")

# ========================================
# Step 4: Polynomial Regression Pipeline
# ========================================
poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge(alpha=1.0))
])

poly_pipeline.fit(X_train, y_train)
y_pred_poly = poly_pipeline.predict(X_test)

poly_r2 = r2_score(y_test, y_pred_poly)
poly_rmse = np.sqrt(mean_squared_error(y_test, y_pred_poly))
poly_mae = mean_absolute_error(y_test, y_pred_poly)

print(f"\n=== Polynomial Pipeline Results ===")
print(f"R² Score: {poly_r2:.4f}")
print(f"RMSE: {poly_rmse:.4f}")
print(f"MAE: {poly_mae:.4f}")

# ========================================
# Step 5: Random Forest Pipeline
# ========================================
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_mae = mean_absolute_error(y_test, y_pred_rf)

print(f"\n=== Random Forest Pipeline Results ===")
print(f"R² Score: {rf_r2:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"MAE: {rf_mae:.4f}")

# ========================================
# Step 6: Gradient Boosting Pipeline
# ========================================
gb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_test)

gb_r2 = r2_score(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_mae = mean_absolute_error(y_test, y_pred_gb)

print(f"\n=== Gradient Boosting Pipeline Results ===")
print(f"R² Score: {gb_r2:.4f}")
print(f"RMSE: {gb_rmse:.4f}")
print(f"MAE: {gb_mae:.4f}")

# ========================================
# Step 7: Model Comparison Visualization
# ========================================
models = ['Linear', 'Polynomial', 'Random Forest', 'Gradient Boosting']
r2_scores = [simple_r2, poly_r2, rf_r2, gb_r2]
rmse_scores = [simple_rmse, poly_rmse, rf_rmse, gb_rmse]
mae_scores = [simple_mae, poly_mae, rf_mae, gb_mae]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# R² Comparison
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
axes[0].bar(models, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Model Comparison - R² Score', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1])
for i, v in enumerate(r2_scores):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# RMSE Comparison
axes[1].bar(models, rmse_scores, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('RMSE', fontweight='bold')
axes[1].set_title('Model Comparison - RMSE', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(rmse_scores):
    axes[1].text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold')

# MAE Comparison
axes[2].bar(models, mae_scores, color=colors, edgecolor='black', linewidth=1.5)
axes[2].set_ylabel('MAE', fontweight='bold')
axes[2].set_title('Model Comparison - MAE', fontweight='bold', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(mae_scores):
    axes[2].text(i, v + 1, f'{v:.2f}', ha='center', fontweight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('outputs/pipeline_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPipeline comparison plot saved!")

# ========================================
# Step 8: Predictions Visualization
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

predictions_list = [
    (y_pred_simple, 'Linear Regression', simple_r2),
    (y_pred_poly, 'Polynomial Regression', poly_r2),
    (y_pred_rf, 'Random Forest', rf_r2),
    (y_pred_gb, 'Gradient Boosting', gb_r2)
]

for idx, (y_pred, model_name, r2) in enumerate(predictions_list):
    ax = axes[idx // 2, idx % 2]
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontweight='bold')
    ax.set_ylabel('Predicted Values', fontweight='bold')
    ax.set_title(f'{model_name}\n(R² = {r2:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Predictions comparison plot saved!")

# ========================================
# Step 9: Cross-Validation Scores
# ========================================
pipelines = [simple_pipeline, poly_pipeline, rf_pipeline, gb_pipeline]
cv_scores = []

for pipeline in pipelines:
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    cv_scores.append(scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Box plot of CV scores
bp = axes[0].boxplot(cv_scores, labels=models, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_ylabel('R² Score', fontweight='bold')
axes[0].set_title('Cross-Validation Scores Distribution', fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim([0, 1])

# Mean CV scores
mean_cv = [np.mean(scores) for scores in cv_scores]
axes[1].bar(models, mean_cv, color=colors, edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Mean R² Score', fontweight='bold')
axes[1].set_title('Mean Cross-Validation Scores', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([0, 1])
for i, v in enumerate(mean_cv):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('outputs/cross_validation_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print("Cross-validation scores plot saved!")

print("\n✅ ML Pipelines Complete!")
print("Generated outputs:")
print("  - outputs/pipeline_comparison.png")
print("  - outputs/predictions_comparison.png")
print("  - outputs/cross_validation_scores.png")
