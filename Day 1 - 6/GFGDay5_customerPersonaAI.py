"""
Day 5: Customer Persona Analysis using AI/ML Clustering
This project performs customer segmentation and creates personas using clustering algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Synthetic Customer Data
# ========================================
np.random.seed(42)
n_customers = 500

customer_data = {
    'Age': np.random.normal(45, 15, n_customers).astype(int),
    'Income': np.random.normal(60000, 25000, n_customers).astype(int),
    'Spending': np.random.normal(5000, 2000, n_customers).astype(int),
    'Frequency': np.random.poisson(10, n_customers),
    'Years_Customer': np.random.randint(1, 15, n_customers)
}

df = pd.DataFrame(customer_data)
df['Income'] = df['Income'].clip(lower=20000)
df['Spending'] = df['Spending'].clip(lower=500)

print("Customer Data Summary:")
print(df.describe())

# ========================================
# Step 2: Data Preprocessing
# ========================================
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# ========================================
# Step 3: Determine Optimal Clusters (Elbow Method)
# ========================================
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/elbow_method.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nElbow method plot saved!")

# ========================================
# Step 4: Final Clustering (k=4)
# ========================================
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_scaled)

print(f"\nCustomer Distribution Across Clusters:")
print(df['Cluster'].value_counts().sort_index())

# ========================================
# Step 5: Cluster Visualization with PCA
# ========================================
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], 
                     cmap='viridis', s=100, alpha=0.6, edgecolors='black')
plt.scatter(pca.transform(kmeans_final.cluster_centers_)[:, 0],
           pca.transform(kmeans_final.cluster_centers_)[:, 1],
           c='red', s=300, marker='X', edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
plt.title('Customer Personas - Cluster Visualization', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('outputs/customer_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("Cluster visualization saved!")

# ========================================
# Step 6: Persona Profiles
# ========================================
persona_profiles = df.groupby('Cluster')[['Age', 'Income', 'Spending', 'Frequency', 'Years_Customer']].mean()

print("\n=== CUSTOMER PERSONAS ===")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} - Persona Profile:")
    print(persona_profiles.loc[cluster].round(2))

# ========================================
# Step 7: Heatmap of Persona Characteristics
# ========================================
plt.figure(figsize=(10, 6))
sns.heatmap(persona_profiles, annot=True, fmt='.0f', cmap='YlGnBu', 
           cbar_kws={'label': 'Average Value'}, linewidths=0.5)
plt.title('Customer Persona Characteristics Heatmap', fontsize=14, fontweight='bold')
plt.ylabel('Cluster/Persona', fontsize=12)
plt.savefig('outputs/persona_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nPersona heatmap saved!")

# ========================================
# Step 8: Feature Distribution by Cluster
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

features = ['Age', 'Income', 'Spending', 'Frequency', 'Years_Customer']
for idx, feature in enumerate(features):
    for cluster in range(optimal_k):
        cluster_data = df[df['Cluster'] == cluster][feature]
        axes[idx].hist(cluster_data, alpha=0.5, label=f'Cluster {cluster}', bins=20)
    axes[idx].set_title(f'{feature} Distribution by Cluster', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

axes[-1].remove()
plt.tight_layout()
plt.savefig('outputs/feature_distribution_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature distribution plot saved!")

print("\n✅ Customer Persona Analysis Complete!")
print("Generated outputs:")
print("  - outputs/elbow_method.png")
print("  - outputs/customer_clusters.png")
print("  - outputs/persona_heatmap.png")
print("  - outputs/feature_distribution_clusters.png")
