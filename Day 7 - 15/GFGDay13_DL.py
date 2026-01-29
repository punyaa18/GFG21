"""
Day 13: Deep Learning and Neural Networks
Demonstrates neural network architecture, training, and visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Neural Network Class
# ========================================
class SimpleNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self.initialize_weights()
        self.losses = []
        
    def initialize_weights(self):
        np.random.seed(42)
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, X):
        self.activations = [X]
        self.z_values = []
        
        A = X
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.z_values.append(Z)
            self.activations.append(A)
        
        # Output layer (sigmoid)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self.sigmoid(Z)
        self.z_values.append(Z)
        self.activations.append(A)
        
        return A
    
    def backward_propagation(self, y):
        m = y.shape[0]
        deltas = []
        
        # Output layer
        output_error = self.activations[-1] - y
        delta = output_error * self.sigmoid_derivative(self.activations[-1])
        deltas.append(delta)
        
        # Hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(deltas[0], self.weights[i+1].T) * self.relu_derivative(self.activations[i+1])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            dW = np.dot(self.activations[i].T, deltas[i]) / m
            dB = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * dB
    
    def train(self, X, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                output = self.forward_propagation(X_batch)
                self.backward_propagation(y_batch)
            
            # Calculate loss
            output = self.forward_propagation(X)
            loss = -np.mean(y * np.log(output + 1e-8) + (1-y) * np.log(1-output + 1e-8))
            self.losses.append(loss)
    
    def predict(self, X):
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        return self.forward_propagation(X).flatten()

# ========================================
# Step 2: Create Datasets
# ========================================
print("Creating datasets...")

# Classification dataset
X_class, y_class = make_classification(n_samples=500, n_features=2, n_informative=2, 
                                      n_redundant=0, random_state=42)

# Moons dataset
X_moons, y_moons = make_moons(n_samples=500, noise=0.1, random_state=42)

# Circles dataset
X_circles, y_circles = make_circles(n_samples=500, noise=0.05, factor=0.3, random_state=42)

datasets = {
    'Classification': (X_class, y_class),
    'Moons': (X_moons, y_moons),
    'Circles': (X_circles, y_circles)
}

# ========================================
# Step 3: Visualize Datasets
# ========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (name, (X, y)) in enumerate(datasets.items()):
    scatter = axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, alpha=0.7, edgecolors='black')
    axes[idx].set_title(f'{name} Dataset', fontweight='bold')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/datasets.png', dpi=300, bbox_inches='tight')
plt.close()

print("Datasets visualization saved!")

# ========================================
# Step 4: Train Networks on Different Architectures
# ========================================
architectures = {
    'Shallow (2-2-1)': [2, 2, 1],
    'Medium (2-16-8-1)': [2, 16, 8, 1],
    'Deep (2-32-16-8-1)': [2, 32, 16, 8, 1],
    'Wide (2-64-1)': [2, 64, 1]
}

results = {}

for X, y in datasets.values():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    for arch_name, layer_sizes in architectures.items():
        nn = SimpleNeuralNetwork(layer_sizes, learning_rate=0.1)
        nn.train(X_train, y_train, epochs=200, batch_size=32)
        
        y_pred = nn.predict(X_test).reshape(-1, 1)
        accuracy = accuracy_score(y_test, y_pred)
        
        if arch_name not in results:
            results[arch_name] = []
        results[arch_name].append(accuracy)

# ========================================
# Step 5: Architecture Comparison
# ========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

arch_names = list(architectures.keys())
dataset_names = list(datasets.keys())

# Create data for heatmap
accuracy_matrix = np.zeros((len(arch_names), len(dataset_names)))
for i, (arch, accs) in enumerate(results.items()):
    accuracy_matrix[i] = accs

sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='YlGn', ax=axes[0], 
           xticklabels=dataset_names, yticklabels=arch_names, cbar_kws={'label': 'Accuracy'})
axes[0].set_title('Network Architecture Performance', fontweight='bold')
axes[0].set_ylabel('Architecture', fontweight='bold')
axes[0].set_xlabel('Dataset', fontweight='bold')

# Bar comparison for one dataset
x_pos = np.arange(len(arch_names))
width = 0.25

for i, dataset_name in enumerate(dataset_names):
    accs = [results[arch][i] for arch in arch_names]
    axes[1].bar(x_pos + i*width, accs, width, label=dataset_name, alpha=0.8)

axes[1].set_xlabel('Network Architecture', fontweight='bold')
axes[1].set_ylabel('Accuracy', fontweight='bold')
axes[1].set_title('Performance Across Architectures', fontweight='bold')
axes[1].set_xticks(x_pos + width)
axes[1].set_xticklabels(arch_names, rotation=15, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Architecture comparison plot saved!")

# ========================================
# Step 6: Training Curves
# ========================================
X, y = datasets['Classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, (arch_name, layer_sizes) in enumerate(list(architectures.items())[:4]):
    nn = SimpleNeuralNetwork(layer_sizes, learning_rate=0.1)
    nn.train(X_train, y_train, epochs=300, batch_size=32)
    
    ax = axes[idx // 2, idx % 2]
    ax.plot(nn.losses, linewidth=2, color='steelblue')
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title(f'{arch_name} - Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

print("Training curves plot saved!")

# ========================================
# Step 7: Decision Boundaries Visualization
# ========================================
X, y = datasets['Classification']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, (arch_name, layer_sizes) in enumerate(list(architectures.items())[:4]):
    nn = SimpleNeuralNetwork(layer_sizes, learning_rate=0.1)
    nn.train(X_train, y_train, epochs=200, batch_size=32)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = nn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = axes[idx // 2, idx % 2]
    ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train.flatten(), cmap='RdYlBu', 
              edgecolors='black', s=40, alpha=0.8)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(f'{arch_name} - Decision Boundary', fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('outputs/decision_boundaries.png', dpi=300, bbox_inches='tight')
plt.close()

print("Decision boundaries plot saved!")

# ========================================
# Step 8: Model Complexity vs Performance
# ========================================
param_counts = []
for layer_sizes in architectures.values():
    count = 0
    for i in range(len(layer_sizes) - 1):
        count += (layer_sizes[i] * layer_sizes[i+1]) + layer_sizes[i+1]
    param_counts.append(count)

fig, ax = plt.subplots(figsize=(12, 6))

colors_scatter = plt.cm.viridis(np.linspace(0, 1, len(arch_names)))
for i, dataset_name in enumerate(dataset_names):
    accs = [results[arch][i] for arch in arch_names]
    ax.scatter(param_counts, accs, s=200, alpha=0.7, label=dataset_name, 
              color=colors_scatter[i], edgecolors='black', linewidth=2)

ax.set_xlabel('Number of Parameters', fontweight='bold', fontsize=12)
ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
ax.set_title('Model Complexity vs Performance (Bias-Variance Tradeoff)', fontweight='bold', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/complexity_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Complexity vs performance plot saved!")

print("\n✅ Deep Learning Analysis Complete!")
print("Generated outputs:")
print("  - outputs/datasets.png")
print("  - outputs/architecture_comparison.png")
print("  - outputs/training_curves.png")
print("  - outputs/decision_boundaries.png")
print("  - outputs/complexity_performance.png")
