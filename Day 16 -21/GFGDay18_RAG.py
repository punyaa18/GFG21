"""
Day 18: RAG (Retrieval Augmented Generation)
Demonstrates Retrieval Augmented Generation concepts and document processing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Document Database
# ========================================
print("Creating document database for RAG...")

documents = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Natural language processing is crucial for text analysis and understanding.",
    "Computer vision enables machines to interpret visual information from images.",
    "Reinforcement learning trains agents to make sequential decisions.",
    "Supervised learning requires labeled training data for model training.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Transfer learning leverages pre-trained models for new tasks.",
    "Transformer models revolutionized natural language processing with attention mechanisms.",
    "Neural networks are inspired by biological neurons in the human brain.",
    "Data preprocessing is essential for machine learning model performance.",
    "Feature engineering creates meaningful features from raw data.",
    "Model evaluation metrics help assess machine learning performance.",
    "Cross-validation ensures robust model generalization.",
    "Hyperparameter tuning optimizes model performance.",
]

df_docs = pd.DataFrame({'Document_ID': range(len(documents)), 'Text': documents})

print(f"Created document database with {len(documents)} documents")
print("\nSample documents:")
for i, doc in enumerate(documents[:3]):
    print(f"{i+1}. {doc[:60]}...")

# ========================================
# Step 2: Document Vectorization
# ========================================
print("\nVectorizing documents...")

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ========================================
# Step 3: Similarity Analysis
# ========================================
# Calculate document similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap of document similarities
im = axes[0].imshow(similarity_matrix, cmap='YlOrRd', aspect='auto')
axes[0].set_xlabel('Document ID', fontweight='bold')
axes[0].set_ylabel('Document ID', fontweight='bold')
axes[0].set_title('Document Similarity Matrix', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=axes[0], label='Cosine Similarity')

# Most similar document pairs
similarities_list = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarities_list.append({
            'Doc1': i,
            'Doc2': j,
            'Similarity': similarity_matrix[i][j]
        })

similarities_df = pd.DataFrame(similarities_list).sort_values('Similarity', ascending=False)

top_similar = similarities_df.head(8)
labels = [f"Doc{row['Doc1']}-Doc{row['Doc2']}" for _, row in top_similar.iterrows()]
values = top_similar['Similarity'].values

axes[1].barh(range(len(labels)), values, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_yticks(range(len(labels)))
axes[1].set_yticklabels(labels)
axes[1].set_xlabel('Cosine Similarity', fontweight='bold')
axes[1].set_title('Top 8 Most Similar Document Pairs', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/document_similarity.png', dpi=300, bbox_inches='tight')
plt.close()

print("Document similarity plot saved!")

# ========================================
# Step 4: Query Processing and Retrieval
# ========================================
queries = [
    "What is deep learning?",
    "How does machine learning work?",
    "Explain neural networks",
    "What is transfer learning?",
    "Tell me about natural language processing"
]

print("\n=== QUERY RETRIEVAL RESULTS ===")

retrieval_results = []

for query_idx, query in enumerate(queries):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:3]  # Get top 3
    
    print(f"\nQuery {query_idx + 1}: \"{query}\"")
    
    for rank, doc_idx in enumerate(top_indices, 1):
        sim_score = similarities[doc_idx]
        print(f"  {rank}. [Similarity: {sim_score:.3f}] {documents[doc_idx][:70]}...")
        retrieval_results.append({
            'Query': query_idx,
            'Document': doc_idx,
            'Similarity': sim_score,
            'Rank': rank
        })

df_retrieval = pd.DataFrame(retrieval_results)

# ========================================
# Step 5: Retrieval Performance Visualization
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Similarity scores distribution
axes[0, 0].hist(df_retrieval['Similarity'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Similarity Score', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Distribution of Retrieved Document Similarities', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Average similarity by query
query_avg = df_retrieval.groupby('Query')['Similarity'].mean()
axes[0, 1].bar(range(len(queries)), query_avg.values, color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_xticks(range(len(queries)))
axes[0, 1].set_xticklabels([f'Q{i+1}' for i in range(len(queries))])
axes[0, 1].set_ylabel('Average Similarity', fontweight='bold')
axes[0, 1].set_title('Average Retrieval Quality by Query', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Similarity by rank
rank_data = df_retrieval.groupby('Rank')['Similarity'].agg(['mean', 'std'])
axes[1, 0].bar(rank_data.index, rank_data['mean'], yerr=rank_data['std'], 
              color='coral', edgecolor='black', alpha=0.7, capsize=5)
axes[1, 0].set_xlabel('Retrieval Rank', fontweight='bold')
axes[1, 0].set_ylabel('Average Similarity', fontweight='bold')
axes[1, 0].set_title('Similarity Degradation by Rank', fontweight='bold')
axes[1, 0].set_xticks([1, 2, 3])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Query performance heatmap
heatmap_data = df_retrieval.pivot_table(values='Similarity', index='Query', columns='Rank')
im = axes[1, 1].imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto')
axes[1, 1].set_xticks(range(len(heatmap_data.columns)))
axes[1, 1].set_xticklabels(heatmap_data.columns)
axes[1, 1].set_yticks(range(len(heatmap_data.index)))
axes[1, 1].set_yticklabels([f'Q{i+1}' for i in heatmap_data.index])
axes[1, 1].set_xlabel('Rank', fontweight='bold')
axes[1, 1].set_ylabel('Query', fontweight='bold')
axes[1, 1].set_title('Retrieval Similarity Heatmap', fontweight='bold')
plt.colorbar(im, ax=axes[1, 1], label='Similarity')

plt.tight_layout()
plt.savefig('outputs/retrieval_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nRetrieval performance plot saved!")

# ========================================
# Step 6: TF-IDF Feature Analysis
# ========================================
feature_names = vectorizer.get_feature_names_out()

# Get top features for a few documents
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

doc_indices = [0, 3, 7, 12]

for idx, doc_id in enumerate(doc_indices):
    ax = axes[idx // 2, idx % 2]
    
    # Get TF-IDF scores for this document
    doc_tfidf = tfidf_matrix[doc_id].toarray().flatten()
    top_indices = np.argsort(doc_tfidf)[::-1][:10]
    
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [doc_tfidf[i] for i in top_indices]
    
    ax.barh(range(len(top_features)), top_scores, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('TF-IDF Score', fontweight='bold')
    ax.set_title(f'Document {doc_id} - Top 10 Features', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/tfidf_features.png', dpi=300, bbox_inches='tight')
plt.close()

print("TF-IDF features plot saved!")

# ========================================
# Step 7: Document Clustering Analysis
# ========================================
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(tfidf_matrix)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cluster distribution
cluster_counts = Counter(clusters)
axes[0].bar(cluster_counts.keys(), cluster_counts.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
           edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Cluster ID', fontweight='bold')
axes[0].set_ylabel('Number of Documents', fontweight='bold')
axes[0].set_title('Document Distribution Across Clusters', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Cluster details
cluster_text = "DOCUMENT CLUSTERING RESULTS\n" + "="*50 + "\n\n"
for cluster_id in sorted(cluster_counts.keys()):
    docs_in_cluster = [i for i, c in enumerate(clusters) if c == cluster_id]
    cluster_text += f"Cluster {cluster_id} ({len(docs_in_cluster)} docs):\n"
    for doc_id in docs_in_cluster[:3]:
        cluster_text += f"  - Doc {doc_id}: {documents[doc_id][:50]}...\n"
    cluster_text += "\n"

axes[1].text(0.05, 0.95, cluster_text, fontsize=10, family='monospace',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/document_clustering.png', dpi=300, bbox_inches='tight')
plt.close()

print("Document clustering plot saved!")

# ========================================
# Step 8: RAG Pipeline Metrics
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Retrieval statistics
num_queries = len(queries)
num_docs = len(documents)
avg_similarity = df_retrieval['Similarity'].mean()
top1_avg = df_retrieval[df_retrieval['Rank'] == 1]['Similarity'].mean()

metrics_text = f"""
RAG PIPELINE METRICS
{'='*40}

DOCUMENT DATABASE:
  Total Documents: {num_docs}
  Avg Document Length: {np.mean([len(d.split()) for d in documents]):.1f} words
  Total Words: {sum([len(d.split()) for d in documents])}

QUERIES:
  Total Queries: {num_queries}
  Retrieved Documents per Query: 3
  Total Retrievals: {len(df_retrieval)}

RETRIEVAL QUALITY:
  Avg Similarity (All): {avg_similarity:.4f}
  Top-1 Avg Similarity: {top1_avg:.4f}
  Top-1 Retrieval Rate: {len(df_retrieval[df_retrieval['Rank']==1 and df_retrieval['Similarity']>0.3])}/{num_queries}

CLUSTERING:
  Number of Clusters: 3
  Cluster Sizes: {list(cluster_counts.values())}
"""

axes[0, 0].text(0.05, 0.95, metrics_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
axes[0, 0].axis('off')
axes[0, 0].set_title('RAG System Metrics', fontweight='bold')

# Vocabulary analysis
vocab_size = len(vectorizer.get_feature_names_out())
axes[0, 1].text(0.5, 0.5, f'Vocabulary Size\n{vocab_size}\nwords', 
               fontsize=20, ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[0, 1].axis('off')

# Performance over queries
query_performance = df_retrieval.groupby('Query')['Similarity'].apply(list).values
performance_data = [[s for s in perf] for perf in query_performance]

bp = axes[1, 0].boxplot(performance_data, labels=[f'Q{i+1}' for i in range(len(queries))], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
axes[1, 0].set_ylabel('Similarity Score', fontweight='bold')
axes[1, 0].set_title('Query Performance Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# TF-IDF statistics
axes[1, 1].text(0.5, 0.8, f'Feature Matrix\n{tfidf_matrix.shape[0]}×{tfidf_matrix.shape[1]}\n(sparse)', 
               fontsize=14, ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
axes[1, 1].text(0.5, 0.2, f'Sparsity: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))*100:.1f}%', 
               fontsize=12, ha='center', va='center')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/rag_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("RAG metrics plot saved!")

print("\n✅ RAG Implementation Complete!")
print("Generated outputs:")
print("  - outputs/document_similarity.png")
print("  - outputs/retrieval_performance.png")
print("  - outputs/tfidf_features.png")
print("  - outputs/document_clustering.png")
print("  - outputs/rag_metrics.png")
