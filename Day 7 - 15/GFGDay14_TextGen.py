"""
Day 14: Text Generation and Natural Language Processing
Demonstrates NLP techniques, text generation, and language model concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Sample Corpus for NLP
# ========================================
corpus = """
Natural language processing is a fascinating field. Machine learning powers NLP applications.
Text generation creates coherent sentences. Language models learn from large corpora.
Deep learning transformers revolutionize NLP. BERT and GPT models are game-changers.
Sentiment analysis determines positive or negative emotions. Named entity recognition identifies entities.
Machine translation converts one language to another. Question answering systems provide instant responses.
Natural language understanding enables human-computer interaction. Chatbots use NLP for conversations.
Word embeddings represent words in vector space. Transformers use attention mechanisms.
Recurrent neural networks process sequential data. LSTM networks overcome vanishing gradients.
Text classification categorizes documents. Topic modeling discovers hidden themes.
"""

print("Corpus loaded!")
print(f"Corpus length: {len(corpus)} characters")

# ========================================
# Step 2: Text Preprocessing
# ========================================
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = text.split()
    return tokens

tokens = preprocess_text(corpus)
unique_tokens = set(tokens)

print(f"\nTokens count: {len(tokens)}")
print(f"Unique tokens: {len(unique_tokens)}")
print(f"Vocabulary: {sorted(unique_tokens)[:20]}...")

# ========================================
# Step 3: Tokenization Analysis
# ========================================
token_freq = Counter(tokens)
top_tokens = token_freq.most_common(20)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Token frequency
tokens_list, freq_list = zip(*top_tokens)
axes[0, 0].barh(range(len(tokens_list)), freq_list, color='steelblue', edgecolor='black')
axes[0, 0].set_yticks(range(len(tokens_list)))
axes[0, 0].set_yticklabels(tokens_list)
axes[0, 0].set_xlabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Top 20 Most Frequent Tokens', fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Token length distribution
token_lengths = [len(token) for token in tokens]
axes[0, 1].hist(token_lengths, bins=range(min(token_lengths), max(token_lengths)+2), 
               color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Token Length', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Token Length Distribution', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Vocabulary growth
vocab_growth = []
vocab_size = set()
for token in tokens:
    vocab_size.add(token)
    vocab_growth.append(len(vocab_size))

axes[1, 0].plot(range(len(vocab_growth)), vocab_growth, linewidth=2, color='purple')
axes[1, 0].fill_between(range(len(vocab_growth)), vocab_growth, alpha=0.3, color='purple')
axes[1, 0].set_xlabel('Token Position', fontweight='bold')
axes[1, 0].set_ylabel('Vocabulary Size', fontweight='bold')
axes[1, 0].set_title('Vocabulary Growth', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Cumulative frequency
sorted_freqs = sorted([f for _, f in token_freq.items()], reverse=True)
cumsum_freq = np.cumsum(sorted_freqs)
axes[1, 1].plot(range(len(cumsum_freq)), cumsum_freq, linewidth=2, color='orange', marker='o', markersize=4)
axes[1, 1].set_xlabel('Unique Tokens (ranked)', fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Frequency', fontweight='bold')
axes[1, 1].set_title('Cumulative Token Frequency (Zipf\'s Law)', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/tokenization_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Tokenization analysis plot saved!")

# ========================================
# Step 4: Bigrams and Trigrams
# ========================================
def get_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

bigrams = get_ngrams(tokens, 2)
trigrams = get_ngrams(tokens, 3)

bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bigrams
bigram_data = bigram_freq.most_common(12)
bigram_words, bigram_counts = zip(*bigram_data)
axes[0].barh(range(len(bigram_words)), bigram_counts, color='teal', edgecolor='black')
axes[0].set_yticks(range(len(bigram_words)))
axes[0].set_yticklabels(bigram_words, fontsize=10)
axes[0].set_xlabel('Frequency', fontweight='bold')
axes[0].set_title('Top 12 Bigrams', fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Trigrams
trigram_data = trigram_freq.most_common(10)
trigram_words, trigram_counts = zip(*trigram_data)
axes[1].barh(range(len(trigram_words)), trigram_counts, color='coral', edgecolor='black')
axes[1].set_yticks(range(len(trigram_words)))
axes[1].set_yticklabels(trigram_words, fontsize=9)
axes[1].set_xlabel('Frequency', fontweight='bold')
axes[1].set_title('Top 10 Trigrams', fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/ngrams_text.png', dpi=300, bbox_inches='tight')
plt.close()

print("N-grams plot saved!")

# ========================================
# Step 5: Sentence Analysis
# ========================================
sentences = [s.strip() for s in corpus.split('.') if s.strip()]
sentence_lengths = [len(preprocess_text(s)) for s in sentences]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sentence length distribution
axes[0, 0].hist(sentence_lengths, bins=15, color='navy', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Sentence Length (tokens)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Sentence Length Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Sentence lengths sorted
axes[0, 1].plot(sorted(sentence_lengths), marker='o', linewidth=2, markersize=6, color='green')
axes[0, 1].set_xlabel('Sentence Index (sorted)', fontweight='bold')
axes[0, 1].set_ylabel('Length (tokens)', fontweight='bold')
axes[0, 1].set_title('Sorted Sentence Lengths', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Sentence count by length
length_bins = [0, 10, 15, 20, 30, 100]
length_labels = ['0-10', '10-15', '15-20', '20-30', '30+']
hist, _ = np.histogram(sentence_lengths, bins=length_bins)
axes[1, 0].bar(length_labels, hist, color='purple', edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('Sentence Count by Length Range', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Text statistics
stats_text = f"""
Total Sentences: {len(sentences)}
Mean Sentence Length: {np.mean(sentence_lengths):.1f}
Median Sentence Length: {np.median(sentence_lengths):.1f}
Min Length: {min(sentence_lengths)}
Max Length: {max(sentence_lengths)}

Total Tokens: {len(tokens)}
Unique Tokens: {len(unique_tokens)}
Type-Token Ratio: {len(unique_tokens)/len(tokens):.3f}

Lexical Diversity: {len(unique_tokens)/len(tokens):.2%}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Text Statistics', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/sentence_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Sentence analysis plot saved!")

# ========================================
# Step 6: Text Similarity - Co-occurrence Matrix
# ========================================
target_tokens = [t for t, c in token_freq.most_common(10)]
window_size = 5

cooccurrence = {}
for token in target_tokens:
    cooccurrence[token] = Counter()
    for i, t in enumerate(tokens):
        if t == token:
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            context = tokens[start:end]
            cooccurrence[token].update(context)
    del cooccurrence[token][token]

# Create matrix
matrix = np.zeros((len(target_tokens), len(target_tokens)))
for i, token1 in enumerate(target_tokens):
    for j, token2 in enumerate(target_tokens):
        if token2 in cooccurrence[token1]:
            matrix[i][j] = cooccurrence[token1][token2]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matrix, xticklabels=target_tokens, yticklabels=target_tokens, 
           cmap='YlOrRd', annot=True, fmt='.0f', cbar_kws={'label': 'Co-occurrence'})
ax.set_title('Token Co-occurrence Matrix', fontweight='bold', fontsize=14)
ax.set_xlabel('Token', fontweight='bold')
ax.set_ylabel('Token', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/cooccurrence_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("Co-occurrence matrix plot saved!")

# ========================================
# Step 7: Word Clouds / Bubble Chart
# ========================================
fig, ax = plt.subplots(figsize=(14, 8), facecolor='lightblue')

top_15 = token_freq.most_common(15)
x_pos = np.random.uniform(0.1, 0.9, len(top_15))
y_pos = np.random.uniform(0.1, 0.9, len(top_15))

max_freq = max([freq for _, freq in top_15])
colors_bubble = plt.cm.rainbow(np.linspace(0, 1, len(top_15)))

for (token, freq), x, y, color in zip(top_15, x_pos, y_pos, colors_bubble):
    size = (freq / max_freq) * 0.12
    circle = plt.Circle((x, y), size, color=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, token, ha='center', va='center', fontsize=int(freq*1.5)+8, 
           fontweight='bold', color='white')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Top Tokens Visualization', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('outputs/text_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Text bubble chart plot saved!")

# ========================================
# Step 8: Language Model Concepts
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Token frequency distribution (Zipf's law)
frequencies = sorted([f for _, f in token_freq.items()], reverse=True)
axes[0, 0].loglog(range(1, len(frequencies)+1), frequencies, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Token Rank (log scale)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency (log scale)', fontweight='bold')
axes[0, 0].set_title("Zipf's Law - Token Frequency Distribution", fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, which='both')

# 2. Vocabulary size vs corpus size
vocab_sizes = [len(set(tokens[:i])) for i in range(100, len(tokens), 100)]
corpus_sizes = range(100, len(tokens), 100)
axes[0, 1].plot(corpus_sizes, vocab_sizes, 'go-', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Corpus Size (tokens)', fontweight='bold')
axes[0, 1].set_ylabel('Vocabulary Size', fontweight='bold')
axes[0, 1].set_title('Vocabulary Growth with Corpus Size', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. N-gram diversity
ngram_counts = {
    'Unigrams': len(unique_tokens),
    'Bigrams': len(set(bigrams)),
    'Trigrams': len(set(trigrams))
}
axes[1, 0].bar(ngram_counts.keys(), ngram_counts.values(), color=['red', 'green', 'blue'], 
              edgecolor='black', alpha=0.7)
axes[1, 0].set_ylabel('Unique N-grams', fontweight='bold')
axes[1, 0].set_title('N-gram Diversity', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Entropy and perplexity concepts
entropy_values = []
for i in range(1, len(tokens)):
    probs = [token_freq[tokens[j]]/len(tokens) for j in range(i)]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    entropy_values.append(entropy)

axes[1, 1].plot(entropy_values, color='purple', linewidth=2)
axes[1, 1].fill_between(range(len(entropy_values)), entropy_values, alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Position in Corpus', fontweight='bold')
axes[1, 1].set_ylabel('Entropy (bits)', fontweight='bold')
axes[1, 1].set_title('Entropy Over Corpus', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/nlp_concepts.png', dpi=300, bbox_inches='tight')
plt.close()

print("NLP concepts plot saved!")

print("\n✅ Text Generation and NLP Complete!")
print("Generated outputs:")
print("  - outputs/tokenization_analysis.png")
print("  - outputs/ngrams_text.png")
print("  - outputs/sentence_analysis.png")
print("  - outputs/cooccurrence_matrix.png")
print("  - outputs/text_bubble_chart.png")
print("  - outputs/nlp_concepts.png")
