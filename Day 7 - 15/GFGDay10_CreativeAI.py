"""
Day 10: Creative AI - Text Generation and Artistic AI
Demonstrates creative AI applications including text generation and artistic generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import random
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Sample Text Data for Analysis
# ========================================
sample_text = """
Machine learning is transforming the world. Artificial intelligence and deep learning 
are becoming essential skills. Data science helps us understand complex patterns in data. 
Creative AI generates art, music, and text. Neural networks learn from examples. 
Text generation models create human-like responses. Creative applications of AI inspire innovation.
Machine learning algorithms process information. AI solves real-world problems. Deep learning 
networks have multiple layers. Creativity meets technology in modern applications. 
Data-driven decisions improve outcomes. AI systems become smarter with more training data.
Artificial neural networks mimic brain functions. Creative AI pushes technological boundaries.
"""

print("Sample Text for Creative Analysis:")
print(sample_text[:200] + "...")

# ========================================
# Step 2: Text Analysis and Visualization
# ========================================
# Clean and tokenize
words = sample_text.lower().replace('\n', ' ').split()
words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:')]

# Word frequency
word_freq = Counter(words)
top_words = word_freq.most_common(15)

# Plotting word frequency
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Bar chart
words_list, freq_list = zip(*top_words)
axes[0, 0].barh(range(len(words_list)), freq_list, color='steelblue', edgecolor='black')
axes[0, 0].set_yticks(range(len(words_list)))
axes[0, 0].set_yticklabels(words_list)
axes[0, 0].set_xlabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Top 15 Most Frequent Words', fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Cumulative frequency
sorted_freq = sorted(freq_list)
cumsum = np.cumsum(sorted_freq)
axes[0, 1].plot(range(len(sorted_freq)), cumsum, 'o-', linewidth=2, markersize=6, color='green')
axes[0, 1].fill_between(range(len(sorted_freq)), cumsum, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Word Rank', fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Frequency', fontweight='bold')
axes[0, 1].set_title('Cumulative Word Frequency', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Word length distribution
word_lengths = [len(w) for w in words]
axes[1, 0].hist(word_lengths, bins=range(min(word_lengths), max(word_lengths)+2), 
               color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Word Length (characters)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Word Length Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Pie chart of top words
top_10_words, top_10_freq = zip(*top_words[:10])
axes[1, 1].pie(top_10_freq, labels=top_10_words, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Top 10 Words - Proportion', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/text_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Text analysis plot saved!")

# ========================================
# Step 3: N-gram Analysis
# ========================================
def get_ngrams(text, n):
    words = text.lower().replace('\n', ' ').split()
    words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:')]
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(ngrams)

bigrams = get_ngrams(sample_text, 2)
trigrams = get_ngrams(sample_text, 3)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bigrams
bigram_data = bigrams.most_common(10)
bigram_words, bigram_freq = zip(*bigram_data)
axes[0].barh(range(len(bigram_words)), bigram_freq, color='teal', edgecolor='black')
axes[0].set_yticks(range(len(bigram_words)))
axes[0].set_yticklabels(bigram_words, fontsize=10)
axes[0].set_xlabel('Frequency', fontweight='bold')
axes[0].set_title('Top 10 Bigrams (2-word sequences)', fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Trigrams
trigram_data = trigrams.most_common(8)
trigram_words, trigram_freq = zip(*trigram_data)
axes[1].barh(range(len(trigram_words)), trigram_freq, color='coral', edgecolor='black')
axes[1].set_yticks(range(len(trigram_words)))
axes[1].set_yticklabels(trigram_words, fontsize=10)
axes[1].set_xlabel('Frequency', fontweight='bold')
axes[1].set_title('Top 8 Trigrams (3-word sequences)', fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/ngram_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("N-gram analysis plot saved!")

# ========================================
# Step 4: Markov Chain Text Generation
# ========================================
def build_markov_chain(text, order=2):
    words = text.lower().replace('\n', ' ').split()
    words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:')]
    
    chains = {}
    for i in range(len(words) - order):
        key = tuple(words[i:i+order])
        next_word = words[i+order]
        if key not in chains:
            chains[key] = []
        chains[key].append(next_word)
    
    return chains, words

def generate_text(chains, start_key, length=20):
    generated = list(start_key)
    for _ in range(length):
        key = tuple(generated[-2:])
        if key in chains:
            next_word = random.choice(chains[key])
            generated.append(next_word)
        else:
            break
    return ' '.join(generated)

chains, original_words = build_markov_chain(sample_text, order=2)

# Generate multiple samples
generated_samples = []
for i in range(5):
    start_key = random.choice(list(chains.keys()))
    text = generate_text(chains, start_key, length=10)
    generated_samples.append(text)

print("\n=== GENERATED TEXT SAMPLES ===")
for i, sample in enumerate(generated_samples, 1):
    print(f"{i}. {sample}")

# ========================================
# Step 5: Character-level Analysis
# ========================================
char_freq = Counter(sample_text.lower())
vowels = {'a', 'e', 'i', 'o', 'u'}
consonants = {c for c in char_freq.keys() if c.isalpha() and c not in vowels}

vowel_count = sum(char_freq[v] for v in vowels)
consonant_count = sum(char_freq[c] for c in consonants)
other_count = len(sample_text) - vowel_count - consonant_count

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vowel vs Consonant
categories = ['Vowels', 'Consonants', 'Others']
counts = [vowel_count, consonant_count, other_count]
colors = ['#FF9999', '#66B2FF', '#99FF99']

axes[0].pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('Character Type Distribution', fontweight='bold', fontsize=12)

# Top characters
top_chars = char_freq.most_common(15)
chars, char_counts = zip(*top_chars)
chars_display = [c if c != ' ' else 'SPACE' for c in chars]

axes[1].barh(range(len(chars_display)), char_counts, color='mediumpurple', edgecolor='black')
axes[1].set_yticks(range(len(chars_display)))
axes[1].set_yticklabels(chars_display)
axes[1].set_xlabel('Frequency', fontweight='bold')
axes[1].set_title('Top 15 Most Frequent Characters', fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/character_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCharacter analysis plot saved!")

# ========================================
# Step 6: Artistic Visualization - Word Cloud Style
# ========================================
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

fig, ax = plt.subplots(figsize=(14, 8), facecolor='lightgray')

# Create bubbles for top words
top_20 = word_freq.most_common(20)
x_pos = np.random.uniform(0.1, 0.9, len(top_20))
y_pos = np.random.uniform(0.1, 0.9, len(top_20))

max_freq = max([freq for _, freq in top_20])
colors_bubble = plt.cm.viridis(np.linspace(0, 1, len(top_20)))

for (word, freq), x, y, color in zip(top_20, x_pos, y_pos, colors_bubble):
    size = (freq / max_freq) * 0.15
    circle = Circle((x, y), size, color=color, alpha=0.6, edgecolor='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x, y, word, ha='center', va='center', fontsize=int(freq*2)+8, 
           fontweight='bold', color='white')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Word Bubble Visualization', fontweight='bold', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('outputs/word_bubbles.png', dpi=300, bbox_inches='tight')
plt.close()

print("Word bubbles plot saved!")

print("\n✅ Creative AI Complete!")
print("Generated outputs:")
print("  - outputs/text_analysis.png")
print("  - outputs/ngram_analysis.png")
print("  - outputs/character_analysis.png")
print("  - outputs/word_bubbles.png")
