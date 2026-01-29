"""
Day 16: OCR - Optical Character Recognition
Demonstrates OCR techniques for extracting text from images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Sample Image with Text
# ========================================
print("Creating sample image with text...")

# Create image using PIL
width, height = 800, 600
image_pil = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(image_pil)

# Add various texts
texts = [
    ("Optical Character Recognition", (50, 50), 40),
    ("OCR Technology Demo", (60, 120), 35),
    ("Extract Text from Images", (80, 180), 28),
    ("Using Python and AI", (100, 240), 24),
    ("Date: 2024-01-29", (150, 310), 20),
    ("Accuracy: 95%+", (160, 350), 20),
    ("Languages Supported: 100+", (140, 390), 18),
    ("Applications:", (50, 450), 22),
    ("• Document Scanning", (70, 490), 16),
    ("• Invoice Processing", (70, 520), 16),
    ("• Number Plate Detection", (70, 550), 16),
]

for text, position, font_size in texts:
    try:
        # Try to use a default font
        draw.text(position, text, fill='black', font=None)
    except:
        draw.text(position, text, fill='black')

# Add some visual elements
draw.rectangle([40, 440, 780, 575], outline='black', width=2)
draw.ellipse([700, 50, 780, 130], outline='blue', fill='lightblue', width=2)

# Save and convert
image_pil.save('outputs/text_image.png')
image = cv2.imread('outputs/text_image.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Sample image created!")

# ========================================
# Step 2: Image Preprocessing for OCR
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Original
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Grayscale', fontweight='bold')
axes[0, 1].axis('off')

# Thresholding
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
axes[0, 2].imshow(binary, cmap='gray')
axes[0, 2].set_title('Binary Threshold', fontweight='bold')
axes[0, 2].axis('off')

# Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
axes[1, 0].imshow(blurred, cmap='gray')
axes[1, 0].set_title('Gaussian Blur', fontweight='bold')
axes[1, 0].axis('off')

# Adaptive Threshold
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 11, 2)
axes[1, 1].imshow(adaptive, cmap='gray')
axes[1, 1].set_title('Adaptive Threshold', fontweight='bold')
axes[1, 1].axis('off')

# Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(binary, kernel, iterations=1)
axes[1, 2].imshow(dilated, cmap='gray')
axes[1, 2].set_title('Dilation', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/ocr_preprocessing.png', dpi=300, bbox_inches='tight')
plt.close()

print("OCR preprocessing plot saved!")

# ========================================
# Step 3: Text Region Detection
# ========================================
# Find contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size
min_area = 100
text_regions = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    
    if area > min_area:
        aspect_ratio = float(w) / h if h > 0 else 0
        # Filter by aspect ratio (text regions are typically wider than tall)
        if 0.2 < aspect_ratio < 10:
            text_regions.append((x, y, w, h))

print(f"Found {len(text_regions)} text regions")

# ========================================
# Step 4: Visualize Text Regions
# ========================================
image_regions = image.copy()

for x, y, w, h in text_regions:
    cv2.rectangle(image_regions, (x, y), (x+w, y+h), (0, 255, 0), 2)

image_regions_rgb = cv2.cvtColor(image_regions, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(image_regions_rgb)
axes[1].set_title(f'Detected Text Regions ({len(text_regions)} regions)', fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/text_region_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("Text region detection plot saved!")

# ========================================
# Step 5: Character Segmentation Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Region size distribution
region_areas = [w*h for x, y, w, h in text_regions]
region_widths = [w for x, y, w, h in text_regions]
region_heights = [h for x, y, w, h in text_regions]

axes[0, 0].hist(region_areas, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Region Area (pixels)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Text Region Area Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Width vs Height
axes[0, 1].scatter(region_widths, region_heights, s=100, alpha=0.6, 
                  edgecolors='black', c=region_areas, cmap='viridis')
axes[0, 1].set_xlabel('Width (pixels)', fontweight='bold')
axes[0, 1].set_ylabel('Height (pixels)', fontweight='bold')
axes[0, 1].set_title('Region Width vs Height', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Region aspect ratios
aspect_ratios = [w/h if h > 0 else 0 for w, h in zip(region_widths, region_heights)]
axes[1, 0].hist(aspect_ratios, bins=15, color='green', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)', fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontweight='bold')
axes[1, 0].set_title('Region Aspect Ratio Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Statistics
stats_text = f"""
TEXT REGION STATISTICS
{'='*40}

Total Regions Found: {len(text_regions)}

Area Statistics:
  Mean: {np.mean(region_areas):.0f} pixels²
  Median: {np.median(region_areas):.0f} pixels²
  Std Dev: {np.std(region_areas):.0f}

Width Statistics:
  Mean: {np.mean(region_widths):.0f} pixels
  Min: {min(region_widths)} pixels
  Max: {max(region_widths)} pixels

Height Statistics:
  Mean: {np.mean(region_heights):.0f} pixels
  Min: {min(region_heights)} pixels
  Max: {max(region_heights)} pixels

Aspect Ratio:
  Mean: {np.mean(aspect_ratios):.2f}
  Min: {min(aspect_ratios):.2f}
  Max: {max(aspect_ratios):.2f}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='lightyellow', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Analysis Statistics', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/text_region_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Text region analysis plot saved!")

# ========================================
# Step 6: Edge Detection for Text
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Canny edges
edges_canny = cv2.Canny(gray, 100, 200)
axes[0, 0].imshow(edges_canny, cmap='gray')
axes[0, 0].set_title('Canny Edge Detection', fontweight='bold')
axes[0, 0].axis('off')

# Sobel X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
axes[0, 1].imshow(np.abs(sobelx), cmap='gray')
axes[0, 1].set_title('Sobel X (Vertical Edges)', fontweight='bold')
axes[0, 1].axis('off')

# Sobel Y
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
axes[1, 0].imshow(np.abs(sobely), cmap='gray')
axes[1, 0].set_title('Sobel Y (Horizontal Edges)', fontweight='bold')
axes[1, 0].axis('off')

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
axes[1, 1].imshow(np.abs(laplacian), cmap='gray')
axes[1, 1].set_title('Laplacian Edge Detection', fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/edge_detection_text.png', dpi=300, bbox_inches='tight')
plt.close()

print("Edge detection plot saved!")

# ========================================
# Step 7: OCR Performance Metrics
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Simulated OCR confidence scores
np.random.seed(42)
confidences = np.random.uniform(0.7, 0.99, len(text_regions))
sorted_idx = np.argsort(confidences)[::-1]
sorted_confidences = confidences[sorted_idx]

axes[0, 0].hist(confidences, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Confidence Score', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('OCR Confidence Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Sorted confidences
axes[0, 1].plot(sorted_confidences, 'o-', linewidth=2, markersize=6, color='green')
axes[0, 1].fill_between(range(len(sorted_confidences)), sorted_confidences, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Region Index (sorted)', fontweight='bold')
axes[0, 1].set_ylabel('Confidence Score', fontweight='bold')
axes[0, 1].set_title('OCR Confidence - Sorted', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Confidence by region size
axes[1, 0].scatter(region_areas, confidences, s=100, alpha=0.6, 
                  edgecolors='black', color='coral')
axes[1, 0].set_xlabel('Region Area (pixels)', fontweight='bold')
axes[1, 0].set_ylabel('OCR Confidence', fontweight='bold')
axes[1, 0].set_title('Confidence vs Region Size', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# OCR metrics
metrics_text = f"""
OCR PERFORMANCE METRICS
{'='*40}

Detection Results:
  Regions Found: {len(text_regions)}
  Mean Confidence: {np.mean(confidences):.3f}
  Median Confidence: {np.median(confidences):.3f}
  Std Dev: {np.std(confidences):.3f}
  
Confidence Range:
  Min: {np.min(confidences):.3f}
  Max: {np.max(confidences):.3f}
  
Regions by Confidence:
  High (>0.95): {sum(c > 0.95 for c in confidences)}
  Medium (0.85-0.95): {sum(0.85 <= c <= 0.95 for c in confidences)}
  Low (<0.85): {sum(c < 0.85 for c in confidences)}
  
Processing Stats:
  Image Size: {image.shape[1]}x{image.shape[0]}
  Format: RGB (3 channels)
  Preprocessing: Yes
"""

axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round', 
               facecolor='lightcyan', alpha=0.8))
axes[1, 1].axis('off')
axes[1, 1].set_title('Performance Metrics', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/ocr_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("OCR performance plot saved!")

# ========================================
# Step 8: Comparison of Preprocessing Methods
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

preprocessing_methods = {
    'Binary': binary,
    'Adaptive': adaptive,
    'Blur + Binary': cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1],
    'Dilated': dilated,
    'Eroded': cv2.erode(binary, kernel, iterations=1),
    'Morphed': cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
}

for idx, (method_name, result) in enumerate(preprocessing_methods.items()):
    ax = axes[idx // 3, idx % 3]
    ax.imshow(result, cmap='gray')
    ax.set_title(f'{method_name}', fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('outputs/preprocessing_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("Preprocessing comparison plot saved!")

print("\n✅ OCR Analysis Complete!")
print("Generated outputs:")
print("  - outputs/text_image.png")
print("  - outputs/ocr_preprocessing.png")
print("  - outputs/text_region_detection.png")
print("  - outputs/text_region_analysis.png")
print("  - outputs/edge_detection_text.png")
print("  - outputs/ocr_performance.png")
print("  - outputs/preprocessing_comparison.png")
