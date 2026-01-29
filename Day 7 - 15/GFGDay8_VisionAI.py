"""
Day 8: Vision AI - Image Processing and Computer Vision Basics
Demonstrates fundamental computer vision techniques using OpenCV.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (14, 10)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Synthetic Image
# ========================================
# Create a blank image
height, width, channels = 400, 600, 3
image = np.ones((height, width, channels), dtype=np.uint8) * 255

# Draw shapes
cv2.circle(image, (100, 100), 50, (0, 255, 0), -1)
cv2.rectangle(image, (200, 50), (350, 150), (255, 0, 0), 3)
cv2.ellipse(image, (450, 100), (70, 40), 0, 0, 360, (0, 0, 255), 2)
cv2.line(image, (50, 300), (550, 300), (0, 0, 0), 2)
cv2.putText(image, 'Vision AI Demo', (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

cv2.imwrite('outputs/synthetic_image.png', image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Synthetic image created!")

# ========================================
# Step 2: Image Processing Techniques
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

# Blur
blurred = cv2.GaussianBlur(image, (15, 15), 0)
blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
axes[0, 2].imshow(blurred_rgb)
axes[0, 2].set_title('Gaussian Blur', fontweight='bold')
axes[0, 2].axis('off')

# Edge Detection (Canny)
edges = cv2.Canny(gray, 100, 200)
axes[1, 0].imshow(edges, cmap='gray')
axes[1, 0].set_title('Canny Edge Detection', fontweight='bold')
axes[1, 0].axis('off')

# Dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=2)
axes[1, 1].imshow(dilated, cmap='gray')
axes[1, 1].set_title('Dilation', fontweight='bold')
axes[1, 1].axis('off')

# Erosion
eroded = cv2.erode(edges, kernel, iterations=2)
axes[1, 2].imshow(eroded, cmap='gray')
axes[1, 2].set_title('Erosion', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/image_processing_techniques.png', dpi=300, bbox_inches='tight')
plt.close()

print("Image processing techniques plot saved!")

# ========================================
# Step 3: Color Space Conversions
# ========================================
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# BGR
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('BGR Color Space', fontweight='bold')
axes[0, 0].axis('off')

# HSV (show H channel)
axes[0, 1].imshow(hsv[:,:,0], cmap='hsv')
axes[0, 1].set_title('HSV - Hue Channel', fontweight='bold')
axes[0, 1].axis('off')

# LAB (show L channel)
axes[1, 0].imshow(lab[:,:,0], cmap='gray')
axes[1, 0].set_title('LAB - Lightness Channel', fontweight='bold')
axes[1, 0].axis('off')

# HSV V channel
axes[1, 1].imshow(hsv[:,:,2], cmap='gray')
axes[1, 1].set_title('HSV - Value Channel', fontweight='bold')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('outputs/color_space_conversions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Color space conversion plot saved!")

# ========================================
# Step 4: Histograms and Equalization
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Original histogram
for i, color in enumerate(['b', 'g', 'r']):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    axes[0, 0].plot(hist, color=color, linewidth=1.5)
axes[0, 0].set_title('Original Histogram (BGR)', fontweight='bold')
axes[0, 0].set_xlabel('Pixel Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Grayscale histogram
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
axes[0, 1].plot(hist_gray, color='black', linewidth=1.5)
axes[0, 1].fill_between(range(256), hist_gray.flatten(), alpha=0.3)
axes[0, 1].set_title('Grayscale Histogram', fontweight='bold')
axes[0, 1].set_xlabel('Pixel Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Histogram equalization
equalized = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
axes[0, 2].plot(hist_eq, color='green', linewidth=1.5)
axes[0, 2].fill_between(range(256), hist_eq.flatten(), alpha=0.3, color='green')
axes[0, 2].set_title('Equalized Histogram', fontweight='bold')
axes[0, 2].set_xlabel('Pixel Value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, alpha=0.3)

# Display images
axes[1, 0].imshow(gray, cmap='gray')
axes[1, 0].set_title('Original Grayscale', fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(equalized, cmap='gray')
axes[1, 1].set_title('Histogram Equalized', fontweight='bold')
axes[1, 1].axis('off')

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(gray)
axes[1, 2].imshow(clahe_result, cmap='gray')
axes[1, 2].set_title('CLAHE', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/histogram_equalization.png', dpi=300, bbox_inches='tight')
plt.close()

print("Histogram equalization plot saved!")

# ========================================
# Step 5: Morphological Operations
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

kernel_sizes = [5, 11, 21]
morph_ops = ['Original', 'Open', 'Close', 'Gradient', 'TopHat', 'BlackHat']

# Original
axes[0, 0].imshow(edges, cmap='gray')
axes[0, 0].set_title('Original Edge Image', fontweight='bold')
axes[0, 0].axis('off')

# Opening
opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
axes[0, 1].imshow(opening, cmap='gray')
axes[0, 1].set_title('Opening', fontweight='bold')
axes[0, 1].axis('off')

# Closing
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
axes[0, 2].imshow(closing, cmap='gray')
axes[0, 2].set_title('Closing', fontweight='bold')
axes[0, 2].axis('off')

# Gradient
gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)
axes[1, 0].imshow(gradient, cmap='gray')
axes[1, 0].set_title('Gradient', fontweight='bold')
axes[1, 0].axis('off')

# Top Hat
tophat = cv2.morphologyEx(edges, cv2.MORPH_TOPHAT, kernel)
axes[1, 1].imshow(tophat, cmap='gray')
axes[1, 1].set_title('TopHat', fontweight='bold')
axes[1, 1].axis('off')

# Black Hat
blackhat = cv2.morphologyEx(edges, cv2.MORPH_BLACKHAT, kernel)
axes[1, 2].imshow(blackhat, cmap='gray')
axes[1, 2].set_title('BlackHat', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/morphological_operations.png', dpi=300, bbox_inches='tight')
plt.close()

print("Morphological operations plot saved!")

# ========================================
# Step 6: Filtering and Enhancement
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Original
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

# Bilateral Filter
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
axes[0, 1].imshow(bilateral_rgb)
axes[0, 1].set_title('Bilateral Filter', fontweight='bold')
axes[0, 1].axis('off')

# Median Filter
median = cv2.medianBlur(image, 11)
median_rgb = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
axes[0, 2].imshow(median_rgb)
axes[0, 2].set_title('Median Filter', fontweight='bold')
axes[0, 2].axis('off')

# Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
axes[1, 0].imshow(laplacian, cmap='gray')
axes[1, 0].set_title('Laplacian Edge Detection', fontweight='bold')
axes[1, 0].axis('off')

# Sobel X
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
axes[1, 1].imshow(np.abs(sobelx), cmap='gray')
axes[1, 1].set_title('Sobel X', fontweight='bold')
axes[1, 1].axis('off')

# Sobel Y
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
axes[1, 2].imshow(np.abs(sobely), cmap='gray')
axes[1, 2].set_title('Sobel Y', fontweight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/filtering_enhancement.png', dpi=300, bbox_inches='tight')
plt.close()

print("Filtering and enhancement plot saved!")

print("\n✅ Vision AI - Image Processing Complete!")
print("Generated outputs:")
print("  - outputs/synthetic_image.png")
print("  - outputs/image_processing_techniques.png")
print("  - outputs/color_space_conversions.png")
print("  - outputs/histogram_equalization.png")
print("  - outputs/morphological_operations.png")
print("  - outputs/filtering_enhancement.png")
