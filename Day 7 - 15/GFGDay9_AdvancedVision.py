"""
Day 9: Advanced Vision - Object Detection & Advanced CV Techniques
Demonstrates advanced computer vision techniques and object detection preparation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (14, 10)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Complex Test Image
# ========================================
# Create test image with multiple objects
image = np.ones((500, 700, 3), dtype=np.uint8) * 200

# Draw multiple shapes
cv2.circle(image, (100, 150), 60, (0, 255, 0), -1)
cv2.circle(image, (100, 150), 60, (0, 0, 0), 2)
cv2.rectangle(image, (300, 100), (500, 250), (255, 0, 0), -1)
cv2.rectangle(image, (300, 100), (500, 250), (0, 0, 0), 2)
cv2.ellipse(image, (600, 150), (50, 80), 45, 0, 360, (0, 255, 255), -1)
cv2.ellipse(image, (600, 150), (50, 80), 45, 0, 360, (0, 0, 0), 2)

# Add some texture
for i in range(50):
    x, y = np.random.randint(0, 700), np.random.randint(250, 500)
    cv2.circle(image, (x, y), np.random.randint(5, 20), (np.random.randint(0, 255), 
               np.random.randint(0, 255), np.random.randint(0, 255)), -1)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs/complex_test_image.png', image)

print("Complex test image created!")

# ========================================
# Step 2: Contour Detection and Analysis
# ========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (255, 0, 255), 2)
image_contours_rgb = cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Binary Threshold', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(image_contours_rgb)
axes[2].set_title(f'Detected Contours ({len(contours)} found)', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/contour_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("Contour detection plot saved!")

# ========================================
# Step 3: Contour Properties
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Area distribution
areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
axes[0, 0].hist(areas, bins=20, color='blue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Contour Area Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Area (pixels)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Perimeter distribution
perimeters = [cv2.arcLength(cnt, True) for cnt in contours if cv2.contourArea(cnt) > 100]
axes[0, 1].hist(perimeters, bins=20, color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Contour Perimeter Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Perimeter (pixels)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Circularity (Area / Perimeter^2)
circularities = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0 and area > 100:
        circularity = 4 * np.pi * area / (perimeter ** 2)
        circularities.append(circularity)

axes[1, 0].hist(circularities, bins=15, color='red', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('Circularity Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Circularity')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Aspect Ratio
aspect_ratios = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 0 and cv2.contourArea(cnt) > 100:
        aspect_ratio = float(w) / h if h != 0 else 0
        aspect_ratios.append(aspect_ratio)

axes[1, 1].hist(aspect_ratios, bins=15, color='orange', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Aspect Ratio Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Aspect Ratio (width/height)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/contour_properties.png', dpi=300, bbox_inches='tight')
plt.close()

print("Contour properties plot saved!")

# ========================================
# Step 4: Shape Detection
# ========================================
image_shapes = image.copy()
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

for idx, cnt in enumerate(contours):
    if cv2.contourArea(cnt) < 100:
        continue
    
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    
    if len(approx) == 3:
        shape_name = "Triangle"
    elif len(approx) == 4:
        shape_name = "Rectangle"
    elif len(approx) > 8:
        shape_name = "Circle"
    else:
        shape_name = f"Shape-{len(approx)}"
    
    x, y = approx[0][0]
    cv2.putText(image_shapes, shape_name, tuple(map(int, approx[0][0])), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx % len(colors)], 2)

image_shapes_rgb = cv2.cvtColor(image_shapes, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(image_shapes_rgb)
plt.title('Shape Detection Results', fontweight='bold', fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/shape_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("Shape detection plot saved!")

# ========================================
# Step 5: Bounding Boxes and Rotated Rectangles
# ========================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Original
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

# Bounding boxes
image_bbox = image.copy()
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_bbox, (x, y), (x+w, y+h), (0, 255, 255), 2)

image_bbox_rgb = cv2.cvtColor(image_bbox, cv2.COLOR_BGR2RGB)
axes[1].imshow(image_bbox_rgb)
axes[1].set_title('Bounding Boxes', fontweight='bold')
axes[1].axis('off')

# Rotated rectangles
image_rotated = image.copy()
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image_rotated, [box], 0, (255, 0, 255), 2)

image_rotated_rgb = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2RGB)
axes[2].imshow(image_rotated_rgb)
axes[2].set_title('Rotated Rectangles', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/bounding_boxes.png', dpi=300, bbox_inches='tight')
plt.close()

print("Bounding boxes plot saved!")

# ========================================
# Step 6: Color-based Segmentation
# ========================================
# Convert to HSV for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges
colors_to_segment = {
    'Green': ([35, 40, 40], [85, 255, 255]),
    'Blue': ([100, 40, 40], [130, 255, 255]),
    'Red': ([0, 40, 40], [10, 255, 255]),
    'Yellow': ([20, 40, 40], [35, 255, 255])
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

plot_idx = 1
for color_name, (lower, upper) in colors_to_segment.items():
    lower = np.array(lower)
    upper = np.array(upper)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    row = plot_idx // 3
    col = plot_idx % 3
    axes[row, col].imshow(result_rgb)
    axes[row, col].set_title(f'{color_name} Segmentation', fontweight='bold')
    axes[row, col].axis('off')
    plot_idx += 1

# Remove extra subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('outputs/color_segmentation.png', dpi=300, bbox_inches='tight')
plt.close()

print("Color segmentation plot saved!")

# ========================================
# Step 7: Feature Extraction - Histogram
# ========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# BGR histograms
for i, (color, color_name) in enumerate([(0, 'Blue'), (1, 'Green'), (2, 'Red')]):
    hist = cv2.calcHist([image], [color], None, [256], [0, 256])
    axes[0, i].plot(hist, color=color_name.lower(), linewidth=2)
    axes[0, i].fill_between(range(256), hist.flatten(), alpha=0.3, color=color_name.lower())
    axes[0, i].set_title(f'{color_name} Channel Histogram', fontweight='bold')
    axes[0, i].set_xlabel('Pixel Value')
    axes[0, i].set_ylabel('Frequency')
    axes[0, i].grid(True, alpha=0.3)

# HSV histograms
hsv_names = ['Hue', 'Saturation', 'Value']
for i, (channel, channel_name) in enumerate([(0, 'Hue'), (1, 'Saturation'), (2, 'Value')]):
    hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
    axes[1, i].plot(hist, color='purple', linewidth=2)
    axes[1, i].fill_between(range(256), hist.flatten(), alpha=0.3, color='purple')
    axes[1, i].set_title(f'{channel_name} Channel Histogram (HSV)', fontweight='bold')
    axes[1, i].set_xlabel('Pixel Value')
    axes[1, i].set_ylabel('Frequency')
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/advanced_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

print("Advanced histograms plot saved!")

print("\n✅ Advanced Vision Analysis Complete!")
print("Generated outputs:")
print("  - outputs/complex_test_image.png")
print("  - outputs/contour_detection.png")
print("  - outputs/contour_properties.png")
print("  - outputs/shape_detection.png")
print("  - outputs/bounding_boxes.png")
print("  - outputs/color_segmentation.png")
print("  - outputs/advanced_histograms.png")
