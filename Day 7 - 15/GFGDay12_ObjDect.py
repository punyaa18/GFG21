"""
Day 12: Object Detection
Demonstrates object detection concepts and implementation using YOLO-like techniques.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (14, 10)
import os
os.makedirs('outputs', exist_ok=True)

# ========================================
# Step 1: Create Synthetic Image with Objects
# ========================================
image = np.ones((600, 800, 3), dtype=np.uint8) * 220

# Draw various objects
objects = [
    {'center': (100, 100), 'size': 60, 'shape': 'circle', 'color': (0, 255, 0)},
    {'center': (300, 150), 'size': 70, 'shape': 'rect', 'color': (255, 0, 0)},
    {'center': (500, 120), 'size': 50, 'shape': 'circle', 'color': (0, 0, 255)},
    {'center': (700, 150), 'size': 80, 'shape': 'rect', 'color': (255, 255, 0)},
    {'center': (150, 350), 'size': 65, 'shape': 'circle', 'color': (255, 0, 255)},
    {'center': (400, 400), 'size': 90, 'shape': 'rect', 'color': (0, 255, 255)},
    {'center': (650, 380), 'size': 55, 'shape': 'circle', 'color': (128, 0, 128)},
    {'center': (300, 550), 'size': 75, 'shape': 'rect', 'color': (255, 128, 0)},
]

for obj in objects:
    if obj['shape'] == 'circle':
        cv2.circle(image, obj['center'], obj['size'], obj['color'], -1)
        cv2.circle(image, obj['center'], obj['size'], (0, 0, 0), 2)
    else:
        x, y = obj['center']
        s = obj['size']
        cv2.rectangle(image, (x-s, y-s), (x+s, y+s), obj['color'], -1)
        cv2.rectangle(image, (x-s, y-s), (x+s, y+s), (0, 0, 0), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite('outputs/objects_image.png', image)

print("Synthetic image with objects created!")
print(f"Number of objects: {len(objects)}")

# ========================================
# Step 2: Preprocessing for Detection
# ========================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Binary Threshold', fontweight='bold')
axes[1].axis('off')

axes[2].imshow(binary_cleaned, cmap='gray')
axes[2].set_title('Morphological Operations', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('outputs/preprocessing_detection.png', dpi=300, bbox_inches='tight')
plt.close()

print("Preprocessing plot saved!")

# ========================================
# Step 3: Object Detection via Contours
# ========================================
contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

detections = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 500 < area < 50000:  # Filter by area
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'center': (x+w//2, y+h//2)})

print(f"\nDetected {len(detections)} objects via contours")

# ========================================
# Step 4: Visualization of Detections with Bounding Boxes
# ========================================
image_detected = image.copy()

for i, det in enumerate(detections):
    x, y, w, h = det['x'], det['y'], det['w'], det['h']
    cv2.rectangle(image_detected, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image_detected, f'Obj {i+1}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

image_detected_rgb = cv2.cvtColor(image_detected, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(image_rgb)
axes[0].set_title('Original Image', fontweight='bold')
axes[0].axis('off')

axes[1].imshow(image_detected_rgb)
axes[1].set_title(f'Detected Objects ({len(detections)} detections)', fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('outputs/object_detection_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("Object detection results plot saved!")

# ========================================
# Step 5: Object Properties Analysis
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Area distribution
areas = [d['area'] for d in detections]
axes[0, 0].hist(areas, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Object Area (pixels)', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Object Area Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Width and Height
widths = [d['w'] for d in detections]
heights = [d['h'] for d in detections]

axes[0, 1].scatter(widths, heights, s=100, alpha=0.6, edgecolors='black', c=areas, cmap='viridis')
axes[0, 1].set_xlabel('Width (pixels)', fontweight='bold')
axes[0, 1].set_ylabel('Height (pixels)', fontweight='bold')
axes[0, 1].set_title('Object Width vs Height', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Aspect ratio
aspect_ratios = [d['w']/d['h'] if d['h'] > 0 else 0 for d in detections]
axes[1, 0].bar(range(len(detections)), aspect_ratios, color='coral', edgecolor='black', alpha=0.7)
axes[1, 0].axhline(y=1, color='red', linestyle='--', linewidth=2, label='Square (AR=1)')
axes[1, 0].set_xlabel('Object Index', fontweight='bold')
axes[1, 0].set_ylabel('Aspect Ratio (Width/Height)', fontweight='bold')
axes[1, 0].set_title('Object Aspect Ratio', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Center positions
centers = [d['center'] for d in detections]
centers_x = [c[0] for c in centers]
centers_y = [c[1] for c in centers]

axes[1, 1].scatter(centers_x, centers_y, s=150, alpha=0.6, edgecolors='black', c=range(len(detections)), cmap='tab20')
axes[1, 1].set_xlim(0, image.shape[1])
axes[1, 1].set_ylim(image.shape[0], 0)  # Invert y-axis
axes[1, 1].set_xlabel('X Coordinate (pixels)', fontweight='bold')
axes[1, 1].set_ylabel('Y Coordinate (pixels)', fontweight='bold')
axes[1, 1].set_title('Object Centers Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/object_properties.png', dpi=300, bbox_inches='tight')
plt.close()

print("Object properties plot saved!")

# ========================================
# Step 6: Confidence and NMS Simulation
# ========================================
# Simulate confidence scores (higher for larger, more central objects)
for det in detections:
    # Simulate confidence based on size and position
    center_dist = np.sqrt((det['center'][0] - image.shape[1]/2)**2 + 
                         (det['center'][1] - image.shape[0]/2)**2)
    det['confidence'] = (det['area'] / max(areas)) * (1 - 0.1 * center_dist / 500)

detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)

print("\nTop 5 Detections by Confidence:")
for i, det in enumerate(detections_sorted[:5], 1):
    print(f"{i}. Confidence: {det['confidence']:.4f}, Area: {det['area']:.0f}")

# ========================================
# Step 7: Detection Visualization with Confidence
# ========================================
image_confidence = image.copy()

for i, det in enumerate(detections_sorted):
    x, y, w, h = det['x'], det['y'], det['w'], det['h']
    conf = det['confidence']
    
    # Color based on confidence
    color = (0, int(255*conf), int(255*(1-conf)))
    
    cv2.rectangle(image_confidence, (x, y), (x+w, y+h), color, 2)
    label = f"Conf: {conf:.2f}"
    cv2.putText(image_confidence, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

image_confidence_rgb = cv2.cvtColor(image_confidence, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image_confidence_rgb)
ax.set_title('Detections with Confidence Scores', fontweight='bold', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig('outputs/detection_confidence.png', dpi=300, bbox_inches='tight')
plt.close()

print("Detection confidence plot saved!")

# ========================================
# Step 8: Detection Summary Statistics
# ========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Confidence distribution
confidences = [d['confidence'] for d in detections_sorted]
axes[0, 0].hist(confidences, bins=15, color='green', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Confidence Score', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Confidence Score Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Sorted confidence
axes[0, 1].plot(range(len(confidences)), sorted(confidences, reverse=True), 'o-', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Detection Rank', fontweight='bold')
axes[0, 1].set_ylabel('Confidence Score', fontweight='bold')
axes[0, 1].set_title('Detections Ranked by Confidence', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Statistics table
stats_text = f"""
Total Detections: {len(detections)}
Mean Confidence: {np.mean(confidences):.4f}
Max Confidence: {np.max(confidences):.4f}
Min Confidence: {np.min(confidences):.4f}

Mean Area: {np.mean(areas):.1f}
Mean Width: {np.mean(widths):.1f}
Mean Height: {np.mean(heights):.1f}

Objects in Center: {sum(1 for d in detections if np.sqrt((d['center'][0]-image.shape[1]/2)**2 + (d['center'][1]-image.shape[0]/2)**2) < 200)}
Objects on Edges: {sum(1 for d in detections if np.sqrt((d['center'][0]-image.shape[1]/2)**2 + (d['center'][1]-image.shape[0]/2)**2) > 300)}
"""

axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].axis('off')
axes[1, 0].set_title('Detection Statistics', fontweight='bold')

# Size categories
sizes = ['Small\n(<5000px)', 'Medium\n(5000-15000px)', 'Large\n(>15000px)']
size_counts = [
    sum(1 for a in areas if a < 5000),
    sum(1 for a in areas if 5000 <= a < 15000),
    sum(1 for a in areas if a >= 15000)
]
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
axes[1, 1].pie(size_counts, labels=sizes, autopct='%1.1f%%', colors=colors_pie, startangle=90)
axes[1, 1].set_title('Object Size Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/detection_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Detection summary plot saved!")

print("\n✅ Object Detection Complete!")
print("Generated outputs:")
print("  - outputs/objects_image.png")
print("  - outputs/preprocessing_detection.png")
print("  - outputs/object_detection_results.png")
print("  - outputs/object_properties.png")
print("  - outputs/detection_confidence.png")
print("  - outputs/detection_summary.png")
