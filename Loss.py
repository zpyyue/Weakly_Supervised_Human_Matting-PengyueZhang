import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== Core Configuration =====================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
# Only use English font to avoid garbled characters
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== Real Training Data =====================
# Fully Supervised Model
fs_train = [0.2688, 0.2168, 0.2016, 0.1922, 0.1854, 0.1804, 0.1768, 0.1729,
            0.1662, 0.1631, 0.1609, 0.1590, 0.1576, 0.1562, 0.1549, 0.1539,
            0.1510, 0.1498, 0.1488, 0.1480]
fs_val   = [0.2527, 0.2364, 0.2239, 0.2193, 0.2184, 0.2146, 0.2119, 0.2100,
            0.2058, 0.2036, 0.2063, 0.2051, 0.2046, 0.2046, 0.2039, 0.2039,
            0.2009, 0.2018, 0.2009, 0.2033]

# Weakly Supervised (Point Annotation) Model
ws_train = [2.1622, 2.1243, 2.1041, 2.0874, 2.0686, 2.0550, 2.0410, 2.0277,
            2.0030, 1.9894, 1.9802, 1.9710, 1.9634, 1.9584, 1.9529, 1.9487,
            1.9377, 1.9327, 1.9278, 1.9257]
ws_val   = [2.5162, 2.3849, 2.2891, 2.3623, 2.4242, 2.5599, 2.5408, 2.4858,
            2.5654, 2.6376, 2.6685, 2.7597, 2.7500, 2.6842, 2.7228, 2.8293,
            2.8581, 2.8968, 2.8248, 2.9381]

epochs = np.arange(1, 21)

# ===================== Min-Max Normalization (0~1) =====================
def normalize_loss(loss_list):
    """Min-Max normalization: (x - min) / (max - min)"""
    min_loss = min(loss_list)
    max_loss = max(loss_list)
    return [(x - min_loss) / (max_loss - min_loss) for x in loss_list]

# Normalize all loss values
fs_train_norm = normalize_loss(fs_train)
fs_val_norm = normalize_loss(fs_val)
ws_train_norm = normalize_loss(ws_train)
ws_val_norm = normalize_loss(ws_val)

# ===================== Plotting =====================
fig, ax = plt.subplots(figsize=(12, 7))

# Weakly Supervised (Point Annotation) Model
ax.plot(epochs, ws_train_norm, 'b-', linewidth=2, marker='o', markersize=4, 
        label='Weakly Supervised - Train Loss')
ax.plot(epochs, ws_val_norm, 'b--', linewidth=2, marker='s', markersize=4, 
        label='Weakly Supervised - Val Loss')

# Fully Supervised Model
ax.plot(epochs, fs_train_norm, 'r-', linewidth=2, marker='^', markersize=4, 
        label='Fully Supervised - Train Loss')
ax.plot(epochs, fs_val_norm, 'r--', linewidth=2, marker='d', markersize=4, 
        label='Fully Supervised - Val Loss')

# Mark optimal points
ws_best_idx = ws_val.index(min(ws_val)) + 1  # Epoch 3
fs_best_idx = fs_val.index(min(fs_val)) + 1  # Epoch 17
ax.scatter(ws_best_idx, ws_val_norm[ws_best_idx-1], color='darkblue', s=180, zorder=5,
           label=f'WS Optimal (Epoch {ws_best_idx})')
ax.scatter(fs_best_idx, fs_val_norm[fs_best_idx-1], color='darkred', s=180, zorder=5,
           label=f'FS Optimal (Epoch {fs_best_idx})')

# Chart styling
ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalized Loss', fontsize=14, fontweight='bold')
ax.set_title('Loss Trend Comparison: Weakly vs Fully Supervised Models (Normalized)', 
             fontsize=16, fontweight='bold')
ax.set_xticks(epochs[::2])
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
plt.tight_layout()

# Save English version only
save_path = os.path.join(SAVE_DIR, "Loss_Comparison.png")
plt.savefig(save_path, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Save path: {save_path}")