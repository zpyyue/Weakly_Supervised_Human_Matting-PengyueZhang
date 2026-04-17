import matplotlib.pyplot as plt
import numpy as np
import os

# ===================== Core Configuration =====================
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
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

# SAM Weak Supervision Model
sam_train = [
    0.0808, 0.0513, 0.0451, 0.0409, 0.0370, 0.0356, 0.0337, 0.0322,
    0.0287, 0.0272, 0.0262, 0.0255, 0.0248, 0.0245, 0.0238, 0.0232,
    0.0224, 0.0219, 0.0216, 0.0214
]
sam_val = [
    0.0612, 0.0523, 0.0488, 0.0471, 0.0457, 0.0479, 0.0443, 0.0424,
    0.0405, 0.0410, 0.0407, 0.0399, 0.0389, 0.0389, 0.0385, 0.0379,
    0.0380, 0.0380, 0.0371, 0.0379
]

epochs = np.arange(1, 21)

# ===================== 分别归一化 =====================
def normalize_single_curve(loss):
    loss = np.array(loss)
    return (loss - loss.min()) / (loss.max() - loss.min())

# 每条曲线自己独立归一化
fs_train_norm = normalize_single_curve(fs_train)
fs_val_norm   = normalize_single_curve(fs_val)
sam_train_norm = normalize_single_curve(sam_train)
sam_val_norm   = normalize_single_curve(sam_val)

# ===================== Plotting =====================
fig, ax = plt.subplots(figsize=(12, 7))

# SAM Weak Supervision
ax.plot(epochs, sam_train_norm, 'b-', linewidth=2.5, marker='o', markersize=5, 
        label='Weak Supervision - Train')
ax.plot(epochs, sam_val_norm, 'b--', linewidth=2.5, marker='s', markersize=5, 
        label='Weak Supervision - Val')

# Fully Supervised
ax.plot(epochs, fs_train_norm, 'r-', linewidth=2.5, marker='^', markersize=5, 
        label='Fully Supervised - Train')
ax.plot(epochs, fs_val_norm, 'r--', linewidth=2.5, marker='d', markersize=5, 
        label='Fully Supervised - Val')

# 标记最优 epoch
sam_best_epoch = np.argmin(sam_val) + 1
fs_best_epoch  = np.argmin(fs_val) + 1

ax.scatter(sam_best_epoch, sam_val_norm[sam_best_epoch-1], 
           color='blue', s=200, zorder=5,
           label=f'Best Weak Sup (Epoch {sam_best_epoch})')
ax.scatter(fs_best_epoch, fs_val_norm[fs_best_epoch-1], 
           color='red', s=200, zorder=5,
           label=f'Best Full Sup (Epoch {fs_best_epoch})')

# ===================== Style =====================
ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalized Loss (0~1)', fontsize=14, fontweight='bold')
ax.set_title('Loss Curve Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(epochs)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3, linestyle='-')
ax.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
plt.tight_layout()

save_path = os.path.join(SAVE_DIR, "Loss对比图.png")
plt.savefig(save_path, bbox_inches='tight', facecolor='white')
plt.close()

print("图片已生成：Loss对比图.png")