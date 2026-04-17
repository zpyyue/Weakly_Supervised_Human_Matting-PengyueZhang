"""
人体分割实验：全监督 vs 单点监督训练对比（优化+防漏点版）

功能：
1. 自动生成单点标注 + 可视化，保证不丢文件
2. 全监督和单点监督训练对比
3. 混合精度训练 + 小 batch + resize，节省显存
4. 每 epoch 记录训练时间和显存
5. IoU & Pixel Accuracy 可视化比较

依赖：
- Python 3
- PyTorch, torchvision
- OpenCV
- numpy
- matplotlib
- tqdm
"""

import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.cuda.amp import autocast, GradScaler

# ---------------------- 配置 ----------------------
np.random.seed(5)
random.seed(5)
torch.manual_seed(5)

# 数据集路径
DATA_ROOT = r"D:\dataset\database"
TRAIN_IMAGES = os.path.join(DATA_ROOT, "training", "images")
TRAIN_MASKS = os.path.join(DATA_ROOT, "training", "masks")
VAL_IMAGES = os.path.join(DATA_ROOT, "val", "images")
VAL_MASKS = os.path.join(DATA_ROOT, "val", "masks")

# 单点标注保存路径
POINT_MASKS = os.path.join(DATA_ROOT, "training", "masks_point")
POINT_VIS = os.path.join(DATA_ROOT, "training", "masks_point_vis")
os.makedirs(POINT_MASKS, exist_ok=True)
os.makedirs(POINT_VIS, exist_ok=True)

# train_id.txt 路径
LIST_PATH = os.path.join(DATA_ROOT, "training", "train_id.txt")

# 模型和训练参数
NUM_CLASSES = 2  # 背景 + 人体
BATCH_SIZE = 2   # 显存低可改小
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_H, RESIZE_W = 320, 320  # 缩小图片，节省显存

# ---------------------- 单点标注生成函数（防漏点版） ----------------------
def uniform_sample_point(mask_path, save_mask_path, save_vis_path):
    """
    为每张 mask 随机生成一个人体点（如果有），并生成可视化。
    即使 mask 异常，也会生成空 mask，避免丢失。
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 尝试彩色读取并取第一个通道
        img_color = cv2.imread(mask_path)
        if img_color is None:
            print(f"❌ 读取失败: {mask_path}")
            h, w = 256, 256  # 默认大小
            new_gt = np.ones((h, w), dtype=np.uint8) * 255
            vis = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(save_mask_path, new_gt)
            cv2.imwrite(save_vis_path, vis)
            return
        img = img_color[:, :, 0]

    h, w = img.shape
    new_gt = np.ones((h, w), dtype=np.uint8) * 255
    vis = np.zeros((h, w), dtype=np.uint8)

    # 所有非零像素都认为是人体
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        print(f"⚠️ 无有效人体区域: {mask_path}")
        cv2.imwrite(save_mask_path, new_gt)
        cv2.imwrite(save_vis_path, vis)
        return

    # 随机选一个点
    idx = np.random.randint(len(xs))
    y, x = ys[idx], xs[idx]
    new_gt[y, x] = 1
    cv2.circle(vis, (x, y), 6, 128, -1)

    cv2.imwrite(save_mask_path, new_gt)
    cv2.imwrite(save_vis_path, vis)

# ---------------------- 数据集类 ----------------------
class HumanSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace(".png", ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # resize
        image = cv2.resize(image, (RESIZE_W, RESIZE_H))
        mask = cv2.resize(mask, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask

# ---------------------- 模型 ----------------------
def get_model(num_classes=NUM_CLASSES):
    model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    return model.to(DEVICE)

# ---------------------- 指标计算 ----------------------
def pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = target.numel()
    return correct.float() / total

def iou(pred, target, num_classes=NUM_CLASSES):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# ---------------------- 优化训练函数 ----------------------
def train_model_optimized(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()
    best_iou = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_acc": [], "epoch_time": [], "gpu_mem_MB": []}

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)['out']
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # 验证
        model.eval()
        val_loss = 0
        val_iou = 0
        val_acc = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with autocast():
                    outputs = model(imgs)['out']
                    loss = criterion(outputs, masks)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_iou += iou(preds, masks) * imgs.size(0)
                val_acc += pixel_accuracy(preds, masks) * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(time.time() - start_time)
        if torch.cuda.is_available():
            history["gpu_mem_MB"].append(torch.cuda.max_memory_allocated() / 1024 / 1024)

        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | IoU {val_iou:.4f} | Acc {val_acc:.4f} | Time {history['epoch_time'][-1]:.1f}s | GPU {history['gpu_mem_MB'][-1]:.0f} MB")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")

    return history

# ---------------------- 生成单点标注 ----------------------
with open(LIST_PATH, "r") as f:
    ids = [line.strip() for line in f]

missing_count = 0
for name in ids:
    mask_path_jpg = os.path.join(TRAIN_MASKS, name + ".jpg")
    mask_path_png = os.path.join(TRAIN_MASKS, name + ".png")
    mask_path = mask_path_jpg if os.path.exists(mask_path_jpg) else mask_path_png
    if not os.path.exists(mask_path):
        print(f"❌ 找不到文件: {name}")
        missing_count += 1
        continue

    save_path = os.path.join(POINT_MASKS, name + ".png")
    save_vis_path = os.path.join(POINT_VIS, name + "_vis.png")
    uniform_sample_point(mask_path, save_path, save_vis_path)
    print(f"✅ 处理完成: {name}")

print(f"✅ 单点标注生成完成，总缺失文件数: {missing_count}")

# ---------------------- 数据加载 ----------------------
train_dataset_full = HumanSegDataset(TRAIN_IMAGES, TRAIN_MASKS)
train_dataset_point = HumanSegDataset(TRAIN_IMAGES, POINT_MASKS)
val_dataset = HumanSegDataset(VAL_IMAGES, VAL_MASKS)

train_loader_full = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
train_loader_point = DataLoader(train_dataset_point, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------------- 全监督训练 ----------------------
print("===== 开始全监督训练 =====")
model_full = get_model()
history_full = train_model_optimized(model_full, train_loader_full, val_loader)

# ---------------------- 单点监督训练 =====
print("===== 开始单点监督训练 =====")
model_point = get_model()
history_point = train_model_optimized(model_point, train_loader_point, val_loader)

# ---------------------- 性能可视化 ----------------------
epochs_range = range(1, EPOCHS+1)
plt.figure(figsize=(12,5))
plt.plot(epochs_range, history_full["val_iou"], label="Full Supervision IoU")
plt.plot(epochs_range, history_point["val_iou"], label="Point Supervision IoU")
plt.xlabel("Epochs")
plt.ylabel("IoU")
plt.title("IoU Comparison")
plt.legend()
plt.savefig("iou_comparison.png")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(epochs_range, history_full["val_acc"], label="Full Supervision Acc")
plt.plot(epochs_range, history_point["val_acc"], label="Point Supervision Acc")
plt.xlabel("Epochs")
plt.ylabel("Pixel Accuracy")
plt.title("Pixel Accuracy Comparison")
plt.legend()
plt.savefig("acc_comparison.png")
plt.show()

# ---------------------- 最终性能表格 ----------------------
print("===== 最终性能对比 =====")
print(f"全监督 IoU: {history_full['val_iou'][-1]:.4f}, Pixel Acc: {history_full['val_acc'][-1]:.4f}")
print(f"单点监督 IoU: {history_point['val_iou'][-1]:.4f}, Pixel Acc: {history_point['val_acc'][-1]:.4f}")
