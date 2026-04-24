import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import sys
from torchvision import transforms, models
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 环境配置
sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cpu")
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

TRAIN_IMG_DIR = r"D:\dataset\database\training\images"
TRAIN_MASK_DIR = r"D:\dataset\database\training\masks"
VAL_IMG_DIR = r"D:\dataset\database\val\images"
VAL_MASK_DIR = r"D:\dataset\database\val\masks"

BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 20
IMAGE_SIZE = (128, 128)
POINT_WEIGHT = 3.0 
EDGE_WEIGHT = 0.5

MODEL_SAVE_DIR = r"D:\dataset\database\models_SAM"
RESUME_FILE = os.path.join(MODEL_SAVE_DIR, "resume_training.pth")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# SAM-COD 增强模块
class SAMCODFusionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.cod_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat, point_mask):
        point_mask = nn.functional.interpolate(point_mask, size=feat.shape[2:], mode='nearest')
        feat_point = feat * point_mask
        att = self.attention(feat_point)
        feat_att = feat * att
        feat_cod = self.cod_enhance(feat_att)
        return feat_cod

# ResNet50 + SAM-COD
class ResNet50_SAMCOD(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        for i, param in enumerate(self.encoder.parameters()):
            param.requires_grad = i >= 5

        self.sam_cod = SAMCODFusionModule(2048)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, point_mask):
        feat = self.encoder(x)
        feat = self.sam_cod(feat, point_mask)
        out = self.decoder(feat)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# MattingDataset
class MattingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_size=(128, 128), is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_train = is_train
        
        img_ext = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
        self.img_names = [f for f in os.listdir(img_dir) if any(f.endswith(ext) for ext in img_ext)]
        
        self.valid_pairs = []
        for img_name in self.img_names:
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(mask_dir, f"{base_name}.png")
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_name, base_name))
            else:
                print(f"跳过：掩码缺失 {base_name}.png")
        
        self.total_valid = len(self.valid_pairs)
        print(f"加载 {img_dir} | 有效数据: {self.total_valid} 张")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return self.total_valid

    def __getitem__(self, idx):
        try:
            img_name, base_name = self.valid_pairs[idx]
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size)

            mask_path = os.path.join(self.mask_dir, base_name+".png")
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, self.image_size)

            point_mask = np.zeros_like(mask)
            coords_fg = np.argwhere(mask > 127)
            coords_bg = np.argwhere(mask == 0)
            if len(coords_fg) > 0:
                for _ in range(2):
                    y, x = coords_fg[np.random.choice(len(coords_fg))]
                    point_mask[y, x] = 255
            if len(coords_bg) > 0:
                y, x = coords_bg[np.random.choice(len(coords_bg))]
                point_mask[y, x] = 255

            flip = False
            if self.is_train and np.random.rand() > 0.5:
                flip = True
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
                point_mask = cv2.flip(point_mask, 1)

            img = self.transform(img)
            mask = torch.tensor(mask / 255.).float().unsqueeze(0)
            point_mask = torch.tensor(point_mask / 255.).float().unsqueeze(0)
            return img, point_mask, mask

        except Exception as e:
            print(f"读取失败 {img_name}: {str(e)}")
            return torch.rand(3,128,128), torch.zeros(1,128,128), torch.zeros(1,128,128)

# 损失函数
class PointSupLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, point_mask, gt):
        point_loss = self.mse(pred * point_mask, gt * point_mask)
        pred_logit = torch.logit(pred.clamp(1e-6, 1-1e-6))
        global_loss = 0.7*self.mse(pred, gt) + 0.3*self.bce(pred_logit, gt)
        return POINT_WEIGHT * point_loss + global_loss

# 训练 + 断点续训
def train():
    model = ResNet50_SAMCOD().to(DEVICE)
    criterion = PointSupLoss().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    start_epoch = 1
    best_val_loss = 999999

    if os.path.exists(RESUME_FILE):
        ckpt = torch.load(RESUME_FILE, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"续训成功 | 从 Epoch {start_epoch} 开始")

    train_set = MattingDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMAGE_SIZE, is_train=True)
    val_set = MattingDataset(VAL_IMG_DIR, VAL_MASK_DIR, IMAGE_SIZE, is_train=False)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=0)

    for epoch in range(start_epoch, EPOCHS+1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for img, p_mask, gt in pbar:
            img, p_mask, gt = img.to(DEVICE), p_mask.to(DEVICE), gt.to(DEVICE)
            pred = model(img, p_mask)
            loss = criterion(pred, p_mask, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{train_loss/len(pbar):.3f}"})

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, p_mask, gt in val_loader:
                img, p_mask, gt = img.to(DEVICE), p_mask.to(DEVICE), gt.to(DEVICE)
                pred = model(img, p_mask)
                val_loss += criterion(pred, p_mask, gt).item()
        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss
        }, RESUME_FILE)

        print(f"Epoch {epoch} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

    print("\n训练完成！最优模型已保存：best_model.pth")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练中断，下次运行自动继续！")
