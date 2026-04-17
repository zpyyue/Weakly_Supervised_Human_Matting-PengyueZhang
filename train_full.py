import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import sys
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(line_buffering=True)
DEVICE = torch.device("cpu")
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
print(f"线程数: {torch.get_num_threads()}")

TRAIN_IMG_DIR = r"D:\dataset\database\training\images"
TRAIN_MASK_DIR = r"D:\dataset\database\training\masks"
VAL_IMG_DIR = r"D:\dataset\database\val\images"
VAL_MASK_DIR = r"D:\dataset\database\val\masks"

BATCH_SIZE = 4               
LEARNING_RATE = 5e-5         
EPOCHS = 20                  
IMAGE_SIZE = (128, 128)      
EDGE_WEIGHT = 0.5            
MODEL_SAVE_DIR = r"D:\dataset\database\models_resnet50_full_sup"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class CamoPointEnhanceModule(nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.point_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, point_mask):
        point_mask = nn.functional.interpolate(point_mask, size=x.shape[2:], mode='nearest')
        x_weighted = x * point_mask
        atten_weight = self.point_attention(x_weighted)
        x_enhanced = x * atten_weight
        x_fused = self.fusion(torch.cat([x, x_enhanced], dim=1))
        return x_fused

class ResNet50MattingModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        for i, param in enumerate(self.encoder.parameters()):
            param.requires_grad = i >= 5
        
        self.camo_point_enhance = CamoPointEnhanceModule(in_channels=2048)
        
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
        feat_enhanced = self.camo_point_enhance(feat, point_mask)
        out = self.decoder(feat_enhanced)
        out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# 全监督数据集类
class FullSupMattingDataset(Dataset):
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
        
        if self.total_valid == 0:
            raise ValueError("无匹配的图像-掩码文件！请检查路径")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.total_valid

    def __getitem__(self, idx):
        try:
            img_name, base_name = self.valid_pairs[idx]
            
            # 读取图像
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise Exception("图像读取失败")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            # 数据增强
            flip = False
            if self.is_train:
                if np.random.rand() > 0.5:
                    flip = True
                    img = cv2.flip(img, 1)
                if np.random.rand() > 0.5:
                    alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                    beta = np.random.uniform(-0.1, 0.1) * 255
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            
            img = self.transform(img)
            
            # 读取掩码（全监督用完整掩码）
            mask_path = os.path.join(self.mask_dir, f"{base_name}.png")
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                raise Exception("掩码读取失败")
            
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            if flip:
                mask = cv2.flip(mask, 1)
            
            # 归一化
            mask = torch.from_numpy(mask).float() / 255.0
            mask = torch.clamp(mask, 1e-6, 1.0 - 1e-6)
            
            # 全监督不需要点标注，返回全1掩码
            point_mask = torch.ones_like(mask)
            
            return img, point_mask.unsqueeze(0), mask.unsqueeze(0)
        
        except Exception as e:
            print(f"处理 {img_name} 出错：{str(e)}")
            default_img = torch.randn(3, *self.image_size) * 0.01 + 0.5
            default_mask = torch.ones(1, *self.image_size) * 0.5
            return default_img, default_mask, default_mask

# 全监督损失函数
class FullSupLoss(nn.Module):
    def __init__(self, edge_weight=EDGE_WEIGHT):
        super().__init__()
        self.edge_weight = edge_weight
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 边缘损失用Sobel算子
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_x.weight.data = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).float().view(1,1,3,3)
        self.sobel_y.weight.data = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]]).float().view(1,1,3,3)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, gt_mask):
        # 数值稳定
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        gt_mask = torch.clamp(gt_mask, 1e-6, 1.0 - 1e-6)
        
        # 全局损失（MSE + BCE）
        pred_logit = torch.logit(pred)
        global_loss = 0.7 * self.mse(pred, gt_mask) + 0.3 * self.bce(pred_logit, gt_mask)
        
        # 边缘损失
        pred_edge = torch.sqrt(self.sobel_x(pred)**2 + self.sobel_y(pred)**2 + 1e-8)
        gt_edge = torch.sqrt(self.sobel_x(gt_mask)**2 + self.sobel_y(gt_mask)**2 + 1e-8)
        edge_loss = self.mse(pred_edge, gt_edge)
        
        # 总损失
        total_loss = global_loss + self.edge_weight * edge_loss
        total_loss = torch.clamp(total_loss, 1e-4, 100.0)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(1.0).to(DEVICE)
        
        return total_loss

def train():
    RESUME_CHECKPOINT = os.path.join(MODEL_SAVE_DIR, "resume_checkpoint.pth")
    start_epoch = 1
    best_val_loss = float('inf')
    best_model_path = ""
    train_log = []

    # 加载数据
    train_set = FullSupMattingDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMAGE_SIZE, is_train=True)
    val_set = FullSupMattingDataset(VAL_IMG_DIR, VAL_MASK_DIR, IMAGE_SIZE, is_train=False)
    
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, drop_last=True, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, drop_last=True, pin_memory=False
    )
    
    # 初始化模型
    model = ResNet50MattingModel().to(DEVICE)
    criterion = FullSupLoss(EDGE_WEIGHT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=8,
        gamma=0.5
    )

    # 断点续训
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"\n发现续训文件，加载上次训练状态...")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['current_epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_log = checkpoint['train_log']
        
        print(f"续训加载完成 | 上次训练到Epoch: {start_epoch-1} | 最优验证损失: {best_val_loss:.4f}")
    else:
        print("\n无续训文件，从头开始训练...")

    print(f"\n开始全监督训练（共{EPOCHS}轮，从Epoch {start_epoch}开始）")

    for epoch in range(start_epoch, EPOCHS+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", 
            ncols=80, file=sys.stdout, leave=False, dynamic_ncols=False
        )
        
        for imgs, point_masks, gt_masks in pbar:
            try:
                imgs, point_masks, gt_masks = imgs.to(DEVICE), point_masks.to(DEVICE), gt_masks.to(DEVICE)
                
                pred = model(imgs, point_masks)
                loss = criterion(pred, gt_masks)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                avg_batch_loss = train_loss / batch_count
                pbar.set_postfix({'Loss': f'{avg_batch_loss:.4f}'}, refresh=False)
            except Exception as e:
                print(f"\n训练批次出错：{str(e)}")
                batch_count += 1
                continue
        
        pbar.close()
        avg_train_loss = train_loss / max(batch_count, 1)
        if avg_train_loss < 1e-4:
            avg_train_loss = 1.0
            print(f"Epoch {epoch} 训练损失异常，已重置为1.0")
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            pbar_val = tqdm(
                val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]", 
                ncols=80, file=sys.stdout, leave=False, dynamic_ncols=False
            )
            
            for imgs, point_masks, gt_masks in pbar_val:
                try:
                    imgs, point_masks, gt_masks = imgs.to(DEVICE), point_masks.to(DEVICE), gt_masks.to(DEVICE)
                    pred = model(imgs, point_masks)
                    loss = criterion(pred, gt_masks)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
                    avg_val_batch_loss = val_loss / val_batch_count
                    pbar_val.set_postfix({'Val Loss': f'{avg_val_batch_loss:.4f}'}, refresh=False)
                except Exception as e:
                    print(f"\n验证批次出错：{str(e)}")
                    val_batch_count += 1
                    continue
        
        pbar_val.close()
        avg_val_loss = val_loss / max(val_batch_count, 1)
        if avg_val_loss < 1e-4:
            avg_val_loss = 1.0
            print(f" Epoch {epoch} 验证损失异常，已重置为1.0")

        # 日志与保存
        train_log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })
        
        log_str = f"Epoch {epoch:2d}/{EPOCHS} | 训练损失：{avg_train_loss:.4f} | 验证损失：{avg_val_loss:.4f} | LR：{current_lr:.6f}"
        
        model_filename = f"resnet50_full_sup_epoch_{epoch}_train_{avg_train_loss:.4f}_val_{avg_val_loss:.4f}.pth"
        model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_log': train_log
        }, model_path)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = model_path
            log_str += f" | 最优模型: {model_filename}"
        else:
            log_str += f" | 保存模型: {model_filename}"
        
        print(log_str)

        resume_checkpoint = {
            'current_epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_log': train_log
        }
        torch.save(resume_checkpoint, RESUME_CHECKPOINT)
        print(f"续训文件已保存：{RESUME_CHECKPOINT}")

    print(f"\n全监督训练完成！已跑完全部{EPOCHS}轮")
    print(f"最优验证损失：{best_val_loss:.4f}")
    print(f"最优模型路径：{best_model_path}")
    print(f"所有模型保存在：{MODEL_SAVE_DIR}")
    print("\n20轮验证损失排名（前5）：")
    log_sorted = sorted(train_log, key=lambda x: x['val_loss'])
    for i, item in enumerate(log_sorted[:5]):
        print(f"{i+1}. Epoch{item['epoch']} | 验证损失：{item['val_loss']:.4f} | 训练损失：{item['train_loss']:.4f}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n训练被手动中断！已保存的模型仍可使用")
    except Exception as e:
        print(f"\n训练异常终止：{str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n全监督训练流程结束")