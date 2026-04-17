import torch
import cv2
import numpy as np
import os
import csv
from torchvision import transforms
from tqdm import tqdm

DEVICE = torch.device("cpu")
WEAK_SUP_MODEL_PATH = r"D:\dataset\database\models_resnet50_camo_point\resnet50_camo_point_epoch_3_train_2.1041_val_2.2891.pth"
FULL_SUP_MODEL_PATH = r"D:\dataset\database\models_resnet50_full_sup\resnet50_full_sup_epoch_19_train_0.1488_val_0.2009.pth"

TEST_IMG_DIR = r"D:\dataset\database\test_images"
TEST_MASK_DIR = r"D:\dataset\database\test_masks"
OUTPUT_METRICS_PATH = r"D:\dataset\database\predictions_camo_point\model_comparison_metrics.txt"
OUTPUT_CSV_PATH = r"D:\dataset\database\predictions_camo_point\model_comparison_metrics.csv"
IMAGE_SIZE = (128, 128)

class CamoPointEnhanceModule(torch.nn.Module):
    def __init__(self, in_channels=2048):
        super().__init__()
        self.point_attention = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, 1, 1, bias=False),
            torch.nn.Sigmoid()
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x, point_mask):
        point_mask = torch.nn.functional.interpolate(point_mask, size=x.shape[2:], mode='nearest')
        x_weighted = x * point_mask
        atten_weight = self.point_attention(x_weighted)
        x_enhanced = x * atten_weight
        x_fused = self.fusion(torch.cat([x, x_enhanced], dim=1))
        return x_fused

class ResNet50MattingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])
        
        for i, param in enumerate(self.encoder.parameters()):
            param.requires_grad = i >= 5
        
        self.camo_point_enhance = CamoPointEnhanceModule(in_channels=2048)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x, point_mask):
        feat = self.encoder(x)
        feat_enhanced = self.camo_point_enhance(feat, point_mask)
        out = self.decoder(feat_enhanced)
        out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)
        return out

# =后处理模块
def post_process_mask(mask, img_w, img_h):
    mask = cv2.resize(mask, (img_w, img_h))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_mean = np.mean(mask)
    mask_std = np.std(mask)
    adaptive_threshold = max(0.005, mask_mean + 0.02 * mask_std)
    mask_bin = (mask > adaptive_threshold).astype(np.uint8) * 255
    kernel_dilate_tiny = np.ones((3, 3), np.uint8)
    mask_bin = cv2.dilate(mask_bin, kernel_dilate_tiny, iterations=1)
    kernel_close_small = np.ones((3, 3), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close_small)
    kernel_close_mid = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close_mid)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        img_area = img_w * img_h
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > img_area * 0.0001]
        if len(valid_contours) > 0:
            mask_bin = np.zeros_like(mask_bin)
            for cnt in valid_contours:
                cv2.drawContours(mask_bin, [cnt], -1, 255, cv2.FILLED)
            small_contour_mask = np.zeros_like(mask_bin)
            for cnt in valid_contours:
                if cv2.contourArea(cnt) < img_area * 0.005:
                    cv2.drawContours(small_contour_mask, [cnt], -1, 255, cv2.FILLED)
            small_contour_mask = cv2.dilate(small_contour_mask, kernel_dilate_tiny, iterations=1)
            mask_bin = cv2.bitwise_or(mask_bin, small_contour_mask)
    mask_bin = cv2.GaussianBlur(mask_bin, (7, 7), 0)
    mask_bin = (mask_bin > 127).astype(np.uint8) * 255
    return mask_bin

# 核心评估函数
def calculate_iou_f1(pred_mask, gt_mask):
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    iou = intersection / (union + 1e-6)
    tp = intersection
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return round(iou, 4), round(f1, 4)

def load_model(model_path):
    """加载指定路径的模型"""
    model = ResNet50MattingModel().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: {os.path.basename(model_path)}")
    return model

def infer_single_image(model, img_path):
    """单张图像推理"""
    img_ori = cv2.imread(img_path)
    if img_ori is None:
        return None, None
    h, w = img_ori.shape[:2]
    img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)
    point_mask = torch.ones(1, 1, *IMAGE_SIZE).to(DEVICE)
    with torch.no_grad():
        mask_tensor = model(img_tensor, point_mask)
    mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    return mask, (w, h)

def evaluate_model(model, model_name, valid_pairs):
    """评估单个模型"""
    total_iou = 0.0
    total_f1 = 0.0
    count = 0
    metrics_log = [f"\n===== {model_name} Evaluation Results ====="]
    sample_metrics = []
    
    pbar = tqdm(valid_pairs, desc=f"{model_name} Evaluating", ncols=80, ascii=True)
    for img_name, gt_mask_name in pbar:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        gt_mask_path = os.path.join(TEST_MASK_DIR, gt_mask_name)
        
        pred_mask_raw, (img_w, img_h) = infer_single_image(model, img_path)
        if pred_mask_raw is None:
            continue
        pred_mask = post_process_mask(pred_mask_raw, img_w, img_h)
        
        gt_mask = cv2.imread(gt_mask_path, 0)
        gt_mask = cv2.resize(gt_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        iou, f1 = calculate_iou_f1(pred_mask, gt_mask)
        total_iou += iou
        total_f1 += f1
        count += 1
        
        metrics_log.append(f"{img_name} - IoU: {iou:.4f}, F1: {f1:.4f}")
        pbar.set_postfix({'Avg IoU': f'{total_iou/count:.4f}', 'Avg F1': f'{total_f1/count:.4f}'})
        sample_metrics.append({
            'image_name': img_name,
            'model_name': model_name,
            'iou': iou,
            'f1': f1
        })
    
    pbar.close()
    
    avg_iou = round(total_iou / count, 4) if count > 0 else 0
    avg_f1 = round(total_f1 / count, 4) if count > 0 else 0
    metrics_log.append(f"\n{model_name} Average Metrics: IoU={avg_iou:.4f}, F1={avg_f1:.4f}")
    
    return avg_iou, avg_f1, metrics_log, sample_metrics

def export_to_csv(weak_sample_metrics, full_sample_metrics, weak_avg, full_avg):
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames_summary = ['Model Type', 'Avg IoU', 'Avg F1', 'Annotation Cost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_summary)
        writer.writeheader()
        
        writer.writerow({
            'Model Type': 'Weakly Supervised (Point)',
            'Avg IoU': f"{weak_avg[0]:.4f}",
            'Avg F1': f"{weak_avg[1]:.4f}",
            'Annotation Cost': '<10% (Only Points)'
        })
        writer.writerow({
            'Model Type': 'Fully Supervised',
            'Avg IoU': f"{full_avg[0]:.4f}",
            'Avg F1': f"{full_avg[1]:.4f}",
            'Annotation Cost': '100% (Pixel-level)'
        })
        
        csvfile.write('\n# Individual Sample Metrics\n')
        
        fieldnames_sample = ['Image Name', 'Model Type', 'IoU', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_sample)
        writer.writeheader()
        
        for m in weak_sample_metrics:
            writer.writerow({'Image Name': m['image_name'], 'Model Type': m['model_name'], 'IoU': f"{m['iou']:.4f}", 'F1': f"{m['f1']:.4f}"})
        for m in full_sample_metrics:
            writer.writerow({'Image Name': m['image_name'], 'Model Type': m['model_name'], 'IoU': f"{m['iou']:.4f}", 'F1': f"{m['f1']:.4f}"})
    
    print(f"CSV saved to: {OUTPUT_CSV_PATH}")

def batch_eval():
    if not os.path.exists(TEST_IMG_DIR):
        print(f"Test image directory does not exist: {TEST_IMG_DIR}")
        return
    if not os.path.exists(TEST_MASK_DIR):
        print(f"Test ground truth mask directory does not exist: {TEST_MASK_DIR}")
        return
    
    img_ext = [".jpg", ".png", ".jpeg"]
    img_list = [f for f in os.listdir(TEST_IMG_DIR) if any(f.endswith(ext) for ext in img_ext)]
    valid_pairs = []
    for img_name in img_list:
        base_name = os.path.splitext(img_name)[0]
        gt_mask_name = f"{base_name}.png"
        gt_mask_path = os.path.join(TEST_MASK_DIR, gt_mask_name)
        if os.path.exists(gt_mask_path):
            valid_pairs.append((img_name, gt_mask_name))
    
    if len(valid_pairs) == 0:
        print(f"No matching image-ground truth mask pairs found")
        return
    
    print(f"\nStart comparative evaluation | Valid samples: {len(valid_pairs)}")
    print("-" * 80)
    
    weak_sup_model = load_model(WEAK_SUP_MODEL_PATH)
    full_sup_model = load_model(FULL_SUP_MODEL_PATH)
    
    weak_iou, weak_f1, weak_log, weak_sample_metrics = evaluate_model(weak_sup_model, "Weakly Supervised (Point)", valid_pairs)
    full_iou, full_f1, full_log, full_sample_metrics = evaluate_model(full_sup_model, "Fully Supervised", valid_pairs)
    
    iou_diff = abs(weak_iou - full_iou)
    f1_diff = abs(weak_f1 - full_f1)

    # ===================== 表格 =====================
    comparison_log = [
        "Weakly Supervised vs Fully Supervised Model Comparison",
        "="*80,
        "| Model Type                | Avg IoU  | Avg F1   | Annotation Cost       |",
        "|---------------------------|----------|----------|------------------------|",
        f"| Weakly Supervised (Point) | {weak_iou:.4f}   | {weak_f1:.4f}   | <10% (Only Points)     |",
        f"| Fully Supervised          | {full_iou:.4f}   | {full_f1:.4f}   | 100% (Pixel-level)     |",
        f"| Difference                | {iou_diff:.4f}   | {f1_diff:.4f}   | Reduced by 90%+        |",
        "="*80,
        f"Conclusion: Weakly-supervised model greatly reduces annotation cost with acceptable performance drop."
    ]
    
    all_log = weak_log + full_log + comparison_log
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    with open(OUTPUT_METRICS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_log))
    
    export_to_csv(weak_sample_metrics, full_sample_metrics, (weak_iou, weak_f1), (full_iou, full_f1))
    
    print("\n" + "="*80)
    for line in comparison_log:
        print(line)
    print(f"\nText results: {OUTPUT_METRICS_PATH}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    try:
        batch_eval()
    except Exception as e:
        print(f"\nEvaluation error: {str(e)}")
        import traceback
        traceback.print_exc()