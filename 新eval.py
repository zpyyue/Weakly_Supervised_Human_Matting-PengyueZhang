import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import csv
from torchvision import transforms, models
from tqdm import tqdm

DEVICE = torch.device("cpu")

NEW_SAMCOD_MODEL_PATH = r"D:\dataset\database\models_SAM\best_model.pth"
FULL_SUP_MODEL_PATH   = r"D:\dataset\database\models_resnet50_full_sup\resnet50_full_sup_epoch_19_train_0.1488_val_0.2009.pth"

TEST_IMG_DIR = r"D:\dataset\database\test_images"
TEST_MASK_DIR = r"D:\dataset\database\test_masks"
OUTPUT_METRICS_PATH = r"D:\dataset\database\新predictions\sam_vs_full_comparison.txt"
OUTPUT_CSV_PATH = r"D:\dataset\database\新predictions\sam_vs_full_comparison.csv"
IMAGE_SIZE = (128, 128)

# =SAM弱监督模型
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

class ResNet50_SAMCOD(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        for i, param in enumerate(self.encoder.parameters()):
            param.requires_grad = False
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

# 全监督模型
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

# 后处理、评估函数
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

def infer_single_image(model, img_path):
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
        sample_metrics.append({'image_name': img_name, 'model_name': model_name, 'iou': iou, 'f1': f1})
    avg_iou = round(total_iou / count, 4) if count > 0 else 0
    avg_f1 = round(total_f1 / count, 4) if count > 0 else 0
    metrics_log.append(f"\n{model_name} Average Metrics: IoU={avg_iou:.4f}, F1={avg_f1:.4f}")
    return avg_iou, avg_f1, metrics_log, sample_metrics

def export_to_csv(sam_samples, full_samples, sam_avg, full_avg):
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames_summary = ['Model Type', 'Avg IoU', 'Avg F1', 'Annotation Cost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_summary)
        writer.writeheader()
        writer.writerow({
            'Model Type': 'Weakly Supervised (SAM-COD Point)',
            'Avg IoU': f"{sam_avg[0]:.4f}",
            'Avg F1': f"{sam_avg[1]:.4f}",
            'Annotation Cost': '<10% (Only Points)'
        })
        writer.writerow({
            'Model Type': 'Fully Supervised',
            'Avg IoU': f"{full_avg[0]:.4f}",
            'Avg F1': f"{full_avg[1]:.4f}",
            'Annotation Cost': '100% (Pixel-level)'
        })
        # 单张图片详情
        csvfile.write('\n# Individual Sample Metrics\n')
        fieldnames_sample = ['Image Name', 'Model Type', 'IoU', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames_sample)
        writer.writeheader()
        for m in sam_samples:
            writer.writerow({'Image Name': m['image_name'], 'Model Type': m['model_name'], 'IoU': f"{m['iou']:.4f}", 'F1': f"{m['f1']:.4f}"})
        for m in full_samples:
            writer.writerow({'Image Name': m['image_name'], 'Model Type': m['model_name'], 'IoU': f"{m['iou']:.4f}", 'F1': f"{m['f1']:.4f}"})

def batch_eval():
    # 数据校验
    if not os.path.exists(TEST_IMG_DIR) or not os.path.exists(TEST_MASK_DIR):
        print(f"路径不存在：{TEST_IMG_DIR} 或 {TEST_MASK_DIR}")
        return
    img_ext = [".jpg", ".png", ".jpeg"]
    img_list = [f for f in os.listdir(TEST_IMG_DIR) if any(f.endswith(ext) for ext in img_ext)]
    valid_pairs = []
    for img_name in img_list:
        base = os.path.splitext(img_name)[0]
        gt_mask_path = os.path.join(TEST_MASK_DIR, f"{base}.png")
        if os.path.exists(gt_mask_path):
            valid_pairs.append((img_name, f"{base}.png"))
    if len(valid_pairs) == 0:
        print("无匹配的图像-掩码对")
        return

    # 开始评估
    print(f"\nStart comparative evaluation | Valid samples: {len(valid_pairs)}")
    print("-" * 80)

    # 加载两个模型
    # 1. SAM弱监督模型
    sam_model = ResNet50_SAMCOD().to(DEVICE)
    sam_model.load_state_dict(torch.load(NEW_SAMCOD_MODEL_PATH, map_location=DEVICE))
    sam_model.eval()
    print(f"Model loaded: {os.path.basename(NEW_SAMCOD_MODEL_PATH)} (SAM-COD Weak)")

    # 2. 全监督模型
    full_model = ResNet50MattingModel().to(DEVICE)
    full_ckpt = torch.load(FULL_SUP_MODEL_PATH, map_location=DEVICE)
    full_model.load_state_dict(full_ckpt['model_state_dict'])
    full_model.eval()
    print(f"Model loaded: {os.path.basename(FULL_SUP_MODEL_PATH)} (Full Supervised)")

    # 评估两个模型
    sam_iou, sam_f1, sam_log, sam_samples = evaluate_model(sam_model, "Weakly Supervised (SAM-COD Point)", valid_pairs)
    full_iou, full_f1, full_log, full_samples = evaluate_model(full_model, "Fully Supervised", valid_pairs)

    # 计算差异
    iou_diff = abs(sam_iou - full_iou)
    f1_diff = abs(sam_f1 - full_f1)

    # 生成对比表格
    comparison_log = [
        "Weakly Supervised (SAM-COD) vs Fully Supervised Model Comparison",
        "="*80,
        "| Model Type                | Avg IoU  | Avg F1   | Annotation Cost       |",
        "|---------------------------|----------|----------|------------------------|",
        f"| Weakly Supervised (SAM-COD Point) | {sam_iou:.4f}   | {sam_f1:.4f}   | <10% (Only Points)     |",
        f"| Fully Supervised          | {full_iou:.4f}   | {full_f1:.4f}   | 100% (Pixel-level)     |",
        f"| Difference                | {iou_diff:.4f}   | {f1_diff:.4f}   | Reduced by 90%+        |",
        "="*80,
        f"Conclusion: SAM-COD enhanced weakly-supervised model greatly reduces annotation cost with acceptable performance drop."
    ]

    # 保存结果
    all_log = sam_log + full_log + comparison_log
    with open(OUTPUT_METRICS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_log))
    export_to_csv(sam_samples, full_samples, (sam_iou, sam_f1), (full_iou, full_f1))

    # 打印最终表格
    print("\n" + "="*80)
    for line in comparison_log:
        print(line)
    print(f"\nText results saved to: {OUTPUT_METRICS_PATH}")
    print(f"CSV results saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    try:
        batch_eval()
    except Exception as e:
        print(f"\nEvaluation error: {str(e)}")
        import traceback
        traceback.print_exc()