import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torchvision import transforms, models
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")
NEW_SAM_MODEL_PATH = r"D:\dataset\database\models_SAM\best_model.pth"
FULL_SUP_MODEL_PATH = r"D:\dataset\database\models_resnet50_full_sup\resnet50_full_sup_epoch_19_train_0.1488_val_0.2009.pth"
TEST_IMG_DIR = r"D:\dataset\database\test_images"
COMPARISON_OUTPUT_DIR = r"D:\dataset\database\新抠图效果图"
IMAGE_SIZE = (128, 128)
KEY_SAMPLES = None

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

# ===================== 全监督模型 =====================
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

# ===================== 后处理 =====================
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

# ===================== 加载模型 =====================
def load_model(model_path, model_type):
    if model_type == "sam":
        model = ResNet50_SAMCOD().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        model = ResNet50MattingModel().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"加载成功：{os.path.basename(model_path)}")
    return model

# ===================== 推理 =====================
def infer_single_image(model, img_path):
    img_ori = cv2.imread(img_path)
    if img_ori is None:
        return None, None
    img_h, img_w = img_ori.shape[:2]
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

    pred_mask_raw = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    pred_mask = post_process_mask(pred_mask_raw, img_w, img_h)
    return img_ori, pred_mask

# ===================== 生成对比图=====================
def generate_dual_model_comparison(img_ori, weak_mask, full_mask, img_name, output_dir):
    # 修复在这里！！！
    img_rgb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    
    weak_mask_rgb = cv2.cvtColor(weak_mask, cv2.COLOR_GRAY2RGB)
    weak_alpha = img_ori.copy()
    weak_alpha[weak_mask == 0] = [0, 0, 0]
    weak_alpha_rgb = cv2.cvtColor(weak_alpha, cv2.COLOR_BGR2RGB)

    full_mask_rgb = cv2.cvtColor(full_mask, cv2.COLOR_GRAY2RGB)
    full_alpha = img_ori.copy()
    full_alpha[full_mask == 0] = [0, 0, 0]
    full_alpha_rgb = cv2.cvtColor(full_alpha, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(25, 5))
    plt.subplot(1, 5, 1)
    plt.imshow(img_rgb)
    plt.title("Input Image", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(weak_mask_rgb, cmap='gray')
    plt.title("Weak Supervision Model - Mask", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(weak_alpha_rgb)
    plt.title("Weak Supervision Model - Result", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(full_mask_rgb, cmap='gray')
    plt.title("Full Supervision Model - Mask", fontsize=14, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(full_alpha_rgb)
    plt.title("Full Supervision Model - Result", fontsize=14, fontweight='bold')
    plt.axis('off')

    plot_save_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_sam_mask.png"), weak_mask)
    cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_sam_alpha.png"), weak_alpha)
    cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_full_mask.png"), full_mask)
    cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_full_alpha.png"), full_alpha)
    print(f"对比图已保存：{plot_save_path}")

# ===================== 批量 =====================
def batch_generate_comparison():
    os.makedirs(COMPARISON_OUTPUT_DIR, exist_ok=True)
    sam_model = load_model(NEW_SAM_MODEL_PATH, "sam")
    full_model = load_model(FULL_SUP_MODEL_PATH, "full")

    img_ext = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    img_list = [f for f in os.listdir(TEST_IMG_DIR) if any(f.endswith(ext) for ext in img_ext)]
    if KEY_SAMPLES is not None:
        img_list = [f for f in img_list if f in KEY_SAMPLES]

    print(f"\n样本数量：{len(img_list)}")
    for img_name in img_list:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        print(f"\n处理：{img_name}")
        img_ori, sam_mask = infer_single_image(sam_model, img_path)
        if img_ori is None:
            continue
        _, full_mask = infer_single_image(full_model, img_path)
        generate_dual_model_comparison(img_ori, sam_mask, full_mask, img_name, COMPARISON_OUTPUT_DIR)

    print(f"保存路径：{COMPARISON_OUTPUT_DIR}")

if __name__ == "__main__":
    try:
        batch_generate_comparison()
    except Exception as e:
        print(f"\n错误：{str(e)}")
        import traceback
        traceback.print_exc()