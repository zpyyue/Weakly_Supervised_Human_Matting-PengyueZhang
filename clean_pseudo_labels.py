import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
import sys
import site

env_packages_path = r"C:\Users\HUAWEI\anaconda3\envs\matting\Lib\site-packages"
if env_packages_path not in sys.path:
    sys.path.insert(0, env_packages_path)
RAW_MASK_DIR = r"D:\dataset\database\training\pseudo_labels"
CLEAN_MASK_DIR = r"D:\dataset\database\pseudo_labels_cleaned"
MIN_FOREGROUND_AREA_RATIO = 0.05  # 前景面积至少占5%
MAX_NOISE_RATIO = 0.1  # 噪声占比不超过10%
os.makedirs(CLEAN_MASK_DIR, exist_ok=True)

def calculate_foreground_ratio(mask):
    foreground_pixels = np.sum(mask > 127)
    total_pixels = mask.shape[0] * mask.shape[1]
    return foreground_pixels / total_pixels if total_pixels > 0 else 0

def calculate_noise_ratio(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    noise_pixels = np.sum(np.abs(mask - mask_clean) > 127)
    total_pixels = mask.shape[0] * mask.shape[1]
    return noise_pixels / total_pixels if total_pixels > 0 else 0

def is_mask_valid(mask):
    # 1. 前景面积足够
    fg_ratio = calculate_foreground_ratio(mask)
    if fg_ratio < MIN_FOREGROUND_AREA_RATIO:
        # 按区间分类，避免零散数值
        if fg_ratio < 0.01:
            return False, "前景面积占比过低（0~1%）"
        elif fg_ratio < 0.03:
            return False, "前景面积占比过低（1~3%）"
        else:
            return False, "前景面积占比过低（3~5%）"
    
    # 2. 噪声占比低
    noise_ratio = calculate_noise_ratio(mask)
    if noise_ratio > MAX_NOISE_RATIO:
        return False, "噪声占比过高（＞10%）"
    
    # 3. 不是全黑/全白
    if np.all(mask == 0) or np.all(mask == 255):
        return False, "全黑/全白掩码"
    
    return True, "有效"

# 清洗伪标签
def clean_pseudo_labels():
    mask_list = [f for f in os.listdir(RAW_MASK_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"开始清洗伪标签 | 原始数量：{len(mask_list)}")
    
    valid_count = 0
    invalid_count = 0
    invalid_reasons = {}

    pbar = tqdm(
        mask_list, 
        desc="清洗伪标签", 
        ncols=80,
        file=sys.stdout,
        leave=True,
        ascii=True
    )
    
    update_interval = 20
    processed_count = 0
    
    for mask_name in pbar:
        mask_path = os.path.join(RAW_MASK_DIR, mask_name)
        mask = cv2.imread(mask_path, 0)
        processed_count += 1
        
        if mask is None:
            reason = "无法读取文件"
            invalid_count += 1
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        else:
            is_valid, reason = is_mask_valid(mask)
            if is_valid:
                shutil.copy(mask_path, os.path.join(CLEAN_MASK_DIR, mask_name))
                valid_count += 1
            else:
                invalid_count += 1
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        
        if processed_count % update_interval == 0 or processed_count == len(mask_list):
            pbar.set_postfix({'有效': valid_count, '无效': invalid_count})
    
    pbar.close()
    print(f"整体统计：")
    print(f"      - 原始数量：{len(mask_list)} 张")
    print(f"      - 有效掩码：{valid_count} 张（{valid_count/len(mask_list)*100:.1f}%）")
    print(f"      - 无效掩码：{invalid_count} 张（{invalid_count/len(mask_list)*100:.1f}%）")
    print(f"清洗后目录：{CLEAN_MASK_DIR}")
    
    # 打印合并后的无效原因
    if invalid_reasons:
        print(f"\n无效原因统计：")
        for reason, count in sorted(invalid_reasons.items()):
            ratio = count / invalid_count * 100 if invalid_count > 0 else 0
            print(f"   - {reason}：{count} 张（{ratio:.1f}%）")
    else:
        print(f"\n所有掩码均有效！")

def generate_high_quality_masks():

    try:
        import rembg
        print("\n开始生成高质量掩码...")
        
        img_dir = r"D:\dataset\database\training\images"
        # 检查图片目录是否存在
        if not os.path.exists(img_dir):
            print(f"图片目录不存在：{img_dir}")
            return
        
        img_list = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if len(img_list) == 0:
            print(f"图片目录中未找到图片文件：{img_dir}")
            return
        
        pbar = tqdm(
            img_list, 
            desc="生成高质量掩码", 
            ncols=80,
            file=sys.stdout,
            leave=True,
            ascii=True
        )
        
        for img_name in pbar:
            img_path = os.path.join(img_dir, img_name)
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(CLEAN_MASK_DIR, mask_name)
            
            # 如果已有有效掩码，跳过
            if os.path.exists(mask_path):
                continue
            
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # rembg处理
            img_rgba = rembg.remove(img)
            mask = img_rgba[:, :, 3]
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
            
            # 保存掩码
            cv2.imwrite(mask_path, mask)
        
        pbar.close()
        print("高质量掩码生成完成！")
    except ImportError as e:
        print(f"   错误详情：{e}")
    except Exception as e:
        print(f"\n生成高质量掩码出错：{str(e)}")

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    clean_pseudo_labels()
    generate_high_quality_masks()
    print(f"\n最终清洗后伪标签目录：{CLEAN_MASK_DIR}")