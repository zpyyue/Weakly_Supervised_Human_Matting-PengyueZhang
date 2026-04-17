import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

DATA_ROOT = r"D:\dataset\database\training"
IMG_DIR = os.path.join(DATA_ROOT, "images")          # 原始图像文件夹
POINT_MASK_DIR = os.path.join(DATA_ROOT, "masks_point")  # 点标注掩码文件夹
PSEUDO_SAVE_DIR = os.path.join(DATA_ROOT, "pseudo_labels")  # 伪标签保存文件夹
MAX_WORKERS = 8

# 伪标签生成参数
DILATE_KERNEL_SIZE = 15  # 关键点膨胀核大小
GAUSSIAN_KERNEL_SIZE = (51, 51)  # 高斯模糊核大小
GAUSSIAN_SIGMA = 15  # 高斯模糊标准差
SIMILARITY_THRESHOLD = 0.8  # 颜色相似度阈值

def get_color_similarity(img, seed_pixels, h, w):
    # 计算种子点的平均颜色
    seed_colors = [img[y, x] for (y, x) in seed_pixels if 0<=y<h and 0<=x<w]
    if len(seed_colors) == 0:
        return np.zeros((h, w), dtype=np.float32)
    
    seed_color = np.mean(seed_colors, axis=0).reshape(1, 1, 3)
    
    # 计算欧氏距离并归一化到0-1
    color_dist = np.sqrt(np.sum((img - seed_color) ** 2, axis=2))
    max_dist = np.max(color_dist) if np.max(color_dist) > 0 else 1.0
    color_similarity = 1 - (color_dist / max_dist)
    
    return color_similarity

def generate_pseudo_label(img_path, point_mask_path, save_path):
    try:
        # 1. 读取图像和点掩码
        img = cv2.imread(img_path)
        point_mask = cv2.imread(point_mask_path, 0)
        
        if img is None or point_mask is None:
            print(f"\n读取失败: {os.path.basename(img_path)} / {os.path.basename(point_mask_path)}")
            return False
        
        h, w = img.shape[:2]
        
        # 2. 提取前景/背景关键点
        fg_coords = np.argwhere(point_mask == 1)
        bg_coords = np.argwhere(point_mask == 0)
        
        if len(fg_coords) == 0 or len(bg_coords) == 0:
            print(f"\n关键点不足: {os.path.basename(img_path)}")
            return False
        
        # 3. 计算颜色相似度
        fg_similarity = get_color_similarity(img, fg_coords.tolist(), h, w)
        bg_similarity = get_color_similarity(img, bg_coords.tolist(), h, w)
        
        # 4. 生成初始伪标签（前景相似度 > 背景相似度 + 阈值）
        pseudo_label = np.zeros((h, w), dtype=np.float32)
        pseudo_label[fg_similarity > (bg_similarity + (1 - SIMILARITY_THRESHOLD))] = 1.0
        
        # 5. 膨胀关键点区域（增强前景）
        kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        fg_mask = (point_mask == 1).astype(np.uint8) * 255
        fg_dilated = cv2.dilate(fg_mask, kernel, iterations=1)
        pseudo_label[fg_dilated > 0] = 1.0
        
        # 6. 高斯模糊平滑边缘（适配抠图需求）
        pseudo_label = cv2.GaussianBlur(pseudo_label, GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA)
        
        # 7. 归一化到0-255并保存
        pseudo_label = (pseudo_label * 255).astype(np.uint8)
        cv2.imwrite(save_path, pseudo_label)
        
        return True
    
    except Exception as e:
        print(f"\n处理异常 {os.path.basename(img_path)}: {str(e)}")
        return False

def batch_generate_pseudo_labels():
    # 1. 创建保存目录
    os.makedirs(PSEUDO_SAVE_DIR, exist_ok=True)
    
    # 2. 获取文件列表并匹配
    img_files = glob.glob(os.path.join(IMG_DIR, "*.[jp][pn]g"))
    point_mask_files = glob.glob(os.path.join(POINT_MASK_DIR, "*.png"))
    
    # 构建文件名到路径的映射
    img_map = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
    point_mask_map = {os.path.splitext(os.path.basename(f))[0]: f for f in point_mask_files}
    
    # 找到匹配的文件对
    common_names = set(img_map.keys()) & set(point_mask_map.keys())
    if len(common_names) == 0:
        print("未找到匹配的图像和点掩码文件")
        return
    
    # 构建任务列表
    tasks = []
    for name in sorted(common_names):
        img_path = img_map[name]
        point_mask_path = point_mask_map[name]
        save_path = os.path.join(PSEUDO_SAVE_DIR, f"{name}.png")
        
        # 跳过已生成的文件
        if os.path.exists(save_path):
            continue
            
        tasks.append((img_path, point_mask_path, save_path))
    
    if len(tasks) == 0:
        print("所有伪标签已生成，无需处理")
        return
    
    print(f"开始生成伪标签 | 总任务数: {len(tasks)} | 线程数: {MAX_WORKERS}")
    print(f"保存路径: {PSEUDO_SAVE_DIR}")
    
    # 3. 多线程执行
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_task = {
            executor.submit(generate_pseudo_label, task[0], task[1], task[2]): task
            for task in tasks
        }
        
        # 进度条监控
        with tqdm(total=len(tasks), desc="生成伪标签", unit="file") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
    
    # 4. 输出统计结果
    print(f"生成完成 | 成功: {success_count} | 失败: {fail_count}")
    print(f"总处理文件: {success_count + fail_count}")
    print(f"伪标签文件夹总文件数: {len(glob.glob(os.path.join(PSEUDO_SAVE_DIR, '*.png')))}")

if __name__ == "__main__":
    # 打印配置信息
    print(f"数据根目录: {DATA_ROOT}")
    print(f"图像数量: {len(glob.glob(os.path.join(IMG_DIR, '*.[jp][pn]g')))}")
    print(f"点掩码数量: {len(glob.glob(os.path.join(POINT_MASK_DIR, '*.png')))}")
    batch_generate_pseudo_labels()