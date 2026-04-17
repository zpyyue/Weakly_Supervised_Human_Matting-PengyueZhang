import cv2
import numpy as np
import os
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(42)
np.random.seed(42)

def generate_point_mask(full_mask_path, save_path, num_foreground=2, num_background=1):
    try:
        # 读取全监督掩码（灰度图）
        full_mask = cv2.imread(full_mask_path, 0)
        if full_mask is None:
            print(f"\n读取掩码失败：{full_mask_path}")
            return False
        
        # 初始化点标注掩码
        point_mask = np.ones_like(full_mask) * 255
        
        # 提取前景/背景像素的坐标
        foreground_coords = np.argwhere((full_mask > 128) | (full_mask == 1))  # 前景像素
        background_coords = np.argwhere((full_mask == 0) | (full_mask < 128))  # 背景像素
        
        # 生成前景关键点
        if len(foreground_coords) >= num_foreground:
            selected_fg = random.sample(foreground_coords.tolist(), num_foreground)
            for (y, x) in selected_fg:
                point_mask[y, x] = 1
        else:
            if len(foreground_coords) > 0:
                center_y, center_x = foreground_coords[len(foreground_coords)//2]
                point_mask[center_y, center_x] = 1
            else:
                h, w = full_mask.shape
                point_mask[h//2, w//2] = 1
        
        # 生成背景关键点
        if len(background_coords) >= num_background:
            selected_bg = random.sample(background_coords.tolist(), num_background)
            for (y, x) in selected_bg:
                point_mask[y, x] = 0
        else:
            if len(background_coords) > 0:
                corner_y, corner_x = background_coords[0]
                point_mask[corner_y, corner_x] = 0
            else:
                h, w = full_mask.shape
                point_mask[0, 0] = 0
        
        # 保存点标注掩码
        cv2.imwrite(save_path, point_mask)
        return True
    except Exception as e:
        print(f"\n处理 {full_mask_path} 出错：{str(e)}")
        return False

def batch_generate_point_masks(data_root, max_workers=8):
    train_mask_dir = os.path.join(data_root, "masks")
    point_mask_dir = os.path.join(data_root, "masks_point")
    os.makedirs(point_mask_dir, exist_ok=True)
    
    # 遍历所有全监督掩码
    mask_names = [f for f in os.listdir(train_mask_dir) if f.endswith(".png")]
    # 跳过已生成的点掩码文件
    mask_names = [f for f in mask_names if not os.path.exists(os.path.join(point_mask_dir, f))]
    
    print(f"剩余待处理 {len(mask_names)} 个掩码文件...")
    
    # 构建任务列表
    tasks = []
    for mask_name in mask_names:
        full_mask_path = os.path.join(train_mask_dir, mask_name)
        point_mask_path = os.path.join(point_mask_dir, mask_name)
        tasks.append((full_mask_path, point_mask_path))
    
    # 多线程执行
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_task = {
            executor.submit(generate_point_mask, task[0], task[1]): task 
            for task in tasks
        }
        
        # 进度条
        with tqdm(total=len(tasks), desc="批量生成点标注掩码") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                if result:
                    success_count += 1
                else:
                    fail_count += 1
                pbar.update(1)
    
    # 输出统计
    print(f"\n处理完成！")
    print(f"成功生成：{success_count} 个点标注掩码")
    print(f"处理失败：{fail_count} 个文件")

if __name__ == "__main__":
    DATA_ROOT = r"D:\dataset\database\training"
    batch_generate_point_masks(DATA_ROOT, max_workers=8)