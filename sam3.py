import os
import cv2
import numpy as np
from glob import glob
from ultralytics.models.sam import SAM3SemanticPredictor

# ==================== 配置区 ====================
# 支持多个 Market-1501 训练集/测试集路径
INPUT_DIRS =[
    "/root/datasets/market1501/bounding_box_train",
    "/root/market1501/bounding_box_train_less",
    # 你可以继续在这里添加更多文件夹路径
]

# Mask 输出的基础根目录
MASKS_BASE_DIR = "/root/autodl-fs/datasets/market1501/masks"

# SAM 3 模型路径
MODEL_PATH = "/root/autodl-fs/sam3.pt"

# 论文中定义的上半身组件 (作为文本 Prompt)
UPPER_PROMPTS =[
    "head", "hair", "face", "neck", "upper-clothes", 
    "coat", "shirt", "arms", "hands", "backpack", "bag"
]

# 论文中定义的下半身组件
LOWER_PROMPTS =[
    "pants", "skirt", "legs", "shoes", "feet"
]

# 合并所有 prompt 进行单次查询以提高效率
ALL_PROMPTS = UPPER_PROMPTS + LOWER_PROMPTS
UPPER_INDICES = set(range(len(UPPER_PROMPTS)))  # 前半部分索引归属于上身
# ================================================

def process_soft_mask(binary_mask, kernel_size=3, blur_size=5):
    """
    根据《潜空间结构化先验输入.md》的定义：
    binary mask -> dilation -> Gaussian blur -> clamp to[0, 1]
    
    注：Market-1501 分辨率为 64x128，极低。因此使用较小的 kernel (3, 5)。
    """
    if binary_mask is None or binary_mask.max() == 0:
        return np.zeros_like(binary_mask, dtype=np.float32)
        
    # 确保 mask 类型为 uint8 (255)
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    # 1. 膨胀 (Dilation) 减少断裂
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # 2. 高斯模糊 (Gaussian blur) 产生平滑边缘 (soft mask)
    blurred = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
    
    # 3. 截断归一化至[0, 1]
    soft_mask = blurred.astype(np.float32) / 255.0
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    
    return soft_mask

def main():
    print("正在初始化 SAM 3 模型 (只需加载一次)...")
    # 初始化 SAM 3 语义预测器
    overrides = dict(
        conf=0.20,           # 置信度阈值可按需调整
        task="segment", 
        mode="predict", 
        model=MODEL_PATH,    
        half=True,           # 开启 FP16 提速
        verbose=False        # 关闭单次推理输出日志
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("SAM 3 模型加载成功！\n" + "-"*40)

    # 遍历处理每一个文件夹
    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            print(f"[跳过] 路径不存在: {input_dir}")
            continue

        # 获取当前文件夹的名称，例如 "bounding_box_train"
        folder_name = os.path.basename(os.path.normpath(input_dir))
        
        # 动态构建对应的输出路径
        out_upper_dir = os.path.join(MASKS_BASE_DIR, folder_name, "upper")
        out_lower_dir = os.path.join(MASKS_BASE_DIR, folder_name, "lower")
        
        # 创建输出文件夹
        os.makedirs(out_upper_dir, exist_ok=True)
        os.makedirs(out_lower_dir, exist_ok=True)
        
        image_paths = glob(os.path.join(input_dir, "*.jpg"))
        if not image_paths:
            print(f"[警告] 未在 {input_dir} 找到任何 .jpg 图像！")
            continue
            
        print(f"开始处理目录: {folder_name} | 发现 {len(image_paths)} 张图片")
        
        # 遍历当前文件夹内的所有图片
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            
            # 1. 读取原图尺寸 (用于还原 Mask 尺寸)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            h, w = img.shape[:2]
            
            # 2. 设置图像并推理
            predictor.set_image(img_path)
            results = predictor(text=ALL_PROMPTS)
            
            # 初始化空的二值掩码合集
            upper_mask = np.zeros((h, w), dtype=np.float32)
            lower_mask = np.zeros((h, w), dtype=np.float32)
            
            # 3. 解析 SAM 3 的预测结果
            result = results[0] 
            
            if result.masks is not None:
                masks_data = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for mask_idx, cls_idx in enumerate(classes):
                    m = masks_data[mask_idx]
                    
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                    if cls_idx in UPPER_INDICES:
                        upper_mask = np.maximum(upper_mask, m)
                    else:
                        lower_mask = np.maximum(lower_mask, m)
                        
            # 4. 生成 Soft Mask
            upper_soft = process_soft_mask(upper_mask)
            lower_soft = process_soft_mask(lower_mask)
            
            # 5. 保存结果
            out_upper_path = os.path.join(out_upper_dir, img_name.replace(".jpg", ".png"))
            out_lower_path = os.path.join(out_lower_dir, img_name.replace(".jpg", ".png"))
            
            cv2.imwrite(out_upper_path, (upper_soft * 255).astype(np.uint8))
            cv2.imwrite(out_lower_path, (lower_soft * 255).astype(np.uint8))
            
            if (i + 1) % 500 == 0:
                print(f"  [{folder_name}] 已处理 {i + 1}/{len(image_paths)} 张图片...")
                
        print(f"✔ 目录 {folder_name} 处理完毕！\n" + "-"*40)

    print("🎉 所有指定文件夹掩码处理全部完成！")

if __name__ == "__main__":
    main()