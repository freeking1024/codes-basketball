# 使用训练好的模型对指定路径下的所有图片进行检测，并将检测结果保存在与 images 同级的 labels 目录下
from ultralytics import YOLO
import os
from pathlib import Path

def run_detection_and_save_labels(model_path, images_dir, confidence_threshold=0.5):
    """
    使用训练好的模型对指定路径下的所有图片进行检测，
    并将检测结果保存在与 images 同级的 labels 目录下
    
    Args:
        model_path (str): 训练好的模型路径
        images_dir (str): 包含待检测图片的目录路径
        confidence_threshold (float): 置信度阈值
    """
    # 加载训练好的模型
    model = YOLO(model_path)
    
    # 获取模型的类别名称
    class_names = getattr(model, 'names', None)
    
    # 检查输入路径是否存在
    if not os.path.exists(images_dir):
        print(f"❌ 路径不存在: {images_dir}")
        return
    
    # 创建 labels 目录，与 images 目录同级
    parent_dir = os.path.dirname(images_dir)
    labels_dir = os.path.join(parent_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"📁 输入图片目录: {images_dir}")
    print(f"📂 标签输出目录: {labels_dir}")
    
    # 支持的图片格式
    image_extensions = {".jpg", ".jpeg", ".png"}
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(images_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"❌ 目录 {images_dir} 中没有找到支持的图片文件")
        return
    
    print(f"📊 找到 {len(image_files)} 个图片文件需要处理")
    
    # 处理每个图片文件
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(images_dir, filename)
        print(f"🔄 正在处理 ({i}/{len(image_files)}): {filename}")
        
        # 进行推理
        results = model.predict(source=image_path, conf=confidence_threshold, verbose=False)
        
        # 生成标签文件名（与图片同名，但扩展名为.txt）
        name_without_ext = Path(filename).stem
        label_filename = f"{name_without_ext}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # 保存检测结果到标签文件
        save_detection_results(results, label_path, class_names)
        print(f"✅ 检测结果已保存至: {label_path}")

def save_detection_results(results, label_path, class_names):
    """
    将检测结果保存为YOLO格式的标签文件
    
    Args:
        results: 模型预测结果
        label_path (str): 标签文件保存路径
        class_names: 类别名称列表
    """
    with open(label_path, 'w') as f:
        for result in results:
            if result.boxes is not None:
                # 获取图像尺寸
                img_height, img_width = result.orig_shape
                
                # 遍历所有检测到的目标
                for i, box in enumerate(result.boxes):
                    # 获取类别ID
                    cls_id = int(box.cls)
                    
                    # 获取边界框坐标 (xywh格式)
                    x_center, y_center, width, height = box.xywh[0].cpu().numpy()
                    
                    # 归一化坐标
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height
                    
                    # 写入标签文件 (class_id x_center y_center width height)
                    f.write(f"{cls_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")

if __name__ == "__main__":
    # 修改为你的 best.pt 路径和输入图片目录路径
    model_path = "/home/jiahao.wu/DATACENTER1/basketball/codes/best.pt"
    images_directory = "/home/jiahao.wu/DATACENTER1/basketball/datasets/screenshot"
    confidence_threshold = 0.30
    
    run_detection_and_save_labels(model_path, images_directory, confidence_threshold=confidence_threshold)