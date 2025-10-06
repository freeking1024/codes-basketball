# 训练集测试集划分脚本
# 将 shoot 文件夹下的数据按照指定比例拆分为训练集和验证集
import os
import shutil
import random
from pathlib import Path

def split_dataset(shoot_dir, train_ratio=0.8):
    """
    将 shoot 文件夹下的数据按照指定比例拆分为训练集和验证集
    
    Args:
        shoot_dir (str): shoot 文件夹路径
        train_ratio (float): 训练集比例，默认为 0.8 (80%)
    """
    
    # 定义源文件夹路径
    images_dir = os.path.join(shoot_dir, "images")
    labels_dir = os.path.join(shoot_dir, "labels")
    
    # 定义目标文件夹路径
    train_dir = os.path.join(shoot_dir, "train")
    valid_dir = os.path.join(shoot_dir, "valid")
    
    # 检查源文件夹是否存在
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ 源文件夹不存在: {images_dir} 或 {labels_dir}")
        return
    
    # 创建训练集和验证集目录结构
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")
    valid_images_dir = os.path.join(valid_dir, "images")
    valid_labels_dir = os.path.join(valid_dir, "labels")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    
    print(f"📁 训练集目录: {train_dir}")
    print(f"📁 验证集目录: {valid_dir}")
    
    # 获取所有图片文件
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = []
    
    for file in os.listdir(images_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    if not image_files:
        print(f"❌ {images_dir} 中没有找到图片文件")
        return
    
    # 随机打乱文件列表
    random.shuffle(image_files)
    
    # 计算训练集和验证集的数量
    total_count = len(image_files)
    train_count = int(total_count * train_ratio)
    
    # 拆分文件列表
    train_files = image_files[:train_count]
    valid_files = image_files[train_count:]
    
    print(f"📊 总文件数: {total_count}")
    print(f"📊 训练集文件数: {len(train_files)}")
    print(f"📊 验证集文件数: {len(valid_files)}")
    
    # 复制训练集文件
    print("\n🔄 正在复制训练集文件...")
    copy_files(images_dir, labels_dir, train_images_dir, train_labels_dir, train_files)
    
    # 复制验证集文件
    print("\n🔄 正在复制验证集文件...")
    copy_files(images_dir, labels_dir, valid_images_dir, valid_labels_dir, valid_files)
    
    print(f"\n✅ 数据拆分完成!")
    print(f"   训练集: {len(train_files)} 个文件")
    print(f"   验证集: {len(valid_files)} 个文件")

def copy_files(src_images_dir, src_labels_dir, dst_images_dir, dst_labels_dir, file_list):
    """
    复制图片和标签文件到目标目录
    
    Args:
        src_images_dir (str): 源图片目录
        src_labels_dir (str): 源标签目录
        dst_images_dir (str): 目标图片目录
        dst_labels_dir (str): 目标标签目录
        file_list (list): 要复制的文件列表
    """
    
    for filename in file_list:
        # 构造源文件路径
        src_image_path = os.path.join(src_images_dir, filename)
        
        # 构造目标文件路径
        dst_image_path = os.path.join(dst_images_dir, filename)
        
        # 复制图片文件
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
        else:
            print(f"⚠️  图片文件不存在: {src_image_path}")
            continue
        
        # 处理对应的标签文件
        name_without_ext = Path(filename).stem
        label_filename = f"{name_without_ext}.txt"
        
        src_label_path = os.path.join(src_labels_dir, label_filename)
        dst_label_path = os.path.join(dst_labels_dir, label_filename)
        
        # 复制标签文件
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"⚠️  标签文件不存在: {src_label_path}")

# 使用示例
if __name__ == "__main__":
    # 修改为你的 shoot 文件夹路径
    shoot_directory = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shoot"  # 替换为实际路径
    
    # 执行数据拆分 (80% 训练集, 20% 验证集)
    split_dataset(shoot_directory, train_ratio=0.8)