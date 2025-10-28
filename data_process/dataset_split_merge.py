#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
from pathlib import Path
# 针对merge_multi_data.py  文件处理后的数据集进行train和val的划分，直接可用于yolo模型的训练
def merge_and_split_datasets(
    source_root_dir,       # 多类别数据的根目录（即 merge-data）
    target_root_dir,       # 拆分后训练集/测试集的根目录
    train_ratio=0.8,       # 训练集比例（默认8:2）
    random_seed=42         # 随机种子（保证拆分结果可复现）
):
    """
    1. 合并 source_root_dir 下所有类别的 images/labels 数据
    2. 按 8:2 原则拆分训练集/测试集，保持图像与标签一一对应
    """
    # -------------------------- 步骤1：初始化路径与参数 --------------------------
    # 1.1 定义目标目录结构（最终会生成 train/images、train/labels、val/images、val/labels）
    target_train_img = Path(target_root_dir) / "train" / "images"
    target_train_lab = Path(target_root_dir) / "train" / "labels"
    target_val_img = Path(target_root_dir) / "val" / "images"
    target_val_lab = Path(target_root_dir) / "val" / "labels"
    
    # 自动创建所有目标目录
    for dir_path in [target_train_img, target_train_lab, target_val_img, target_val_lab]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # 1.2 定义支持的图像格式（与之前脚本保持一致）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    all_image_paths = []  # 存储所有收集到的图像路径


    # -------------------------- 步骤2：收集所有类别的数据 --------------------------
    print(f"开始收集 {source_root_dir} 下所有类别的数据...")
    
    # 遍历 source_root_dir 下的所有类别文件夹（如 blockshot、layup 等）
    for category_dir in Path(source_root_dir).iterdir():
        if not category_dir.is_dir():
            continue  # 跳过非文件夹（如可能的隐藏文件）
        
        # 每个类别文件夹下需符合 "类别名/train/images" 和 "类别名/train/labels" 结构（与之前脚本输出一致）
        category_img_dir = category_dir / "train" / "images"
        category_lab_dir = category_dir / "train" / "labels"
        
        # 校验当前类别文件夹的结构是否正确
        if not category_img_dir.exists() or not category_lab_dir.exists():
            print(f"警告：{category_dir.name} 目录结构异常（缺少 train/images 或 train/labels），跳过该类别")
            continue
        
        # 收集当前类别的所有图像（标签会通过图像前缀自动匹配）
        for img_path in category_img_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                # 检查对应标签是否存在（避免后续拆分后缺标签）
                lab_path = category_lab_dir / f"{img_path.stem}.txt"
                if lab_path.exists():
                    all_image_paths.append(img_path)
                else:
                    print(f"警告：{img_path.name} 无对应标签，跳过该图像")
    
    # 校验收集结果
    total_data_count = len(all_image_paths)
    if total_data_count == 0:
        print("错误：未收集到任何有效数据（检查类别目录结构或图像标签匹配情况）")
        return
    print(f"数据收集完成！共收集到 {total_data_count} 组有效数据（图像+标签）")


    # -------------------------- 步骤3：按8:2拆分训练集/测试集 --------------------------
    # 设置随机种子（保证每次运行拆分结果一致，便于复现）
    random.seed(random_seed)
    # 打乱所有数据的顺序（避免同类数据集中在某一部分）
    random.shuffle(all_image_paths)
    
    # 计算训练集与测试集的分割点
    train_split_idx = int(total_data_count * train_ratio)
    train_image_paths = all_image_paths[:train_split_idx]  # 前80%为训练集
    val_image_paths = all_image_paths[train_split_idx:]    # 后20%为测试集（val）
    
    print(f"\n拆分完成：训练集 {len(train_image_paths)} 组，测试集 {len(val_image_paths)} 组")


    # -------------------------- 步骤4：复制数据到目标目录 --------------------------
    def copy_data(image_paths, target_img_dir, target_lab_dir, data_type):
        """辅助函数：将指定的图像和对应标签复制到目标目录（data_type 为 "训练集" 或 "测试集"）"""
        for idx, img_path in enumerate(image_paths, 1):
            # 1. 复制图像
            img_dst = target_img_dir / img_path.name
            shutil.copy2(img_path, img_dst)
            
            # 2. 复制对应标签（标签路径通过图像路径推导：图像所在目录的上级目录的 labels 文件夹）
            lab_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
            lab_dst = target_lab_dir / f"{img_path.stem}.txt"
            shutil.copy2(lab_path, lab_dst)
            
            # 每复制100组打印一次进度（避免日志过多）
            if idx % 100 == 0 or idx == len(image_paths):
                print(f"{data_type}复制进度：{idx}/{len(image_paths)} 组")

    # 复制训练集数据
    print("\n开始复制训练集数据...")
    copy_data(train_image_paths, target_train_img, target_train_lab, "训练集")
    
    # 复制测试集数据
    print("\n开始复制测试集数据...")
    copy_data(val_image_paths, target_val_img, target_val_lab, "测试集")


    # -------------------------- 步骤5：输出最终结果 --------------------------
    print(f"\n所有操作完成！")
    print(f"训练集路径：{target_train_img}（图像）、{target_train_lab}（标签）")
    print(f"测试集路径：{target_val_img}（图像）、{target_val_lab}（标签）")
    print(f"训练集数量：{len(train_image_paths)} 组，测试集数量：{len(val_image_paths)} 组")


if __name__ == "__main__":
    # ==================== 请根据你的实际路径修改以下参数 ====================
    # 1. source_root_dir：多类别数据的根目录（即你之前的 merge-data，里面包含 blockshot、layup 等类别文件夹）
    source_root_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/merge-data"
    # 2. target_root_dir：拆分后训练集/测试集的保存目录（建议新建一个目录，如 "final_dataset"）
    target_root_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/final_train_dataset"
    
    # 调用函数执行合并与拆分（默认8:2拆分，如需调整比例可修改 train_ratio 参数，如 train_ratio=0.7 表示7:3）
    merge_and_split_datasets(
        source_root_dir=source_root_dir,
        target_root_dir=target_root_dir,
        train_ratio=0.8,  # 可按需调整（如0.7代表7:3拆分）
        random_seed=42    # 随机种子（保持默认即可，保证拆分结果可复现）
    )