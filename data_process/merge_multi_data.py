#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path

# 将标注后的数据集进行统一重命名，防止出现名字重复
def rename_files_in_directory(src_dir, dst_dir, core_folder="blockshot", prefix="blockshot"):
    """
    处理混合文件夹（图像和标签在同一文件夹中）
    自动创建以 core_folder 命名的核心文件夹（默认 blockshot），按前缀精准匹配
    """
    # 目标目录结构：dst_dir/blockshot/train/images 和 dst_dir/blockshot/train/labels
    images_dst_dir = Path(dst_dir) / core_folder / "train" / "images"
    labels_dst_dir = Path(dst_dir) / core_folder / "train" / "labels"
    images_dst_dir.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录（含blockshot）
    labels_dst_dir.mkdir(parents=True, exist_ok=True)

    # 校验源目录是否存在
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"错误：源目录 {src_dir} 不存在")
        return

    # 1. 分类并索引文件（按文件名前缀）
    image_index = {}  # key: 文件名前缀, value: 图像路径
    label_index = {}  # key: 文件名前缀, value: 标签路径
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    for file_path in src_path.iterdir():
        if not file_path.is_file():
            continue
        
        file_prefix = file_path.stem
        file_suffix = file_path.suffix.lower()
        if file_suffix in image_extensions:
            image_index[file_prefix] = file_path
        elif file_suffix == '.txt':
            label_index[file_prefix] = file_path

    # 2. 数量校验
    total_images = len(image_index)
    total_labels = len(label_index)
    print(f"扫描完成：找到 {total_images} 个图像文件，{total_labels} 个标签文件")

    if total_images == 0 and total_labels == 0:
        print("错误：源目录中未找到任何图像或标签文件")
        return
    if total_images == 0:
        print("错误：源目录中未找到任何图像文件")
        return
    if total_labels == 0:
        print("错误：源目录中未找到任何标签文件")
        return

    # 3. 匹配并复制文件
    matched_count = 0
    sorted_image_prefixes = sorted(image_index.keys())

    for idx, img_prefix in enumerate(sorted_image_prefixes, 1):
        img_file = image_index[img_prefix]
        if img_prefix in label_index:
            label_file = label_index[img_prefix]
            # 新文件名前缀与核心文件夹名一致（blockshot_01.jpg）
            img_new_name = f"{prefix}_{idx:02d}{img_file.suffix}"
            label_new_name = f"{prefix}_{idx:02d}{label_file.suffix}"
            # 复制到 blockshot 目录下
            img_dst_path = images_dst_dir / img_new_name
            label_dst_path = labels_dst_dir / label_new_name
            shutil.copy2(img_file, img_dst_path)
            shutil.copy2(label_file, label_dst_path)
            print(f"[{idx:02d}] 匹配成功：{img_file.name} → {img_new_name} | {label_file.name} → {label_new_name}")
            matched_count += 1
        else:
            print(f"[{idx:02d}] 警告：图像 {img_file.name} 未找到对应标签，跳过该文件")

    # 4. 未匹配标签提示
    unmatched_labels = [p for p in label_index.keys() if p not in image_index]
    if unmatched_labels:
        print(f"\n警告：有 {len(unmatched_labels)} 个标签文件无对应图像，未处理：")
        for p in unmatched_labels:
            print(f"  - {label_index[p].name}")

    # 最终结果提示（明确 blockshot 文件夹路径）
    print(f"\n处理完成！共成功匹配 {matched_count} 组文件，已保存到 {dst_dir}/{core_folder}/train")


def rename_files_in_separate_folders(images_src_dir, labels_src_dir, dst_dir, core_folder="blockshot", prefix="blockshot"):
    """
    处理分离文件夹（图像和标签在不同文件夹中）
    自动创建以 core_folder 命名的核心文件夹（默认 blockshot），按前缀精准匹配
    """
    # 目标目录结构：dst_dir/blockshot/train/images 和 dst_dir/blockshot/train/labels
    images_dst_dir = Path(dst_dir) / core_folder / "train" / "images"
    labels_dst_dir = Path(dst_dir) / core_folder / "train" / "labels"
    images_dst_dir.mkdir(parents=True, exist_ok=True)  # 自动创建 blockshot 及子目录
    labels_dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. 校验源目录
    images_src_path = Path(images_src_dir)
    labels_src_path = Path(labels_src_dir)
    if not images_src_path.exists():
        print(f"错误：图像源目录 {images_src_dir} 不存在")
        return
    if not labels_src_path.exists():
        print(f"错误：标签源目录 {labels_src_dir} 不存在")
        return

    # 2. 索引标签文件
    label_index = {}
    for label_file in labels_src_path.iterdir():
        if label_file.is_file() and label_file.suffix.lower() == '.txt':
            label_index[label_file.stem] = label_file
    total_labels = len(label_index)
    print(f"扫描标签目录：找到 {total_labels} 个标签文件")
    if total_labels == 0:
        print("错误：标签源目录中未找到任何标签文件")
        return

    # 3. 索引图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in images_src_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    total_images = len(image_files)
    print(f"扫描图像目录：找到 {total_images} 个图像文件")
    if total_images == 0:
        print("错误：图像源目录中未找到任何图像文件")
        return

    # 4. 匹配并复制文件
    matched_count = 0
    sorted_image_files = sorted(image_files, key=lambda x: x.stem)

    for idx, img_file in enumerate(sorted_image_files, 1):
        img_prefix = img_file.stem
        if img_prefix in label_index:
            label_file = label_index[img_prefix]
            # 新文件名前缀与 blockshot 文件夹一致
            img_new_name = f"{prefix}_{idx:02d}{img_file.suffix}"
            label_new_name = f"{prefix}_{idx:02d}{label_file.suffix}"
            img_dst_path = images_dst_dir / img_new_name
            label_dst_path = labels_dst_dir / label_new_name
            shutil.copy2(img_file, img_dst_path)
            shutil.copy2(label_file, label_dst_path)
            print(f"[{idx:02d}] 匹配成功：{img_file.name} → {img_new_name} | {label_file.name} → {label_new_name}")
            matched_count += 1
        else:
            print(f"[{idx:02d}] 警告：图像 {img_file.name} 未找到对应标签，跳过该文件")

    # 5. 未匹配标签提示
    unmatched_labels = [p for p in label_index.keys() if p not in [f.stem for f in sorted_image_files]]
    if unmatched_labels:
        print(f"\n警告：有 {len(unmatched_labels)} 个标签文件无对应图像，未处理：")
        for p in unmatched_labels:
            print(f"  - {label_index[p].name}")

    # 最终结果提示（明确 blockshot 路径）
    print(f"\n处理完成！共成功匹配 {matched_count} 组文件，已保存到 {dst_dir}/{core_folder}/train")


if __name__ == "__main__":
    # ==================== 配置参数（无需修改 blockshot 相关，已默认设置） ====================
    # 模式1：处理混合文件夹（如需使用，取消注释并修改路径）
    # src_directory = "/path/to/your/mixed/folder"  # 你的混合源目录
    # dst_directory = "/home/jiahao.wu/DATACENTER1/basketball/datasets/merge-data"  # 目标根目录
    # rename_files_in_directory(
    #     src_dir=src_directory,
    #     dst_dir=dst_directory,
    #     core_folder="blockshot",  # 核心文件夹名（固定为blockshot）
    #     prefix="blockshot"  # 文件名前缀（与文件夹名一致）
    # )

    # 模式2：处理分离文件夹（当前启用，匹配你的blockshot数据集路径）
    images_source_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shoot/images"  # 你的图像源目录
    labels_source_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shoot/labels"  # 你的标签源目录
    destination_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/merge-data"  # 目标根目录
    rename_files_in_separate_folders(
        images_src_dir=images_source_dir,
        labels_src_dir=labels_source_dir,
        dst_dir=destination_dir,
        core_folder="shoot",  # 自动创建的核心文件夹名（无需手动创建）
        prefix="shoot"  # 文件名前缀（如 blockshot_01.jpg，与文件夹名对应）
    )