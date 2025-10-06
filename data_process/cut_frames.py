# 视频切帧脚本
import os
import cv2
from pathlib import Path

def cut_frames_from_videos(video_dir, output_base_dir="output"):
    """
    对指定路径下的所有视频文件进行切帧处理
    
    Args:
        video_dir (str): 包含视频文件的目录路径
        output_base_dir (str): 输出基础目录路径
    """
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    # 遍历目录下的所有文件
    for filename in os.listdir(video_dir):
        file_path = os.path.join(video_dir, filename)
        
        # 检查是否为文件且是视频格式
        if os.path.isfile(file_path):
            # 获取文件扩展名并转换为小写
            _, ext = os.path.splitext(filename)
            
            if ext.lower() in video_extensions:
                # 根据文件名创建对应文件夹
                video_name = os.path.splitext(filename)[0]
                video_output_dir = os.path.join(output_base_dir, video_name, "images")
                
                # 创建输出目录（如果不存在）
                os.makedirs(video_output_dir, exist_ok=True)
                
                # 对视频进行切帧
                extract_frames(file_path, video_output_dir)
                print(f"完成视频 {filename} 的切帧，保存至 {video_output_dir}")

def extract_frames(video_path, output_dir, frame_interval=3):
    """
    从视频中提取帧并保存为图片
    
    Args:
        video_path (str): 视频文件路径
        output_dir (str): 输出图片的目录
        frame_interval (int): 帧间隔（默认每3帧保存一次）
    """
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 每隔一定帧数保存一次图片
        if frame_count % frame_interval == 0:
            # 生成图片文件名
            image_name = f"frame_{saved_count:06d}.jpg"
            image_path = os.path.join(output_dir, image_name)
            
            # 保存图片
            cv2.imwrite(image_path, frame)
            saved_count += 1
            
            print(f"保存图片: {image_name}")
        
        frame_count += 1
    
    # 释放视频捕获对象
    cap.release()
    print(f"总共处理 {frame_count} 帧，保存 {saved_count} 张图片")

# 使用示例
if __name__ == "__main__":
    # 指定视频目录路径
    video_directory = "/home/jiahao.wu/DATACENTER1/basketball/datasets/videos/dataset"  # 替换为你的视频目录路径
    
    # 执行切帧操作
    cut_frames_from_videos(video_directory)