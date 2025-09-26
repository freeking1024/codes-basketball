import sys
import cv2
import numpy as np
import os
import time
import requests
from datetime import datetime
from ultralytics import YOLO
import math
from moviepy.editor import VideoFileClip

# 关键点名称映射（对应YOLO11n-Pose的17个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

shooting_count = 0
prev_shooting_state = False
cooldown_frames = 60  # 冷却帧数，根据视频FPS调整
cooldown_counter = 0
processed_shots = set()  # 记录已处理的投篮时刻，避免重复截取


def calculate_angle(p1, p2, p3):
    """计算三点构成的角度（p2为顶点）"""
    if None in [p1, p2, p3]:
        return None
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def is_shooting_pose(keypoints, conf_threshold=0.5):
    """判断是否为投篮姿势"""
    kp = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y, conf = keypoints[i]
        kp[name] = (x, y) if conf > conf_threshold else None

    # 检查必要关键点
    required_points = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder"]
    if any(kp[name] is None for name in required_points):
        return False

    # 计算右臂角度
    arm_angle = calculate_angle(kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"])
    if arm_angle is None:
        return False

    # 计算躯干角度
    torso_angle = calculate_angle(kp["left_shoulder"], kp["right_shoulder"], kp["right_elbow"])
    if torso_angle is None:
        return False

    # 手腕高于肩膀
    wrist_above_shoulder = kp["right_wrist"][1] < kp["right_shoulder"][1]

    # 右利手判断
    is_shooting = 70 < arm_angle < 140 and 30 < torso_angle < 120 and wrist_above_shoulder

    # 左利手补充判断
    if kp["left_wrist"] and kp["left_elbow"]:
        left_arm_angle = calculate_angle(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
        left_wrist_above = kp["left_wrist"][1] < kp["left_shoulder"][1]
        if 70 < left_arm_angle < 140 and left_wrist_above:
            is_shooting = True

    return is_shooting


def extract_video_segment(input_path, output_path, start_time, duration=6):
    """截取视频片段（前3秒到后3秒）"""
    try:
        actual_start = max(0, start_time - 3)
        with VideoFileClip(input_path) as video:
            # 确保截取不超过视频时长
            end_time = min(actual_start + duration, video.duration)
            clip = video.subclip(actual_start, end_time)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"成功截取: {output_path}")
        return True
    except Exception as e:
        print(f"截取失败: {str(e)}")
        return False


def get_video_name_from_url(url):
    """从URL提取视频名称（不含扩展名）"""
    filename = os.path.basename(url)
    return os.path.splitext(filename)[0]


def process_remote_video(http_url, model):
    """处理远程HTTP视频流，检测投篮并截取片段"""
    global shooting_count, prev_shooting_state, cooldown_counter, processed_shots
    
    # 创建存储目录
    video_name = get_video_name_from_url(http_url)
    output_dir = os.path.join(os.getcwd(), video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"视频片段将保存至: {output_dir}")

    # 打开视频流
    cap = cv2.VideoCapture(http_url)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps  # 当前视频时间（秒）
        
        # 检测人体关键点
        results = model(frame, conf=0.3, stream=True)
        
        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.data.cpu().numpy():
                    is_shooting = is_shooting_pose(keypoints)
                    
                    # 投篮检测与计数逻辑
                    if is_shooting and not prev_shooting_state and cooldown_counter <= 0:
                        shooting_count += 1
                        cooldown_counter = cooldown_frames
                        
                        # 防重复截取
                        time_key = round(current_time, 1)
                        if time_key not in processed_shots:
                            processed_shots.add(time_key)
                            
                            # 生成输出路径
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = os.path.join(
                                output_dir, 
                                f"shot_{shooting_count}_{timestamp}.mp4"
                            )
                            
                            # 截取视频片段
                            print(f"检测到第{shooting_count}次投篮，开始截取...")
                            extract_video_segment(http_url, output_path, current_time)
                    
                    prev_shooting_state = is_shooting
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"处理完成，共检测到 {shooting_count} 次投篮")


def process_local_video(video_path, model):
    """处理本地视频文件"""
    global shooting_count, prev_shooting_state, cooldown_counter, processed_shots
    
    # 创建存储目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"视频片段将保存至: {output_dir}")

    # 打开本地视频
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        results = model(frame, conf=0.3, stream=True)
        
        for result in results:
            if result.keypoints is not None:
                for keypoints in result.keypoints.data.cpu().numpy():
                    is_shooting = is_shooting_pose(keypoints)
                    
                    if is_shooting and not prev_shooting_state and cooldown_counter <= 0:
                        shooting_count += 1
                        cooldown_counter = cooldown_frames
                        
                        time_key = round(current_time, 1)
                        if time_key not in processed_shots:
                            processed_shots.add(time_key)
                            
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = os.path.join(
                                output_dir, 
                                f"shot_{shooting_count}_{timestamp}.mp4"
                            )
                            
                            print(f"检测到第{shooting_count}次投篮，开始截取...")
                            extract_video_segment(video_path, output_path, current_time)
                    
                    prev_shooting_state = is_shooting
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"处理完成，共检测到 {shooting_count} 次投篮")


if __name__ == "__main__":
    model = YOLO("yolo11n-pose.pt")
    
    # 处理远程视频示例
    remote_url = "/home/jiahao.wu/DATACENTER1/basketball/data/f0f3b7dc83af72dcb4cb8589dfd38564.mp4"
    process_remote_video(remote_url, model)
    
    # 处理本地视频示例
    # local_path = "/home/jiahao.wu/DATACENTER1/basketball/data/single.mp4"
    # process_local_video(local_path, model)