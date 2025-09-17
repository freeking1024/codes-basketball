import cv2
import mediapipe as mp
import os

# 初始化MediaPipe Pose组件
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 用于绘制关键点和骨骼
mp_drawing_styles = mp.solutions.drawing_styles  # 提供默认绘制样式

def process_image(input_path, output_path="output_image.jpg"):
    """处理单张图像，检测并绘制姿态关键点"""
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图像: {input_path}")
        return

    # 转换为RGB格式（MediaPipe需要RGB输入）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 初始化Pose模型
    with mp_pose.Pose(
        static_image_mode=True,  # 静态图像模式（处理图片时设为True）
        model_complexity=2,  # 模型复杂度（0-2，越高精度越好但速度越慢）
        enable_segmentation=False,  # 不启用分割
        min_detection_confidence=0.5) as pose:  # 检测置信度阈值

        # 处理图像
        results = pose.process(image_rgb)

        # 绘制关键点和骨骼连接
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,  # 要绘制的图像
                results.pose_landmarks,  # 关键点数据
                mp_pose.POSE_CONNECTIONS,  # 骨骼连接关系
                # 关键点样式
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                # 骨骼线条样式
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

        # 保存结果图像
        cv2.imwrite(output_path, image)
        print(f"处理完成，结果保存至: {output_path}")

def process_video(input_path, output_path="output_video.mp4"):
    """处理视频，实时检测并绘制姿态关键点"""
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 初始化Pose模型（视频用动态模式）
    with mp_pose.Pose(
        static_image_mode=False,  # 动态模式（处理视频时设为False）
        model_complexity=1,  # 中等复杂度，平衡速度和精度
        smooth_landmarks=True,  # 平滑关键点（减少抖动）
        min_detection_confidence=0.5,  # 检测置信度阈值
        min_tracking_confidence=0.5) as pose:  # 跟踪置信度阈值

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"处理中... 已完成 {frame_count} 帧")

            # 转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # 提高性能

            # 处理帧
            results = pose.process(frame_rgb)

            # 恢复可写性并绘制关键点
            frame_rgb.flags.writeable = True
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )

            # 写入结果帧
            out.write(frame_bgr)

    # 释放资源
    cap.release()
    out.release()
    print(f"视频处理完成，结果保存至: {output_path}")

if __name__ == "__main__":
    # 输入文件路径（替换为你的图像或视频路径）
    input_file = "input.jpg"  # 或 "input.mp4"

    # 判断文件类型并处理
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
    elif input_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        process_image(input_file)
    elif input_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(input_file)
    else:
        print("不支持的文件格式")
    