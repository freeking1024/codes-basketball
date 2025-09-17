import sys
import cv2
import numpy as np
from ultralytics import YOLO
import math

# 关键点名称映射（对应YOLO11n-Pose的17个关键点）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
shooting_count = 0
prev_shooting_state = False
cooldown_frames = 60  # 冷却帧数，根据视频FPS调整（例如FPS=30时，1秒冷却时间）
cooldown_counter = 0
def calculate_angle(p1, p2, p3):
    """计算三点构成的角度（p2为顶点）"""
    if None in [p1, p2, p3]:
        return None
    # 转为向量
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    # 计算夹角余弦值
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    cos_angle = dot_product / (norm_v1 * norm_v2)
    # 确保值在[-1, 1]范围内（避免浮点误差）
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    # 转为角度
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def is_shooting_pose(keypoints, conf_threshold=0.5):
    """
    判断是否为投篮姿势
    基于以下特征：
    1. 投篮手（右利手为例）的肩膀、手肘、手腕角度（约90-130度）
    2. 手腕位置高于肩膀
    3. 非投篮手可能辅助托球（可选判断）
    """
    # 提取关键点位（过滤低置信度关键点）
    kp = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y, conf = keypoints[i]
        if conf > conf_threshold:
            kp[name] = (x, y)
        else:
            kp[name] = None

    # 检查必要关键点是否存在
    required_points = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder"]
    if any(kp[name] is None for name in required_points):
        return False, "Missing keypoints"

    # 计算投篮手臂角度（肩-肘-腕）
    arm_angle = calculate_angle(
        kp["right_shoulder"], 
        kp["right_elbow"], 
        kp["right_wrist"]
    )
    if arm_angle is None:
        return False, "Cannot calculate arm angle"

    # 计算躯干角度（左肩-右肩-右肘）
    torso_angle = calculate_angle(
        kp["left_shoulder"], 
        kp["right_shoulder"], 
        kp["right_elbow"]
    )
    if torso_angle is None:
        return False, "Cannot calculate torso angle"

    # 判断手腕是否高于肩膀
    wrist_above_shoulder = kp["right_wrist"][1] < kp["right_shoulder"][1]

    # 投篮姿势规则（可根据实际场景调整阈值）
    is_shooting = (
        70 < arm_angle < 140 and  # 手臂弯曲角度
        30 < torso_angle < 120 and  # 躯干与手臂角度
        wrist_above_shoulder  # 手腕高于肩膀
    )

    # 左利手补充判断（如果检测到左手更可能是投篮手）
    if kp["left_wrist"] and kp["left_elbow"]:
        left_arm_angle = calculate_angle(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
        left_wrist_above = kp["left_wrist"][1] < kp["left_shoulder"][1]
        if 70 < left_arm_angle < 140 and left_wrist_above:
            is_shooting = True

    return is_shooting, f"Arm angle: {arm_angle:.1f}°, Torso angle: {torso_angle:.1f}°, Wrist above shoulder: {wrist_above_shoulder}"

def draw_keypoints_only(image, result):
    """只绘制关键点和骨架，不绘制边界框和标签"""
    annotated_img = image.copy()
    
    if result.keypoints is not None:
        keypoints = result.keypoints.data.cpu().numpy()
        
        # COCO格式17点骨架连接
        skeleton = [
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [13, 15], [12, 14], [14, 16]
        ]
        
        for person_kps in keypoints:
            points = []
            for kp in person_kps:
                x, y, conf = kp
                if conf > 0.5:
                    point = (int(x), int(y))
                    points.append(point)
                    cv2.circle(annotated_img, point, 5, (0, 255, 0), -1)
                else:
                    points.append(None)
            
            for connection in skeleton:
                start_idx, end_idx = connection[0], connection[1]
                if points[start_idx] is not None and points[end_idx] is not None:
                    cv2.line(annotated_img, points[start_idx], points[end_idx], 
                             (255, 0, 0), 2)
    
    return annotated_img
def process_image(image_path, model):
    """处理单张图片并检测是否在投篮"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    results = model(img, conf=0.3)
    
    for result in results:
        if result.keypoints is not None:
            for keypoints in result.keypoints.data.cpu().numpy():
                is_shooting, reason = is_shooting_pose(keypoints)
                
                # 使用自定义绘制函数，只绘制关键点
                annotated_img = draw_keypoints_only(img, result)
                
                # 添加文字标注
                text = f"Shoot: {is_shooting}"
                cv2.putText(
                    annotated_img, 
                    text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0) if is_shooting else (0, 0, 255), 
                    2
                )
                cv2.putText(
                    annotated_img, 
                    reason, 
                    (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 0), 
                    1
                )
                
                cv2.imwrite("shooting_detection_result.jpg", annotated_img)
                print(f"Result saved as: shooting_detection_result.jpg")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, model):
    """处理视频并实时检测投篮动作"""
    global shooting_count, prev_shooting_state, cooldown_counter
    
    cap = cv2.VideoCapture(video_path if video_path else 0)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        "shooting_detection_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=0.3, stream=True)
        
        for result in results:
            # 使用自定义绘制函数，只绘制关键点
            annotated_frame = draw_keypoints_only(frame, result)
            
            if result.keypoints is not None:
                for keypoints in result.keypoints.data.cpu().numpy():
                    is_shooting, reason = is_shooting_pose(keypoints)
                    
                    # 投篮计数逻辑，带有冷却机制
                    if is_shooting and not prev_shooting_state and cooldown_counter <= 0:
                        shooting_count += 1
                        cooldown_counter = cooldown_frames  # 启动冷却计时器
                    
                    prev_shooting_state = is_shooting
                    
                    # 冷却计时器递减
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                    
                    text = f"Shoot: {is_shooting}"
                    cv2.putText(
                        annotated_frame, 
                        text, 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0) if is_shooting else (0, 0, 255), 
                        2
                    )
                    # 显示投篮计数
                    count_text = f"Shots: {shooting_count}"
                    
                    cv2.putText(
                        annotated_frame,
                        count_text,
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )
                    
                    # 显示冷却状态（调试用）
                    cooldown_text = f"Cooldown: {cooldown_counter}"
                    cv2.putText(
                        annotated_frame,
                        cooldown_text,
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1
                    )
                    
                    cv2.putText(
                        annotated_frame, 
                        reason, 
                        (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 0), 
                        1
                    )
        
        out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video result saved as: shooting_detection_video.mp4")
    print(f"Total shots detected: {shooting_count}")

if __name__ == "__main__":
    # 加载模型
    model = YOLO("yolo11n-pose.pt")
    
    # 处理图片（替换为你的图片路径）
    # process_image("basketball_player.jpg", model)
    
    # 处理视频（替换为你的视频路径，或使用0调用摄像头）
    process_video("/home/jiahao.wu/DATACENTER1/basketball/data/single.mp4", model)