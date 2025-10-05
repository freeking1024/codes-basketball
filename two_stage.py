import cv2
import os
import numpy as np
from ultralytics import YOLO

# 加载模型
det_model = YOLO("person-ball.pt")   # 检测模型: person(2), ball(0), rim(3)
pose_model = YOLO("yolo11n-pose.pt") # 姿态模型

KEYPOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

def calculate_angle(p1,p2,p3):
    if None in [p1,p2,p3]:
        return None
    v1 = np.array(p1)-np.array(p2)
    v2 = np.array(p3)-np.array(p2)
    dot = np.dot(v1,v2)
    norm = np.linalg.norm(v1)*np.linalg.norm(v2)
    if norm<1e-6:
        return None
    cos_angle = np.clip(dot/norm,-1,1)
    return np.degrees(np.arccos(cos_angle))

def is_shooting_pose(keypoints, conf_thr=0.5):
    kp = {}
    for i,name in enumerate(KEYPOINT_NAMES):
        if i>=len(keypoints):
            kp[name]=None
            continue
        x,y,conf = keypoints[i]
        kp[name]=(x,y) if conf>conf_thr else None

    # 优先右臂
    if kp["right_shoulder"] and kp["right_elbow"] and kp["right_wrist"]:
        angle = calculate_angle(kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"])
        wrist_above = kp["right_wrist"][1]<kp["right_shoulder"][1]
        return angle is not None and 70<angle<140 and wrist_above
    elif kp["left_shoulder"] and kp["left_elbow"] and kp["left_wrist"]:
        angle = calculate_angle(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
        wrist_above = kp["left_wrist"][1]<kp["left_shoulder"][1]
        return angle is not None and 70<angle<140 and wrist_above
    return False

def distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return np.linalg.norm(np.array(p1)-np.array(p2))

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    last_person = None

    # ---- 投篮计数器 ----
    shoot_count = 0
    prev_shooting = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        det_results = det_model(frame)[0]
        persons = []

        for box in det_results.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 2:  # person
                persons.append((x1, y1, x2, y2))

        # 取第一个人（或者保留上一次的人）
        nearest_person = persons[0] if persons else last_person
        last_person = nearest_person

        is_shooting = False
        if nearest_person:
            x1, y1, x2, y2 = map(int, nearest_person)
            if x2 > x1 and y2 > y1:
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size > 0:
                    pose_results = pose_model(person_roi)[0]
                    if pose_results.keypoints is not None and len(pose_results.keypoints.data) > 0:
                        keypoints = pose_results.keypoints.data[0].cpu().numpy()
                        is_shooting = is_shooting_pose(keypoints, 0.5)

                        # 绘制关键点
                        for kp in keypoints:
                            xk, yk, conf = kp
                            if conf > 0.5:
                                cv2.circle(frame, (int(xk) + x1, int(yk) + y1), 3, (0, 0, 255), -1)

                        # 绘制骨架
                        if hasattr(pose_model, "yaml") and "skeleton" in pose_model.yaml:
                            skeleton = pose_model.yaml["skeleton"]
                            for i, j in skeleton:
                                if i < len(keypoints) and j < len(keypoints):
                                    x_i, y_i = int(keypoints[i][0]) + x1, int(keypoints[i][1]) + y1
                                    x_j, y_j = int(keypoints[j][0]) + x1, int(keypoints[j][1]) + y1
                                    cv2.line(frame, (x_i, y_i), (x_j, y_j), (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ---- 计数逻辑：只基于姿势 ----
        if not prev_shooting and is_shooting:
            shoot_count += 1
            print(f"[Count] 投篮次数 +1, 当前总数: {shoot_count}")

        prev_shooting = is_shooting

        # ---- 显示文字 ----
        text = f"Shoot: {'YES' if is_shooting else 'NO'} | Count: {shoot_count}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        text_color = (0, 255, 0) if is_shooting else (0, 0, 255)
        bg_color = (255, 255, 255)
        pad = 5
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = 10, th + 10
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[Done] {output_video}, 投篮总数: {shoot_count}")


def process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4",".avi",".mov")):
            input_path = os.path.join(input_dir,file)
            output_path = os.path.join(output_dir,file.rsplit(".",1)[0]+"_out.mp4")
            print(f"Processing {file} -> {output_path}")
            process_video(input_path, output_path)

if __name__=="__main__":
    input_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shot-split/test-video/first_batch/part"
    output_dir = "/home/jiahao.wu/DATACENTER1/basketball/output"
    process_videos(input_dir, output_dir)
