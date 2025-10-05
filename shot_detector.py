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

# 投篮计数和冷却机制
shooting_count = 0
prev_shooting_state = False
cooldown_frames = 30  # 冷却帧数
cooldown_counter = 0

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

def process_video(input_video, output_video):
    global shooting_count, prev_shooting_state, cooldown_counter

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    last_ball = None
    last_person = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        det_results = det_model(frame)[0]
        balls, persons, rims = [], [], []

        for box in det_results.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(width-1,x2), min(height-1,y2)

            if cls==0:
                balls.append(((x1+x2)//2,(y1+y2)//2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),2)
            elif cls==2:
                persons.append((x1,y1,x2,y2))
            elif cls==3:
                rims.append((x1,y1,x2,y2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

        ball_center = balls[0] if balls else last_ball
        last_ball = ball_center

        # 找最近的人
        nearest_person = None
        if ball_center and persons:
            px,py = ball_center
            min_dist=float("inf")
            for (x1,y1,x2,y2) in persons:
                cx,cy = (x1+x2)//2, (y1+y2)//2
                dist=(px-cx)**2 + (py-cy)**2
                if dist<min_dist:
                    min_dist=dist
                    nearest_person=(x1,y1,x2,y2)
            last_person=nearest_person
        else:
            nearest_person = last_person

        is_shooting=False
        if nearest_person:
            x1,y1,x2,y2 = nearest_person
            if x2>x1 and y2>y1:
                person_roi = frame[y1:y2,x1:x2]
                if person_roi.size>0:
                    pose_results = pose_model(person_roi)[0]
                    if pose_results.keypoints is not None and len(pose_results.keypoints.data)>0:
                        keypoints = pose_results.keypoints.data[0].cpu().numpy()
                        is_shooting = is_shooting_pose(keypoints,0.5)

                        # 绘制关键点
                        for kp in keypoints:
                            xk,yk,conf = kp
                            if conf>0.5:
                                cv2.circle(frame,(int(xk)+x1,int(yk)+y1),3,(0,0,255),-1)

                        # 绘制骨架
                        if hasattr(pose_model,"yaml") and "skeleton" in pose_model.yaml:
                            skeleton = pose_model.yaml["skeleton"]
                            for i,j in skeleton:
                                if i<len(keypoints) and j<len(keypoints):
                                    x_i, y_i = int(keypoints[i][0])+x1, int(keypoints[i][1])+y1
                                    x_j, y_j = int(keypoints[j][0])+x1, int(keypoints[j][1])+y1
                                    cv2.line(frame,(x_i,y_i),(x_j,y_j),(0,255,0),2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        # 投篮计数逻辑，带冷却机制
        if is_shooting and not prev_shooting_state and cooldown_counter<=0:
            shooting_count += 1
            cooldown_counter = cooldown_frames
        prev_shooting_state = is_shooting
        if cooldown_counter>0:
            cooldown_counter -= 1

        # 绘制 Shoot 文字
        text = f"Shoot: {'YES' if is_shooting else 'NO'}"
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=1.0
        thickness=2
        text_color=(0,255,0) if is_shooting else (0,0,255)
        bg_color=(255,255,255)
        pad=5
        (tw,th),baseline=cv2.getTextSize(text,font,font_scale,thickness)
        x,y=10,th+10
        cv2.rectangle(frame,(x-pad,y-th-pad),(x+tw+pad,y+baseline+pad),bg_color,-1)
        cv2.putText(frame,text,(x,y),font,font_scale,text_color,thickness,cv2.LINE_AA)

        # 绘制投篮计数
        count_text = f"Shots: {shooting_count}"
        cv2.putText(frame,count_text,(10,th+40),font,0.8,(255,255,255),2,cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[Done] {output_video}, Total Shots: {shooting_count}")

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
    output_dir = "/home/jiahao.wu/DATACENTER1/basketball/output-shot"
    process_videos(input_dir, output_dir)
