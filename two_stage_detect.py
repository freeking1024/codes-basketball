import cv2
import torch
import os
from ultralytics import YOLO

# 加载两个模型
det_model = YOLO("person-ball.pt")   # 你自训的检测模型: person(2), ball(0), rim(3)
pose_model = YOLO("yolo11n-pose.pt") # 官方姿态模型

def process_video(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    last_ball = None   # 存储上一次的球位置
    last_person = None # 存储上一次的最近人关键点

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        det_results = det_model(frame)[0]

        balls = []
        persons = []
        rims = []

        # 遍历检测框
        for box in det_results.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls == 0:  # basketball
                balls.append(((x1+x2)//2, (y1+y2)//2))  # 球心
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,165,255), 2) # 球框(橙色)
            elif cls == 2:  # person
                persons.append((x1, y1, x2, y2))
            elif cls == 3:  # rim
                rims.append((x1,y1,x2,y2))
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2) # 篮筐框(黄色)

        # 决定球的位置
        if len(balls) > 0:
            ball_center = balls[0]  # 取第一个球
            last_ball = ball_center
        else:
            ball_center = last_ball  # 没检测到就用上一次

        # 找最近的人
        if ball_center is not None and len(persons) > 0:
            px, py = ball_center
            min_dist = float("inf")
            nearest_person = None
            for (x1,y1,x2,y2) in persons:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                dist = (px-cx)**2 + (py-cy)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_person = (x1,y1,x2,y2)
            # 更新 last_person
            last_person = nearest_person
        else:
            nearest_person = last_person  # 没球时保留上一次的人

        # 姿态估计
        if nearest_person is not None:
            x1, y1, x2, y2 = nearest_person
            person_roi = frame[y1:y2, x1:x2]
            pose_results = pose_model(person_roi)[0]

            for kpts in pose_results.keypoints.xy:
                for (x, y) in kpts:
                    cv2.circle(frame, (int(x)+x1, int(y)+y1), 3, (0,0,255), -1)

            # 画骨架
            if hasattr(pose_model, "names") and "skeleton" in pose_model.yaml:
                skeleton = pose_model.yaml["skeleton"]
                for conn in skeleton:
                    if len(conn) == 2:
                        i, j = conn
                        if i < len(kpts) and j < len(kpts):
                            x_i, y_i = int(kpts[i][0])+x1, int(kpts[i][1])+y1
                            x_j, y_j = int(kpts[j][0])+x1, int(kpts[j][1])+y1
                            cv2.line(frame, (x_i, y_i), (x_j, y_j), (0,255,0), 2)

            # 画人框
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

        out.write(frame)

    cap.release()
    out.release()


def process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith((".mp4", ".avi", ".mov")):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.rsplit(".",1)[0] + "_out.mp4")
            print(f"Processing {file} -> {output_path}")
            process_video(input_path, output_path)


if __name__ == "__main__":
    input_dir = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shot-split/f0f3b7dc83af72dcb4cb8589dfd38564"   # 输入视频文件夹
    output_dir = "/home/jiahao.wu/DATACENTER1/basketball/output" # 输出文件夹
    process_videos(input_dir, output_dir)
