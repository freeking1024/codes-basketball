import cv2
import os
import numpy as np
from ultralytics import YOLO

# 加载两个模型
pose_model = YOLO("yolo11n-pose.pt")
det_model = YOLO("person-ball.pt")

# 输入视频文件夹
input_dir = "/home/jiahao.wu/DATACENTER1/basketball/codes/f0f3b7dc83af72dcb4cb8589dfd38564"
output_dir = "/home/jiahao.wu/DATACENTER1/basketball/codes/result"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.endswith((".mp4", ".avi", ".mov")):
        continue

    input_path = os.path.join(input_dir, filename)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{name}_out{ext}")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 人体骨骼
        pose_results = pose_model(frame, verbose=False)
        for r in pose_results:
            kpts = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else []
            for pt in kpts:
                for x, y in pt:
                    cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

        # 目标检测 (篮球、人、篮筐等)
        det_results = det_model(frame, verbose=False)
        for r in det_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), cls in zip(boxes, classes):
                if cls == 0:  # 篮球
                    cx, cy = (int(x1+x2)//2, int(y1+y2)//2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                    cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"处理完成: {output_path}")
