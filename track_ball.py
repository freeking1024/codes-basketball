import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

def process_video(input_path, output_path="two_stage_pose_result.mp4"):
    # 1. 加载两个模型
    det_model = YOLO("person-ball.pt")      # 目标检测模型：篮球/篮筐/人
    pose_model = YOLO("yolo11n-pose.pt")    # 姿态估计模型：人体关键点

    # 2. 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 存储篮球轨迹点
    ball_history = deque(maxlen=200)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result_frame = frame.copy()

        # ---------------- 第一阶段：目标检测 ----------------
        det_results = det_model(frame)
        human_boxes, rim_boxes = [], []

        for result in det_results:
            for box in result.boxes:
                if box.conf[0] < 0.5:  # 置信度过滤
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                if cls == 2:  # person
                    human_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                elif cls == 0:  # basketball
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    ball_history.append((cx, cy))
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(result_frame, (cx, cy), 4, (0, 0, 255), -1)

                elif cls == 3:  # rim
                    rim_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # ---------------- 第二阶段：人体姿态估计 ----------------
        for (x1, y1, x2, y2) in human_boxes:
            expand_ratio = 0.1
            w, h = x2 - x1, y2 - y1
            x1e = max(0, int(x1 - w * expand_ratio))
            y1e = max(0, int(y1 - h * expand_ratio))
            x2e = min(width, int(x2 + w * expand_ratio))
            y2e = min(height, int(y2 + h * expand_ratio))
            human_roi = frame[y1e:y2e, x1e:x2e]

            if human_roi.size == 0:
                continue

            pose_results = pose_model(human_roi, conf=0.4)

            for result in pose_results:
                if result.keypoints is None or len(result.keypoints.xy) == 0:
                    continue
                keypoints = result.keypoints.xy[0].cpu().numpy()

                # 绘制骨骼
                if hasattr(pose_model, 'yaml') and 'skeleton' in pose_model.yaml:
                    for a, b in pose_model.yaml['skeleton']:
                        if a < len(keypoints) and b < len(keypoints):
                            xa, ya, *ca = keypoints[a]
                            xb, yb, *cb = keypoints[b]
                            conf_a = ca[0] if ca else 1.0
                            conf_b = cb[0] if cb else 1.0
                            if conf_a > 0.5 and conf_b > 0.5:
                                xa, ya = int(x1e + xa * (x2e - x1e) / human_roi.shape[1]), int(y1e + ya * (y2e - y1e) / human_roi.shape[0])
                                xb, yb = int(x1e + xb * (x2e - x1e) / human_roi.shape[1]), int(y1e + yb * (y2e - y1e) / human_roi.shape[0])
                                cv2.line(result_frame, (xa, ya), (xb, yb), (255, 150, 50), 2, cv2.LINE_AA)

                # 绘制关键点
                for kp in keypoints:
                    x, y = kp[:2]
                    conf = kp[2] if len(kp) >= 3 else 1.0
                    if conf > 0.5:
                        ox = int(x1e + x * (x2e - x1e) / human_roi.shape[1])
                        oy = int(y1e + y * (y2e - y1e) / human_roi.shape[0])
                        cv2.circle(result_frame, (ox, oy), 4, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.circle(result_frame, (ox, oy), 2, (0, 100, 255), -1, cv2.LINE_AA)

        # ---------------- 篮球轨迹绘制 ----------------
        if len(ball_history) > 1:
            for i in range(1, len(ball_history)):
                cv2.line(result_frame, ball_history[i-1], ball_history[i], (0, 0, 255), 2)

        # 抛物线拟合
        if len(ball_history) >= 5:
            pts = np.array(ball_history)
            xs, ys = pts[:, 0], pts[:, 1]
            coeffs = np.polyfit(xs, ys, 2)  # y = ax^2 + bx + c
            a, b, c = coeffs

            x_new = np.linspace(xs.min(), xs.max(), 100)
            y_new = a * x_new**2 + b * x_new + c
            for i in range(len(x_new) - 1):
                pt1 = (int(x_new[i]), int(y_new[i]))
                pt2 = (int(x_new[i+1]), int(y_new[i+1]))
                cv2.line(result_frame, pt1, pt2, (0, 255, 0), 2)

            # 顶点
            x_vertex = -b / (2 * a)
            y_vertex = a * x_vertex**2 + b * x_vertex + c
            cv2.circle(result_frame, (int(x_vertex), int(y_vertex)), 6, (255, 0, 0), -1)
            cv2.putText(result_frame, "Apex", (int(x_vertex)+5, int(y_vertex)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        out.write(result_frame)

    cap.release()
    out.release()
    print(f"处理完成，结果保存至: {output_path}")


if __name__ == "__main__":
    input_video = "/home/jiahao.wu/DATACENTER1/basketball/data/f0f3b7dc83af72dcb4cb8589dfd38564.mp4"
    if os.path.exists(input_video):
        process_video(input_video)
    else:
        print(f"视频文件不存在: {input_video}")
