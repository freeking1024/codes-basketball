import cv2
import os
import numpy as np
from ultralytics import YOLO

def process_video(input_path, output_path):
    # 1. 加载模型
    det_model = YOLO("person-ball.pt")   # 你训练的检测模型
    pose_model = YOLO("yolo11n-pose.pt") # 姿态估计模型
    
    # 2. 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[{os.path.basename(input_path)}] 已完成 {frame_count} 帧")

        result_frame = frame.copy()
        
        # 3. 第一阶段：检测人体、篮球和篮筐
        det_results = det_model(frame)
        human_boxes = []
        
        for result in det_results:
            for box in result.boxes:
                if box.conf[0] > 0.5:  # 置信度过滤
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])  # 类别ID
            
                    if cls == 2:  # person
                        human_boxes.append((x1, y1, x2, y2))
                        # 绘制人体检测框
                        overlay = result_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.2, result_frame, 0.8, 0, result_frame)
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        
                    elif cls == 0:  # basketball
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                    elif cls == 3:  # rim
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 4. 第二阶段：对每个检测到的人进行姿态估计
        for (x1, y1, x2, y2) in human_boxes:
            expand_ratio = 0.1
            w = x2 - x1
            h = y2 - y1
            x1_expand = max(0, int(x1 - w * expand_ratio))
            y1_expand = max(0, int(y1 - h * expand_ratio))
            x2_expand = min(width, int(x2 + w * expand_ratio))
            y2_expand = min(height, int(y2 + h * expand_ratio))
            
            human_roi = frame[y1_expand:y2_expand, x1_expand:x2_expand]
            if human_roi.size == 0:
                continue
            
            pose_results = pose_model(human_roi, conf=0.4)
            
            for result in pose_results:
                if result.keypoints is not None and len(result.keypoints.xy) > 0:
                    keypoints = result.keypoints.xy[0].cpu().numpy()
                    
                    # 绘制骨骼
                    if hasattr(pose_model, 'yaml') and 'skeleton' in pose_model.yaml:
                        skeleton = pose_model.yaml['skeleton']
                        for connection in skeleton:
                            a, b = connection
                            if a < len(keypoints) and b < len(keypoints):
                                kp_a = keypoints[a]
                                kp_b = keypoints[b]
                                
                                if len(kp_a) >= 3:
                                    x_a, y_a, conf_a = kp_a
                                else:
                                    x_a, y_a = kp_a[:2]
                                    conf_a = 1.0
                                
                                if len(kp_b) >= 3:
                                    x_b, y_b, conf_b = kp_b
                                else:
                                    x_b, y_b = kp_b[:2]
                                    conf_b = 1.0
                                
                                if conf_a > 0.5 and conf_b > 0.5:
                                    orig_x_a = int(x1_expand + x_a * (x2_expand - x1_expand) / human_roi.shape[1])
                                    orig_y_a = int(y1_expand + y_a * (y2_expand - y1_expand) / human_roi.shape[0])
                                    orig_x_b = int(x1_expand + x_b * (x2_expand - x1_expand) / human_roi.shape[1])
                                    orig_y_b = int(y1_expand + y_b * (y2_expand - y1_expand) / human_roi.shape[0])
                                    
                                    cv2.line(result_frame, (orig_x_a, orig_y_a), (orig_x_b, orig_y_b), 
                                             (255, 150, 50), 2, cv2.LINE_AA)
                    
                    # 绘制关键点
                    for kp in keypoints:
                        if len(kp) >= 3:
                            x, y, conf = kp
                        else:
                            x, y = kp[:2]
                            conf = 1.0
                        if conf > 0.5:
                            orig_x = int(x1_expand + x * (x2_expand - x1_expand) / human_roi.shape[1])
                            orig_y = int(y1_expand + y * (y2_expand - y1_expand) / human_roi.shape[0])
                            cv2.circle(result_frame, (orig_x, orig_y), 4, (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(result_frame, (orig_x, orig_y), 2, (0, 100, 255), -1, cv2.LINE_AA)

        # 写入结果帧
        out.write(result_frame)

    cap.release()
    out.release()
    print(f"处理完成: {output_path}")


if __name__ == "__main__":
    input_dir = "/home/jiahao.wu/DATACENTER1/basketball/codes/f0f3b7dc83af72dcb4cb8589dfd38564"  # 输入文件夹
    output_dir = "/home/jiahao.wu/DATACENTER1/basketball/output"  # 输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, os.path.splitext(filename)[0] + "_out.mp4"
            )
            process_video(input_path, output_path)
