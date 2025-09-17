from ultralytics import YOLO
import cv2
import os

# 判断文件类型
def is_video_file(path):
    ext = path.lower().split('.')[-1]
    return ext in ['mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv']

def is_image_file(path):
    ext = path.lower().split('.')[-1]
    return ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']

# 1. 加载 YOLO 模型（YOLO11n-pose）
model = YOLO("yolo11n-pose.pt")  # 可替换为 yolov8n-pose.pt 等

# 2. 定义上半身关键点索引（COCO 格式，共 8 个关键点）
# 关键点索引：[鼻子, 颈部, 右肩, 右肘, 右手, 左肩, 左肘, 左手]
upper_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7]

# 3. 定义上半身骨骼连接（COCO 格式）
upper_body_skeleton = [
    [1, 2], [1, 5],      # 颈部 -> 右肩 / 左肩
    [2, 3], [3, 4],      # 右肩 -> 右肘 -> 右手
    [5, 6], [6, 7],      # 左肩 -> 左肘 -> 左手
    [1, 0]               # 颈部 -> 鼻子（可选）
]

# 4. 输入路径（支持图片或视频）
input_path = "/home/jiahao.wu/DATACENTER1/basketball/data/fly.mp4"  # 替换为你的路径

# 判断输入类型并处理
if not os.path.exists(input_path):
    print(f"错误：文件不存在 -> {input_path}")
elif is_image_file(input_path):
    print("检测到图像文件，正在处理...")

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print("无法读取图像，请检查路径。")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, conf=0.25)
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 绘制上半身关键点与连接线
        for result in results:
            # 检查是否有检测到的关键点
            if len(result.keypoints.xy) == 0:
                print("未检测到任何关键点，跳过绘制")
                continue
                
            # 获取关键点数据，确保格式正确
            keypoints = result.keypoints.xy[0].cpu().numpy()  # 转换为numpy数组便于处理
            
            for idx in upper_body_keypoints:
                # 处理可能只有x,y两个值的情况
                if len(keypoints[idx]) >= 3:
                    x, y, conf = keypoints[idx]
                else:
                    x, y = keypoints[idx]
                    conf = 1.0  # 假设置信度为1.0
                
                # 转换为整数坐标
                x_int, y_int = int(x), int(y)
                
                if conf > 0.5:  # 置信度过滤
                    cv2.circle(result_img, (x_int, y_int), 5, (0, 255, 0), -1)  # 绿色圆圈

            for pair in upper_body_skeleton:
                a, b = pair
                # 处理关键点a
                if len(keypoints[a]) >= 3:
                    xa, ya, conf_a = keypoints[a]
                else:
                    xa, ya = keypoints[a]
                    conf_a = 1.0
                    
                # 处理关键点b
                if len(keypoints[b]) >= 3:
                    xb, yb, conf_b = keypoints[b]
                else:
                    xb, yb = keypoints[b]
                    conf_b = 1.0
                
                # 转换为整数坐标
                xa_int, ya_int = int(xa), int(ya)
                xb_int, yb_int = int(xb), int(yb)
                
                if conf_a > 0.5 and conf_b > 0.5:
                    cv2.line(result_img, (xa_int, ya_int), (xb_int, yb_int), (255, 0, 0), 2)  # 蓝色线

        # 转回 BGR 并保存
        result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        output_img_path = "yolo_pose_upper_body.jpg"
        cv2.imwrite(output_img_path, result_img_bgr)
        print(f"结果图已保存为：{output_img_path}")

elif is_video_file(input_path):
    print("检测到视频文件，正在处理...")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 定义视频写入器
    output_video_path = "yolo_pose_upper_body_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0  # 用于跟踪处理的帧数
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  # 每处理10帧打印一次进度
            print(f"处理中... 已完成 {frame_count} 帧")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb, conf=0.25)
        result_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 覆盖原始绘制

        # 绘制上半身关键点与连接线
        for result in results:
            # 检查是否有检测到的关键点
            if len(result.keypoints.xy) == 0:
                # 可以选择打印信息，或直接跳过
                # print(f"第 {frame_count} 帧未检测到关键点，跳过绘制")
                continue
                
            # 获取关键点数据，确保格式正确
            keypoints = result.keypoints.xy[0].cpu().numpy()  # 转换为numpy数组
            
            for idx in upper_body_keypoints:
                # 处理可能只有x,y两个值的情况
                if len(keypoints[idx]) >= 3:
                    x, y, conf = keypoints[idx]
                else:
                    x, y = keypoints[idx]
                    conf = 1.0  # 假设置信度为1.0
                
                # 转换为整数坐标
                x_int, y_int = int(x), int(y)
                
                if conf > 0.5:  # 置信度过滤
                    cv2.circle(result_frame, (x_int, y_int), 5, (0, 255, 0), -1)

            for pair in upper_body_skeleton:
                a, b = pair
                # 处理关键点a
                if len(keypoints[a]) >= 3:
                    xa, ya, conf_a = keypoints[a]
                else:
                    xa, ya = keypoints[a]
                    conf_a = 1.0
                    
                # 处理关键点b
                if len(keypoints[b]) >= 3:
                    xb, yb, conf_b = keypoints[b]
                else:
                    xb, yb = keypoints[b]
                    conf_b = 1.0
                
                # 转换为整数坐标
                xa_int, ya_int = int(xa), int(ya)
                xb_int, yb_int = int(xb), int(yb)
                
                if conf_a > 0.5 and conf_b > 0.5:
                    cv2.line(result_frame, (xa_int, ya_int), (xb_int, yb_int), (255, 0, 0), 2)

        # 转回 BGR 并保存
        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        out.write(result_frame_bgr)

    cap.release()
    out.release()
    print(f"结果视频已保存为：{output_video_path}")

else:
    print("不支持的文件格式，请输入 JPG/PNG 图像 或 MP4/AVI/MOV 视频。")
    