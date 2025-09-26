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

# 处理单个图像文件
def process_image(input_path, output_path):
    print(f"检测到图像文件，正在处理: {input_path}")

    # 读取图像
    img = cv2.imread(input_path)
    if img is None:
        print("无法读取图像，请检查路径。")
        return
        
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
            
            if conf > 0.3:  # 降低置信度过滤阈值
                cv2.circle(result_img, (x_int, y_int), 5, (0, 255, 0), -1)  # 绿色圆圈

        for pair in upper_body_skeleton:
            a, b = pair
            # 确保关键点索引在范围内
            if a >= len(keypoints) or b >= len(keypoints):
                continue
                
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
            
            if conf_a > 0.3 and conf_b > 0.3:  # 降低置信度过滤阈值
                cv2.line(result_img, (xa_int, ya_int), (xb_int, yb_int), (255, 0, 0), 2)  # 蓝色线

    # 转回 BGR 并保存
    result_img_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_img_bgr)
    print(f"结果图已保存为：{output_path}")

# 处理单个视频文件
def process_video(input_path, output_path):
    print(f"检测到视频文件，正在处理: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
                
                if conf > 0.3:  # 降低置信度过滤阈值
                    cv2.circle(result_frame, (x_int, y_int), 5, (0, 255, 0), -1)

            for pair in upper_body_skeleton:
                a, b = pair
                # 确保关键点索引在范围内
                if a >= len(keypoints) or b >= len(keypoints):
                    continue
                    
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
                
                if conf_a > 0.3 and conf_b > 0.3:  # 降低置信度过滤阈值
                    cv2.line(result_frame, (xa_int, ya_int), (xb_int, yb_int), (255, 0, 0), 2)

        # 转回 BGR 并保存
        result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        out.write(result_frame_bgr)

    cap.release()
    out.release()
    print(f"结果视频已保存为：{output_path}")

# 处理文件夹中的所有视频文件
def process_folder(input_folder):
    print(f"检测到文件夹，正在处理: {input_folder}")
    
    # 创建结果文件夹
    result_folder = os.path.join(input_folder, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"创建结果文件夹: {result_folder}")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        # 跳过结果文件夹本身
        if filename == "result":
            continue
            
        # 处理视频文件
        if is_video_file(file_path):
            # 生成输出文件路径
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_result{ext}"
            output_path = os.path.join(result_folder, output_filename)
            
            # 处理视频
            process_video(file_path, output_path)
            
        # 处理图像文件
        elif is_image_file(file_path):
            # 生成输出文件路径
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_result{ext}"
            output_path = os.path.join(result_folder, output_filename)
            
            # 处理图像
            process_image(file_path, output_path)

# 1. 加载 YOLO 模型（YOLO11n-pose）
model = YOLO("yolo11n-pose.pt")  # 可替换为 yolov8n-pose.pt 等

# 2. 定义上半身关键点索引（COCO 格式，共 17 个关键点）
# 关键点索引：[0:鼻子, 1:左眼, 2:右眼, 3:左耳, 4:右耳, 5:左肩, 6:右肩, 7:左肘, 8:右肘, 9:左手, 10:右手, 11:左髋, 12:右髋, 13:左膝, 14:右膝, 15:左脚, 16:右脚]
upper_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# 3. 定义优化的上半身骨骼连接（仅包含上半身连接）
upper_body_skeleton = [
    # 头部连接
    [0, 1], [0, 2],   # 鼻子 -> 左眼/右眼
    [1, 3], [2, 4],   # 眼睛 -> 耳朵
    
    # 肩膀连接
    [5, 6],           # 左肩 <-> 右肩
    [0, 5], [0, 6],   # 鼻子 -> 肩膀
    
    # 手臂连接
    [5, 7], [7, 9],   # 左肩 -> 左肘 -> 左手
    [6, 8], [8, 10],  # 右肩 -> 右肘 -> 右手
    
    # 躯干连接
    [5, 11], [6, 12], # 肩膀 -> 髋部
    [11, 12]          # 左髋 <-> 右髋
]

# 4. 输入路径（支持图片、视频或文件夹）
input_path = "/home/jiahao.wu/DATACENTER1/basketball/data/f0f3b7dc83af72dcb4cb8589dfd38564.mp4"  # 替换为你的路径

# 判断输入类型并处理
if not os.path.exists(input_path):
    print(f"错误：文件不存在 -> {input_path}")
elif os.path.isfile(input_path):
    # 处理单个文件（图像或视频）
    dir_name = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    # 创建结果文件夹
    result_folder = os.path.join(dir_name, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    output_path = os.path.join(result_folder, f"{name}_result{ext}")
    
    if is_image_file(input_path):
        process_image(input_path, output_path)
    elif is_video_file(input_path):
        process_video(input_path, output_path)
    else:
        print("不支持的文件格式，请输入 JPG/PNG 图像 或 MP4/AVI/MOV 视频。")
elif os.path.isdir(input_path):
    # 处理文件夹
    process_folder(input_path)
else:
    print("输入路径既不是文件也不是文件夹，请检查路径。")