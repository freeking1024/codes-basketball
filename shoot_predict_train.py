# 投篮：shoot检测
# 目标检测：标注投篮数据集，使用数据集训练200轮后得出best.pt；使用best.pt进行检测
# 检测精确度较高，为了准确统计投篮次数，防止误报的问题将置信度调整至0.85；每帧都可能检测到投篮，设置冷却时间为0.9s
from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def run_inference(model_path, source, save_dir="../detect", confidence_threshold=0.5):
    model = YOLO(model_path)
    class_names = getattr(model, 'names', None)

    if os.path.isfile(source):
        process_single_file(model, source, save_dir, class_names, confidence_threshold)
    elif os.path.isdir(source):
        process_folder(model, source, save_dir, class_names, confidence_threshold)
    else:
        print(f"❌ 路径不存在: {source}")


def detect_shoot_in_results(results, class_names, confidence_threshold=0.5):
    """检查结果中是否包含 shoot 类别且置信度满足阈值"""
    has_shoot = False
    conf = 0.0
    filtered_boxes = []  # 存储需要绘制的框

    allowed_classes = {"shoot", "basketball", "rim"}  # 只保留这三类

    for result in results:
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls)
                class_name = class_names[int(cls_id)] if class_names else f"class_{int(cls_id)}"
                confidence = float(box.conf)

                # 只保留感兴趣的类
                if class_name in allowed_classes:
                    filtered_boxes.append((box.xyxy[0], class_name, confidence))

                # 检查是否检测到 shoot
                if class_name == "shoot" and confidence >= confidence_threshold:
                    has_shoot = True
                    conf = max(conf, confidence)
    return has_shoot, conf, filtered_boxes


def process_single_file(model, file_path, save_dir, class_names, confidence_threshold=0.5):
    """处理单个文件（图片或视频）"""
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        results = model.predict(source=file_path, save=True, project=save_dir, name="images", conf=confidence_threshold)
        has_shoot, confidence, _ = detect_shoot_in_results(results, class_names, confidence_threshold)
        if has_shoot:
            print(f"🎯 图片中检测到 shoot，置信度: {confidence:.2f}")
        print(f"✅ 图片检测完成，结果保存在 {results[0].save_dir}")

    elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {file_path}")
            return

        save_path = os.path.join(save_dir, "videos")
        os.makedirs(save_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        out_path = os.path.join(save_path, f"{name}_result.mp4")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        # 投篮计数变量
        shoot_count = 0
        shoot_detected = False
        cooldown_frames = 0  # 冷却帧计数
        cooldown_limit = int(fps * 0.9)  # 0.5秒冷却时间（防止抖动）

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
            has_shoot, confidence, filtered_boxes = detect_shoot_in_results(results, class_names, confidence_threshold)

            # === 投篮计数逻辑（带冷却） ===
            if has_shoot and not shoot_detected and cooldown_frames == 0:
                shoot_count += 1
                shoot_detected = True
                cooldown_frames = cooldown_limit
                print(f"🎯 第 {frame_count} 帧检测到投篮，累计投篮: {shoot_count} 次 (置信度: {confidence:.2f})")
            elif not has_shoot:
                shoot_detected = False

            # 冷却计时递减
            if cooldown_frames > 0:
                cooldown_frames -= 1

            # === 绘制框（仅 shoot, basketball, rim） ===
            for xyxy, cls_name, conf in filtered_boxes:
                x1, y1, x2, y2 = map(int, xyxy)
                color = (0, 255, 0) if cls_name == "shoot" else (0, 255, 255) if cls_name == "basketball" else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 绘制投篮计数
            cv2.putText(frame, f'Shoots: {shoot_count}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ 视频检测完成，结果保存在 {out_path}")
        print(f"📊 总投篮次数: {shoot_count}")
    else:
        print(f"❌ 不支持的文件类型: {file_path}，只支持 .jpg/.png/.mp4/.avi/.mov")


def process_folder(model, folder_path, save_dir, class_names, confidence_threshold=0.5):
    """处理文件夹中的所有支持的文件"""
    image_extensions = {".jpg", ".jpeg", ".png"}
    video_extensions = {".mp4", ".avi", ".mov"}

    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions or ext in video_extensions:
                files_to_process.append(os.path.join(root, file))

    if not files_to_process:
        print(f"❌ 文件夹 {folder_path} 中没有找到支持的文件")
        return

    print(f"📁 找到 {len(files_to_process)} 个文件需要处理")

    # 创建统一的保存目录
    folder_save_dir = os.path.join(save_dir, "folder_results")
    os.makedirs(folder_save_dir, exist_ok=True)
    
    # 为不同类型的输出创建子目录
    images_save_dir = os.path.join(folder_save_dir, "images")
    videos_save_dir = os.path.join(folder_save_dir, "videos")
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(videos_save_dir, exist_ok=True)

    for i, file_path in enumerate(files_to_process):
        print(f"🔄 正在处理 ({i+1}/{len(files_to_process)}): {file_path}")
        
        # 直接使用统一的保存目录，不再为每个文件创建子目录
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            # 处理图片文件
            results = model.predict(source=file_path, conf=confidence_threshold)
            has_shoot, confidence, _ = detect_shoot_in_results(results, class_names, confidence_threshold)
            if has_shoot:
                print(f"🎯 图片中检测到 shoot，置信度: {confidence:.2f}")
            
            # 保存图片到统一目录
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)
            save_filename = f"{name}_result{ext}"
            results[0].save(filename=os.path.join(images_save_dir, save_filename))
            print(f"✅ 图片检测完成，结果保存在 {os.path.join(images_save_dir, save_filename)}")
            
        elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
            # 处理视频文件
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"❌ 无法打开视频: {file_path}")
                continue

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            out_path = os.path.join(videos_save_dir, f"{name}_result.mp4")

            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            # 投篮计数变量
            shoot_count = 0
            shoot_detected = False
            cooldown_frames = 0  # 冷却帧计数
            cooldown_limit = int(fps * 0.9)  # 0.5秒冷却时间（防止抖动）

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
                has_shoot, confidence, filtered_boxes = detect_shoot_in_results(results, class_names, confidence_threshold)

                # === 投篮计数逻辑（带冷却） ===
                if has_shoot and not shoot_detected and cooldown_frames == 0:
                    shoot_count += 1
                    shoot_detected = True
                    cooldown_frames = cooldown_limit
                    print(f"🎯 第 {frame_count} 帧检测到投篮，累计投篮: {shoot_count} 次 (置信度: {confidence:.2f})")
                elif not has_shoot:
                    shoot_detected = False

                # 冷却计时递减
                if cooldown_frames > 0:
                    cooldown_frames -= 1

                # === 绘制框（仅 shoot, basketball, rim） ===
                for xyxy, cls_name, conf in filtered_boxes:
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = (0, 255, 0) if cls_name == "shoot" else (0, 255, 255) if cls_name == "basketball" else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 绘制投篮计数
                cv2.putText(frame, f'Shoots: {shoot_count}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

                out.write(frame)
                frame_count += 1

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"✅ 视频检测完成，结果保存在 {out_path}")
            print(f"📊 总投篮次数: {shoot_count}")
        else:
            print(f"❌ 不支持的文件类型: {file_path}")

if __name__ == "__main__":
    model_path = "/home/jiahao.wu/DATACENTER1/basketball/codes/runs/train/yolo11n-custom2/weights/best.pt"
    source = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shot-split/test-video/first_batch/part"
    confidence_threshold = 0.85
    run_inference(model_path, source, confidence_threshold=confidence_threshold)
