from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def run_inference(model_path, source, save_dir="runs/detect", confidence_threshold=0.5):
    # 1. 加载训练好的模型
    model = YOLO(model_path)

    # 获取模型的类别名称
    class_names = getattr(model, 'names', None)

    # 检查输入路径类型
    if os.path.isfile(source):
        # 处理单个文件
        process_single_file(model, source, save_dir, class_names, confidence_threshold)
    elif os.path.isdir(source):
        # 处理文件夹
        process_folder(model, source, save_dir, class_names, confidence_threshold)
    else:
        print(f"❌ 路径不存在: {source}")

def detect_shoot_in_results(results, class_names, confidence_threshold=0.5):
    """检查结果中是否包含 shoot 类别且置信度满足阈值"""
    for result in results:
        if result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls):
                class_name = class_names[int(cls_id)] if class_names else f"class_{int(cls_id)}"
                if class_name == "shoot":
                    confidence = result.boxes.conf[i]
                    if confidence >= confidence_threshold:
                        return True, confidence
    return False, 0.0

def process_single_file(model, file_path, save_dir, class_names, confidence_threshold=0.5):
    """处理单个文件（图片或视频）"""
    if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
        # 单张图片
        results = model.predict(source=file_path, save=True, project=save_dir, name="images", conf=confidence_threshold)

        # 检查是否检测到 shoot 类别
        has_shoot, confidence = detect_shoot_in_results(results, class_names, confidence_threshold)
        if has_shoot:
            print(f"🎯 图片中检测到 shoot，置信度: {confidence:.2f}")

        print(f"✅ 图片检测完成，结果保存在 {results[0].save_dir}")

    elif file_path.lower().endswith((".mp4", ".avi", ".mov")):
        # 视频
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

        # 投篮计数相关变量
        shoot_count = 0
        shoot_detected = False  # 当前是否处于“投篮动作中”

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            results = model.predict(source=frame, conf=confidence_threshold, verbose=False)

            # 检查是否检测到 shoot 类别且置信度满足阈值
            has_shoot, confidence = detect_shoot_in_results(results, class_names, confidence_threshold)

            # 投篮检测逻辑：进入-离开
            if has_shoot and not shoot_detected:
                shoot_detected = True
                shoot_count += 1
                print(f"🎯 文件 {os.path.basename(file_path)} 第 {frame_count} 帧检测到投篮，累计投篮: {shoot_count} 次")
            elif not has_shoot and shoot_detected:
                shoot_detected = False

            # 在视频帧上绘制投篮计数
            annotated = results[0].plot()
            cv2.putText(annotated, f'Shoots: {shoot_count}', (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

            out.write(annotated)
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

    folder_save_dir = os.path.join(save_dir, "folder_results")
    os.makedirs(folder_save_dir, exist_ok=True)

    for i, file_path in enumerate(files_to_process):
        print(f"🔄 正在处理 ({i+1}/{len(files_to_process)}): {file_path}")
        file_save_dir = os.path.join(folder_save_dir, f"result_{i+1}")
        process_single_file(model, file_path, file_save_dir, class_names, confidence_threshold)

if __name__ == "__main__":
    # 修改为你的 best.pt 路径 和 输入文件路径（可以是文件或文件夹）
    model_path = "/home/jiahao.wu/DATACENTER1/basketball/codes/runs/train/yolo11n-custom2/weights/best.pt"
    source = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shot-split/test-video/first_batch/part"
    confidence_threshold = 0.30
    run_inference(model_path, source, confidence_threshold=confidence_threshold)
