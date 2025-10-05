import os
import cv2
from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO("person-ball.pt")

# 2. 定义类别索引 (根据你的 dataset.yaml)
target_classes = [0, 2, 3]  # basketball=0, person=2, rim=3

# 3. 输入和输出文件夹
input_folder = "/home/jiahao.wu/DATACENTER1/basketball/datasets/shot-split/test-video/first_batch/part"       # 你的视频文件夹路径
output_folder = "/home/jiahao.wu/DATACENTER1/basketball/outputs-p-b"     # 保存检测后视频
os.makedirs(output_folder, exist_ok=True)

# 4. 遍历文件夹下的所有视频文件
for file in os.listdir(input_folder):
    if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    
    video_path = os.path.join(input_folder, file)
    cap = cv2.VideoCapture(video_path)

    # 获取视频参数
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(output_folder, file)
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"正在处理: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理，只检测 basketball/person/rim
        results = model.predict(frame, classes=target_classes, conf=0.25, verbose=False)

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 写入输出视频
        out.write(annotated_frame)

        # 如果你想实时看效果，可以打开：
        # cv2.imshow("YOLO Detection", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("所有视频处理完成！")
