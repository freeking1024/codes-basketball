from ultralytics import YOLO

def main():
    model = YOLO("/home/jiahao.wu/DATACENTER1/basketball/codes/best.pt")

    # 2. 开始训练
    model.train(
        data="/home/jiahao.wu/DATACENTER1/basketball/ultralytics/ultralytics/cfg/datasets/data.yaml",   # 你的数据集配置文件 (包含 train/val 路径和类别名)
        epochs=200,         # 训练轮次，可以调整
        imgsz=640,          # 输入图像大小 (默认640x640)
        batch=16,           # 批量大小，根据你的显存调整
        device=0,           # GPU编号，CPU 就写 "cpu"
        workers=8,          # 数据加载线程数
        project="runs/train",  # 训练输出保存目录
        name="yolo11n-custom", # 当前训练的实验名称
    )

if __name__ == "__main__":
    main()
