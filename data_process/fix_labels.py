import os

def clean_labels(label_dir, num_classes):
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 非法行

            cls, x, y, w, h = parts
            try:
                cls = int(cls)
                x, y, w, h = map(float, (x, y, w, h))
            except:
                continue  # 非法数字

            # 删除类别越界
            if cls < 0 or cls >= num_classes:
                continue

            # 修正非法坐标
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)

            # YOLO 格式要求 w,h > 0，否则忽略
            if w <= 0 or h <= 0:
                continue

            new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # 如果全被清理，删除文件
        if not new_lines:
            os.remove(path)
            print(f"Deleted empty/invalid label: {path}")
        else:
            with open(path, "w") as f:
                f.writelines(new_lines)

    print("Label cleaning finished ✅")

# 用法
if __name__ == "__main__":
    clean_labels("/home/jiahao.wu/DATACENTER1/basketball/data/person_basketball/train/labels", num_classes=11)  # 改成你的路径和类别数
    clean_labels("/home/jiahao.wu/DATACENTER1/basketball/data/person_basketball/valid/labels", num_classes=11)
