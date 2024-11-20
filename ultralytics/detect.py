from ultralytics import YOLO


if __name__ == "__main__":

    # 加载模型

    model = YOLO("/hy-tmp/ultralytics/yolo11n.pt")  # YOLOv8n模型

    model.predict(
        source="/hy-tmp/ultralytics/ultralytics/assets",
        save=True,  # 保存预测结果
        project="runs/predict",  # 项目名称（可选）
        name="exp",  # 实验名称，结果保存在'project/name'目录下（可选）
    )
