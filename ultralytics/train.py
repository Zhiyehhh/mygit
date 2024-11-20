from ultralytics import YOLO


if __name__ == "__main__":

    # 加载模型

    model = YOLO(
        "/hy-tmp/ultralytics/ultralytics/cfg/models/11/yolo11.yaml", task="detect"
    )  # 不使用预训练权重训练 | detect, segment, classify, pose, obb

    # model = YOLO(r'yolov11.yaml').load("yolov11n.pt") # 使用预训练权重训练

    # 训练参数 ----------------------------------------------------------------------------------------------

    model.train(
        data="/hy-tmp/ultralytics/ultralytics/cfg/datasets/VisDrone.yaml",
        epochs=100,  # (int) 训练的周期数
        patience=50,  # (int) 等待无明显改善以进行早期停止的周期数
        batch=32,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=640,  # (int) 输入图像的大小，整数或w，h
        save=True,  # (bool) 保存训练检查点和预测结果
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=True,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device=" ",  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        workers=8,  # (int) 数据加载的工作线程数（每个DDP进程）
        project="runs/train",  # (str, optional) 项目名称
        name="exp",  # (str, optional) 实验名称，结果保存在'project/name'目录下
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer="SGD",  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=0,  # (int) 用于可重复性的随机种子
        close_mosaic=0,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=False,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
    )
