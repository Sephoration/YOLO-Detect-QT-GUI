# -*- coding: utf-8 -*-
"""
脚本名称：7-Yolo-train.py
功能描述：使用YOLOv11-pose模型训练姿态检测模型的基础脚本
使用方法：修改data参数为数据集配置文件路径，设置合适的训练参数，直接运行脚本
依赖库：ultralytics
"""
from ultralytics import YOLO

if __name__ == '__main__':
    # 初始化YOLOv11-pose模型，使用预训练权重
    model = YOLO("yolo11s-pose.pt")

    # 开始训练模型
    # data: 数据集配置文件（.yaml格式）
    # epochs: 训练轮数
    # imgsz: 输入图像尺寸
    results = model.train(data="tr.yaml", epochs=2, imgsz=640)

