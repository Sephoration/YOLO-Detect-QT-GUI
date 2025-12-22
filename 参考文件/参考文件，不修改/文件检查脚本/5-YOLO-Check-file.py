# -*- coding: utf-8 -*-
"""
脚本名称：5-YOLO-Check-file.py
功能描述：用于检查YOLO数据集的图像和标签文件是否匹配，确保数据完整性
使用方法：修改image_folder和label_folder为目标图像和标签文件夹路径
依赖库：os
"""
import os

# 定义图像和标签文件夹路径
image_folder = "./datasets/Triangle_labelme_25/images"
label_folder = "./datasets/Triangle_labelme_25/yolo"

# 指定根目录路径
dataset_root_target = "./datasets/Triangle_25_1"      # 数据集划分后的目标路径
dataset_root_src = "./datasets/Triangle_labelme_25"  # 原始数据集来源
images_src = os.path.join(dataset_root_src, "images")  # 原始图像文件夹路径
labels_src = os.path.join(dataset_root_src, "yolo")    # 原始标签文件夹路径
