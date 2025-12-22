# -*- coding: utf-8 -*-
"""
脚本名称：ex05-yolo-CheckFile.py
功能描述：检查YOLO数据集的图像和标签文件是否匹配，并输出不匹配的文件信息
使用方法：修改image_folder和label_folder为目标图像和标签文件夹路径，直接运行脚本
依赖库：os
"""
import os

# 定义图像和标签文件夹路径
image_folder = "D:\code\YOLO11-pose\Triangle_labelme_25\images"
label_folder = "D:\code\YOLO11-pose\Triangle_labelme_25\yolo"

# 获取 images 文件夹中的所有文件名（去除后缀）
image_files = [os.path.splitext(file)[0] for file in os.listdir(image_folder)]

# 获取 labels 文件夹中的所有文件名（去除后缀）
label_files = [os.path.splitext(file)[0] for file in os.listdir(label_folder)]

# 检查图像文件对应的标签文件是否存在
print("检查图像文件对应的标签文件：")
for image_file in image_files:
    # 检查对应的 label 文件是否存在
    if image_file not in label_files:
        image_path = os.path.join(image_folder, image_file + ".jpg")
        # os.remove(image_path)  # 删除不匹配的图像文件（默认注释掉）
        print(f"未找到对应标签文件的图像：{image_file}.jpg")

# 检查标签文件对应的图像文件是否存在
print("\n检查标签文件对应的图像文件：")
for label_file in label_files:
    # 检查对应的 image 文件是否存在
    if label_file not in image_files:
        label_path = os.path.join(label_folder, label_file + ".txt")
        # os.remove(label_path)  # 删除不匹配的标签文件（默认注释掉）
        print(f"未找到对应图像文件的标签：{label_file}.txt")
