# -*- coding: utf-8 -*-
"""
脚本名称：6-YOLO-Divide-dataset-1.py
功能描述：将YOLO数据集划分为训练集和验证集，创建标准的YOLOv8/YOLOv11数据集目录结构
使用方法：修改dataset_root_src和dataset_root_target为源目录和目标目录，直接运行脚本
依赖库：os, shutil, random, tqdm
"""
import os
import shutil
import random
from tqdm import tqdm

# 数据集配置
# 目标路径：划分后的数据集保存目录
dataset_root_target = "./datasets/Triangle_25_1"      
# 源路径：原始数据集目录
dataset_root_src = "./datasets/Triangle_labelme_25"  
# 原始图像目录
images_src = os.path.join(dataset_root_src, "images")
# 原始标签目录
labels_src = os.path.join(dataset_root_src, "yolo")

# 创建图像目录结构
# 创建 images 目录及其子目录 train 和 val
target_images = os.path.join(dataset_root_target, "images")
target_images_train = os.path.join(target_images, "train")
target_images_val = os.path.join(target_images, "val")
os.makedirs(target_images, exist_ok=True)
os.makedirs(target_images_train, exist_ok=True)
os.makedirs(target_images_val, exist_ok=True)

# 创建标签目录结构
# 创建 labels 目录及其子目录 train 和 val
target_labels = os.path.join(dataset_root_target, "labels")
target_labels_train = os.path.join(target_labels, "train")
target_labels_val = os.path.join(target_labels, "val")
os.makedirs(target_labels, exist_ok=True)
os.makedirs(target_labels_train, exist_ok=True)
os.makedirs(target_labels_val, exist_ok=True)

# 数据集划分配置
test_frac = 0.2  # 验证集比例（20%）
random.seed(123) # 随机数种子，确保划分结果可复现

# 获取所有图像文件名
img_paths = os.listdir(images_src)
random.shuffle(img_paths) # 随机打乱图像顺序

# 计算划分数量
val_number = int(len(img_paths) * test_frac) # 验证集文件个数
train_files = img_paths[val_number:]         # 训练集文件名列表
val_files = img_paths[:val_number]           # 验证集文件名列表

# 打印划分结果
print('数据集文件总数', len(img_paths))
print('训练集文件个数', len(train_files))
print('验证集文件个数', len(val_files))

# 复制训练集文件
print('\n正在复制训练集文件...')
for each in tqdm(train_files):
    # 复制图像文件到训练集图像目录
    shutil.copy2(os.path.join(images_src, each), target_images_train)
    # 复制对应的标签文件到训练集标签目录
    yolo_file = os.path.splitext(each)[0] + '.txt'
    shutil.copy2(os.path.join(labels_src, yolo_file), target_labels_train)

# 复制验证集文件
print('\n正在复制验证集文件...')
for each in tqdm(val_files):
    # 复制图像文件到验证集图像目录
    shutil.copy2(os.path.join(images_src, each), target_images_val)
    # 复制对应的标签文件到验证集标签目录
    yolo_file = os.path.splitext(each)[0] + '.txt'
    shutil.copy2(os.path.join(labels_src, yolo_file), target_labels_val)

print('\n数据集划分完成！')
print(f'训练集图像路径：{target_images_train}')
print(f'训练集标签路径：{target_labels_train}')
print(f'验证集图像路径：{target_images_val}')
print(f'验证集标签路径：{target_labels_val}')