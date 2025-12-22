# -*- coding: utf-8 -*-
"""
脚本名称：2-Annotation.py
功能描述：用于可视化LabelMe格式的标注文件，展示边界框标注效果
使用方法：修改img_path和labelme_path为目标图像和标注文件路径，直接运行脚本
依赖库：cv2, numpy, json, matplotlib
"""
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# 输入图像路径
img_path = "./images/1.jpg"
# 读取图像
img_bgr = cv2.imread(img_path)
# 显示原始图像（默认不显示）
plt.imshow(img_bgr[:,:,::-1])
# plt.show()


# 载入LabelMe格式的JSON标注文件
labelme_path = "images/1_labelme.json"
with open(labelme_path, 'r', encoding='utf-8') as f:
    labelme = json.load(f)


# 可视化配置
# 框的颜色和线宽
bbox_color = (255, 129, 0)           # 框的颜色（BGR格式）
bbox_thickness = 5                      # 框的线宽

# 框类别文字配置
bbox_labelstr = {
        'font_size':6,         # 字体大小
    'font_thickness':14,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-80,       # Y 方向，文字偏移距离，向下为正
}

# 遍历所有标注
for each_ann in labelme['shapes']:  # 遍历每一个标注
    if each_ann['shape_type'] == 'rectangle':  # 筛选出矩形框标注
        # 获取框的类别标签
        bbox_label = each_ann['label']
        # 获取框的两点坐标
        bbox_keypoints = each_ann['points']
        bbox_keypoint_A_xy = bbox_keypoints[0]
        bbox_keypoint_B_xy = bbox_keypoints[1]
        
        # 计算框的左上角和右下角坐标
        # 左上角坐标
        bbox_top_left_x = int(min(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_top_left_y = int(min(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))
        # 右下角坐标
        bbox_bottom_right_x = int(max(bbox_keypoint_A_xy[0], bbox_keypoint_B_xy[0]))
        bbox_bottom_right_y = int(max(bbox_keypoint_A_xy[1], bbox_keypoint_B_xy[1]))

        # 绘制矩形框
        img_bgr = cv2.rectangle(img_bgr, 
                               (bbox_top_left_x, bbox_top_left_y), 
                               (bbox_bottom_right_x, bbox_bottom_right_y),
                               bbox_color, bbox_thickness)

        # 写入框类别文字
        img_bgr = cv2.putText(img_bgr, bbox_label, (
            bbox_top_left_x + bbox_labelstr['offset_x'], 
            bbox_top_left_y + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              bbox_labelstr['font_size'], 
                              bbox_color,
                              bbox_labelstr['font_thickness'])

# 可视化显示标注结果
plt.imshow(img_bgr[:, :, ::-1])  # BGR转RGB显示
plt.show()