# -*- coding: utf-8 -*-
"""
脚本名称：4-labelme2yolo-2.py
功能描述：批量将LabelMe JSON格式的标签转换为YOLO格式，支持边界框和关键点转换
使用方法：修改dataset_root_src和dataset_root_target为源目录和目标目录，直接运行脚本
依赖库：os, json, shutil, numpy, tqdm
"""
import os
import json
import shutil
import numpy as np
from tqdm import tqdm

# 数据集配置
# 源目录：LabelMe JSON文件所在目录
dataset_root_src = "D:\code\YOLO11-pose\Triangle_labelme_25\la"
# 目标目录：YOLO格式标签输出目录
dataset_root_target = "D:\code\YOLO11-pose\Triangle_labelme_25\yolo"
os.makedirs(dataset_root_target, exist_ok=True)  # 创建输出目录（如果不存在）

# 类别定义
# 框的类别映射（名称到ID）
bbox_class = {'sjb_rect':0}

# 关键点的类别列表（按顺序排列）
keypoint_class = ['angle_30', 'angle_60', 'angle_90']

# 单个JSON文件处理函数
def process_single_json(labelme_file):
    """
    处理单个LabelMe JSON文件，转换为YOLO格式
    参数：labelme_file - LabelMe JSON文件名
    """
    # 构建完整文件路径
    labelme_path = os.path.join(dataset_root_src, labelme_file)
    
    # 读取JSON文件
    with open(labelme_path, 'r', encoding='utf-8') as f:
        labelme = json.load(f)
    
    # 获取图像尺寸信息
    img_width = labelme['imageWidth']  # 图像宽度
    img_height = labelme['imageHeight']  # 图像高度

    # 生成YOLO格式标签文件路径
    file_name = os.path.splitext(os.path.basename(labelme_path))[0]
    yolo_txt_path = os.path.join(dataset_root_target, file_name + '.txt')

    # 写入YOLO格式标签
    with open(yolo_txt_path, 'w', encoding='utf-8') as f:
        # 遍历每个标注
        for each_ann in labelme['shapes']:  
            if each_ann['shape_type'] == 'rectangle':  # 处理边界框标注，每个框写一行
                yolo_str = ''
                
                ## 处理边界框信息
                bbox_class_id = bbox_class[each_ann['label']]  # 获取框的类别 ID
                yolo_str += '{} '.format(bbox_class_id)  # 写入类别ID
                
                # 计算边界框的左上角和右下角坐标
                bbox_top_left_x = int(min(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_bottom_right_x = int(max(each_ann['points'][0][0], each_ann['points'][1][0]))
                bbox_top_left_y = int(min(each_ann['points'][0][1], each_ann['points'][1][1]))
                bbox_bottom_right_y = int(max(each_ann['points'][0][1], each_ann['points'][1][1]))
                
                # 计算边界框中心点坐标
                bbox_center_x = int((bbox_top_left_x + bbox_bottom_right_x) / 2)
                bbox_center_y = int((bbox_top_left_y + bbox_bottom_right_y) / 2)
                
                # 计算边界框宽度和高度
                bbox_width = bbox_bottom_right_x - bbox_top_left_x
                bbox_height = bbox_bottom_right_y - bbox_top_left_y
                
                # 归一化坐标（0-1范围）
                bbox_center_x_norm = bbox_center_x / img_width
                bbox_center_y_norm = bbox_center_y / img_height
                bbox_width_norm = bbox_width / img_width
                bbox_height_norm = bbox_height / img_height

                # 写入归一化后的边界框信息
                yolo_str += '{:.5f} {:.5f} {:.5f} {:.5f} '.format(bbox_center_x_norm, bbox_center_y_norm,
                                                                  bbox_width_norm, bbox_height_norm)
                
                # 找到该框中的所有关键点
                bbox_keypoints_dict = {}
                for each_ann in labelme['shapes']:  # 遍历所有标注，寻找关键点
                    if each_ann['shape_type'] == 'point':  # 筛选出关键点标注
                        # 获取关键点XY坐标和类别
                        x = int(each_ann['points'][0][0])
                        y = int(each_ann['points'][0][1])
                        label = each_ann['label']
                        
                        # 筛选出在当前边界框内的关键点
                        if (x > bbox_top_left_x) & (x < bbox_bottom_right_x) & (y < bbox_bottom_right_y) & (
                                y > bbox_top_left_y):  
                            bbox_keypoints_dict[label] = [x, y]

                # 按照预定义顺序写入关键点
                for each_class in keypoint_class:  # 遍历每一类关键点
                    if each_class in bbox_keypoints_dict:  # 关键点存在
                        # 归一化关键点坐标
                        keypoint_x_norm = bbox_keypoints_dict[each_class][0] / img_width
                        keypoint_y_norm = bbox_keypoints_dict[each_class][1] / img_height
                        # 2-可见不遮挡 1-遮挡 0-没有点
                        yolo_str += '{:.5f} {:.5f} {} '.format(keypoint_x_norm, keypoint_y_norm, 2)
                    else:  # 关键点不存在
                        yolo_str += '0 0 0 '  # 写入0表示没有该关键点
                    
                    # 写入最终的YOLO格式字符串到文件
                    f.write(yolo_str + '\n')
                    print('{} --> {} 转换完成'.format(labelme_path, yolo_txt_path))

# 主函数
if __name__ == '__main__':
    # 遍历源目录中的所有JSON文件
    for labelme_file in os.listdir(dataset_root_src):
        try:
            # 处理单个JSON文件
            process_single_json(labelme_file)
        except Exception as e:
            # 处理异常情况
            print('******处理有误******', labelme_file)
            print(f'错误信息：{e}')

    # 转换完成提示
    print('YOLO格式的txt标注文件已保存至 ', dataset_root_target)






