"""
YOLO统一分析器
整合三个模式：检测、分类、关键点
直接调用模型对象，禁止使用.predict()方法
职责：只负责模型推理、数据处理和标准化，不包含任何渲染逻辑
渲染职责已完全移交给 baseDetect.py
"""

# 修正导入语句 - 只在文件开头添加一次
import cv2
import numpy as np
import torch
import time
import json
import os
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple
from ultralytics import YOLO
# 修改导入语句以匹配正确的文件名
from baseDetect import BaseDetect


class UnifiedYOLO:
    """
    统一YOLO处理器 - 专注于模型推理和数据处理
    职责清单：
    1. ✅ 模型加载与管理
    2. ✅ 模型类型识别
    3. ✅ 模型推理调用
    4. ✅ 原始数据提取
    5. ✅ 数据标准化处理
    6. ✅ 统计信息计算
    7. ✅ 配置参数管理
    8. ❌ 不负责任何可视化渲染
    """
    
    def __init__(self, model_path: str, mode: str = 'auto',
                 conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 warmup: bool = True, config_path: str = None):
        """
        初始化YOLO处理器
        Args:
            model_path: 模型文件路径
            mode: 运行模式 ('auto', 'detection', 'classification', 'pose', 'segmentation')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            warmup: 是否启用模型预热
            config_path: 配置文件路径
        """
        
        self.model_path = model_path
        self.mode = self._detect_mode(model_path) if mode == 'auto' else mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.warmup = warmup
        self.config_path = config_path
        
        # 设备选择
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 模型对象（延迟加载）
        self.model = None
        self.model_info = {}
        self.warmed_up = False
        
        # 模型元数据
        self.num_keypoints = 0
        self.keypoint_shape = None
        self.skeleton_connect = []
        
        # 推理参数
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.img_size = None  # 初始为None，将从模型获取真实尺寸
        
        # 加载配置文件
        self.config = self._load_config()
        
        # 设置模式参数
        self._setup_mode_params()
        
        # 新增：渲染器
        self.renderer = BaseDetect()
        
        print(f"🧠 YOLO处理器初始化 | 模式: {self.mode} | 设备: {self.device}")
        print(f"📊 参数配置 | 置信度: {self.conf} | IOU: {self.iou}")
    
    def _detect_mode(self, model_path: str) -> str:
        """
        自动检测模型类型 - 基于文件名关键词
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            str: 模型模式 ('detection', 'classification', 'pose', 'segmentation')
        """
        filename = Path(model_path).name.lower()
        
        # 文件名关键词匹配
        if 'cls' in filename or 'classify' in filename:
            return 'classification'
        elif 'pose' in filename or 'keypoint' in filename:
            return 'pose'
        elif 'seg' in filename:
            return 'segmentation'
        else:  # 默认检测模式
            return 'detection'
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            Dict: 配置字典，如果文件不存在则返回空字典
        """
        config = {}
        
        # 如果没有提供配置文件路径，尝试自动查找
        if not self.config_path:
            model_dir = os.path.dirname(self.model_path)
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            self.config_path = os.path.join(model_dir, f"{model_name}.json")
        
        # 检查配置文件是否存在
        if not os.path.exists(self.config_path):
            print(f"📝 未找到配置文件，使用默认参数: {self.config_path}")
            return config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"✅ 配置文件加载成功: {self.config_path}")
            
            # 从配置中提取关键点信息
            if 'keypoints' in config:
                keypoint_config = config['keypoints']
                self.num_keypoints = keypoint_config.get('num_keypoints', 0)
                self.skeleton_connect = keypoint_config.get('skeleton', [])
                
        except Exception as e:
            print(f"❌ 配置文件加载失败: {e}")
        
        return config
    
    def _setup_mode_params(self):
        """
        根据模式和配置设置推理参数
        
        注意：这里只设置推理参数，不设置渲染参数
        渲染参数应该在 baseDetect 中设置
        """
        # 从配置中获取参数（如果有）
        if 'inference' in self.config:
            inference_config = self.config['inference']
            self.conf = inference_config.get('conf_threshold', self.conf_threshold)
            self.iou = inference_config.get('iou_threshold', self.iou_threshold)
        
        # 模式特定的参数调整
        if self.mode == 'classification':
            # 分类模型的尺寸将在加载模型后从模型本身获取
            self.iou = 0.45  # 分类不需要IOU，但设置一个默认值
        elif self.mode == 'pose':
            self.conf = max(self.conf, 0.3)  # 姿态估计需要更高的置信度
    
    def _get_inference_size(self, frame: np.ndarray) -> int:
        """
        智能获取推理尺寸
        策略：
        1. 优先使用模型预设尺寸 (self.img_size)
        2. 如果输入图片小，使用图片的最大边
        3. 确保尺寸是32的倍数（YOLO要求）
        
        Args:
            frame: 输入图像
            
        Returns:
            int: 推理尺寸
        """
        if self.img_size is not None:
            # 使用模型预设尺寸
            model_size = self.img_size
            
            # 获取图片尺寸
            h, w = frame.shape[:2]
            max_side = max(h, w)
            
            # 如果图片很小，使用图片的最大边（但不超过模型预设尺寸）
            if max_side < model_size:
                # 确保是32的倍数
                smart_size = ((max_side + 31) // 32) * 32
                # 限制最小尺寸为160，最大为模型预设尺寸
                smart_size = max(160, min(smart_size, model_size))
                print(f"📐 智能调整推理尺寸: {model_size} -> {smart_size} (图片尺寸: {w}x{h})")
                return smart_size
            
            return model_size
        
        # 默认回退值
        return 640
    
    # ==================== 模型管理方法 ====================
    
    def load_model(self) -> bool:
        """
        加载模型（延迟加载）
        
        Returns:
            bool: 模型加载是否成功
        """
        if self.model is not None:
            return True
        
        try:
            model_name = Path(self.model_path).name
            print(f"📥 正在加载模型: {model_name}")
            
            # 加载YOLO模型
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # ================== 核心修复：使用模型真实任务类型 ==================
            # 不要信文件名，要信模型本身的 task 属性
            if hasattr(self.model, 'task'):
                real_task = self.model.task  # 获取真实任务类型
                print(f"🔍 模型自报任务类型: {real_task}")
                
                # 建立映射关系 (YOLO task -> 系统 mode)
                task_map = {
                    'pose': 'pose',
                    'detect': 'detection',
                    'classify': 'classification',
                    'segment': 'segmentation'
                }
                
                # 强制更新模式
                if real_task in task_map:
                    old_mode = self.mode
                    self.mode = task_map[real_task]
                    print(f"✅ 已自动修正运行模式为: {self.mode}")
                    
                    # 重新设置模式参数
                    if old_mode != self.mode:
                        self._setup_mode_params()
                        print(f"🔧 已更新模式特定参数")
            
            # ================== 核心修复：获取模型真实输入尺寸 ==================
            if hasattr(self.model, 'imgsz'):
                model_imgsz = self.model.imgsz
                if isinstance(model_imgsz, (list, tuple)):
                    # 如果是列表/元组，取第一个值
                    self.img_size = model_imgsz[0]
                else:
                    # 如果是单个值，直接使用
                    self.img_size = model_imgsz
                
                print(f"📏 模型预设输入尺寸: {self.img_size}")
            else:
                # 回退到默认值
                self.img_size = 640 if self.mode != 'classification' else 224
                print(f"📏 使用默认输入尺寸: {self.img_size}")
            # ================== 核心修复结束 ==================
            
            # 收集模型元信息
            self._collect_model_info()
            
            print(f"✅ 模型加载成功: {model_name}")
            print(f"📋 模型信息 | 任务: {self.model_info.get('task', 'unknown')} | "
                  f"类别数: {self.model_info.get('class_count', 0)}")
            
            # 执行预热
            if self.warmup and not self.warmed_up:
                self._perform_warmup()
                
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def _collect_model_info(self):
        """
        从模型收集元信息
        
        注意：这里只收集模型本身的信息，不收集任何渲染相关信息
        渲染相关信息应该在 baseDetect 中处理
        """
        if self.model is None:
            return
        
        # 基础模型信息
        self.model_info = {
            'mode': self.mode,
            'device': self.device,
            'task': getattr(self.model, 'task', 'unknown'),
            'class_names': [],
            'class_count': 0,
            'input_size': self.img_size,  # 使用真实的模型输入尺寸
        }
        
        try:
            # 获取类别信息
            if hasattr(self.model, 'names') and self.model.names:
                self.model_info['class_names'] = list(self.model.names.values())
                self.model_info['class_count'] = len(self.model.names)
            
            # 获取关键点信息（如果有关键关键点模型）
            if hasattr(self.model, 'nkpt') and self.model.nkpt:
                self.num_keypoints = self.model.nkpt
                self.model_info['num_keypoints'] = self.num_keypoints
            
            if hasattr(self.model, 'kpt_shape'):
                self.keypoint_shape = self.model.kpt_shape
                self.model_info['keypoint_shape'] = self.keypoint_shape
            
            # 获取骨架连接信息（如果有）
            if hasattr(self.model, 'skeleton'):
                self.skeleton_connect = self.model.skeleton
                self.model_info['skeleton'] = self.skeleton_connect
            
            print(f"📊 模型元信息收集完成 | 关键点: {self.num_keypoints} | "
                  f"输入尺寸: {self.img_size}")
                
        except Exception as e:
            print(f"⚠️ 收集模型信息时出错: {e}")
    
    def _perform_warmup(self):
        """
        执行模型预热
        
        预热可以减少首次推理的延迟，特别是对于GPU模型
        """
        if self.model is None:
            return
            
        print(f"🔥 开始模型预热 | 输入尺寸: {self.img_size}")
        start_time = time.time()
        
        try:
            # 创建虚拟输入数据
            dummy_input = np.random.randint(0, 255, 
                                          (self.img_size, self.img_size, 3), 
                                          dtype=np.uint8)
            
            with torch.no_grad():
                for i in range(3):
                    _ = self.model(
                        dummy_input, 
                        conf=self.conf, 
                        iou=self.iou, 
                        imgsz=self.img_size,  # 使用模型预设尺寸预热
                        verbose=False
                    )
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.warmed_up = True
            warmup_time = (time.time() - start_time) * 1000
            print(f"✅ 模型预热完成 | 耗时: {warmup_time:.2f}ms")
            
        except Exception as e:
            print(f"⚠️ 模型预热失败: {e}")
    
    # ==================== 核心数据处理方法 ====================
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理单帧图像 - 主入口方法
        
        职责：只处理数据，不进行任何渲染
        返回标准化数据，供渲染器使用
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            Dict: 包含以下键的字典：
                - success: 处理是否成功
                - error: 错误信息（如果success=False）
                - raw_image: 原始图像副本
                - data_type: 数据类型 ('detection','classification','pose','segmentation')
                - processed_data: 标准化后的处理数据
                - stats: 统计信息
                - model_info: 模型信息（供渲染器参考）
        """
        start_time = time.time()
        
        # 确保模型已加载
        if not self.load_model():
            return self._create_error_result(frame, '模型加载失败')
        
        try:
            # ============ 关键修改：智能获取推理尺寸 ============
            inference_size = self._get_inference_size(frame)
            
            # 根据模式调用不同的数据处理方法
            if self.mode == 'classification':
                result = self._process_classification_data(frame, inference_size)
            elif self.mode == 'pose':
                result = self._process_pose_data(frame, inference_size)
            elif self.mode == 'segmentation':
                result = self._process_segmentation_data(frame, inference_size)
            else:  # detection
                result = self._process_detection_data(frame, inference_size)
            
            # 计算处理时间
            inference_time = time.time() - start_time
            
            # 添加统计信息
            result['stats']['inference_time'] = inference_time * 1000  # 毫秒
            result['stats']['fps'] = 1.0 / inference_time if inference_time > 0 else 0
            result['stats']['inference_size'] = inference_size  # 记录实际使用的推理尺寸
            
            # 添加模型信息
            result['model_info'] = self.model_info.copy()
            
            result['success'] = True
            return result
            
        except Exception as e:
            print(f"❌ 帧处理失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(frame, str(e))
    
    def _create_error_result(self, frame: np.ndarray, error_msg: str) -> Dict[str, Any]:
        """
        创建错误结果
        
        Args:
            frame: 原始图像
            error_msg: 错误信息
            
        Returns:
            Dict: 错误结果字典
        """
        return {
            'success': False,
            'error': error_msg,
            'raw_image': frame.copy(),
            'data_type': self.mode,
            'processed_data': {},
            'stats': {
                'detection_count': 0,
                'avg_confidence': 0.0,
                'inference_time': 0,
                'fps': 0.0,
                'inference_size': self.img_size or 640
            },
            'model_info': self.model_info.copy()
        }
    
    def _process_detection_data(self, frame: np.ndarray, inference_size: int) -> Dict[str, Any]:
        """
        处理目标检测数据
        
        只提取和标准化数据，不进行任何渲染
        
        Args:
            frame: 输入图像
            inference_size: 推理时使用的尺寸
            
        Returns:
            Dict: 标准化检测数据
        """
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=inference_size,  # 使用智能调整后的尺寸
                verbose=False
            )
        
        result = results[0]
        
        # 初始化结果结构
        processed_data = {
            'detection': {
                'boxes': [],
                'labels': [],
                'confidences': [],
                'class_ids': []
            }
        }
        stats = {
            'detection_count': 0,
            'avg_confidence': 0.0,
            'class_distribution': {}
        }
        
        if result.boxes is None:
            return self._create_success_result(frame, 'detection', processed_data, stats)
        
        # 提取检测结果
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 获取类别名称
        class_names = []
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                class_names.append(result.names[cls_id])
            else:
                class_names.append(f"class_{cls_id}")
        
        # 填充标准化数据
        processed_data['detection']['boxes'] = boxes.tolist() if len(boxes) > 0 else []
        processed_data['detection']['labels'] = class_names
        processed_data['detection']['confidences'] = confidences.tolist() if len(confidences) > 0 else []
        processed_data['detection']['class_ids'] = class_ids.tolist() if len(class_ids) > 0 else []
        
        # 计算统计信息
        detection_count = len(boxes)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 类别分布
        class_distribution = {}
        for cls_name in class_names:
            class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
        
        stats['detection_count'] = detection_count
        stats['avg_confidence'] = float(avg_confidence)
        stats['class_distribution'] = class_distribution
        
        return self._create_success_result(frame, 'detection', processed_data, stats)
    
    def _process_classification_data(self, frame: np.ndarray, inference_size: int) -> Dict[str, Any]:
        """
        处理图像分类数据
        
        Args:
            frame: 输入图像
            inference_size: 推理时使用的尺寸
            
        Returns:
            Dict: 标准化分类数据
        """
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf,
                imgsz=inference_size,  # 使用智能调整后的尺寸
                verbose=False
            )
        
        result = results[0]
        
        # 初始化结果结构
        processed_data = {
            'classification': {
                'top_predictions': [],
                'all_probs': []
            }
        }
        stats = {
            'detection_count': 0,
            'avg_confidence': 0.0
        }
        
        if not hasattr(result, 'probs') or result.probs is None:
            return self._create_success_result(frame, 'classification', processed_data, stats)
        
        # 获取概率和类别
        probs = result.probs.data.cpu().numpy()
        
        # 获取前3个预测结果（按置信度降序）
        top_indices = np.argsort(probs)[-3:][::-1]
        top_probs = probs[top_indices]
        
        # 获取类别名称
        top_classes = []
        for idx in top_indices:
            if hasattr(result, 'names'):
                top_classes.append(result.names[idx])
            else:
                top_classes.append(f"class_{idx}")
        
        # 填充标准化数据
        processed_data['classification']['top_predictions'] = [
            (cls_name, float(prob)) 
            for cls_name, prob in zip(top_classes, top_probs)
        ]
        processed_data['classification']['all_probs'] = probs.tolist()
        
        # 计算统计信息
        stats['detection_count'] = 1  # 分类任务固定为1
        stats['avg_confidence'] = float(top_probs[0])  # 最高置信度
        stats['top_class'] = top_classes[0]
        stats['top_confidence'] = float(top_probs[0])
        
        return self._create_success_result(frame, 'classification', processed_data, stats)
    
    def _process_pose_data(self, frame: np.ndarray, inference_size: int) -> Dict[str, Any]:
        """
        处理关键点检测数据
        
        Args:
            frame: 输入图像
            inference_size: 推理时使用的尺寸
            
        Returns:
            Dict: 标准化关键点数据
        """
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=inference_size,  # 使用智能调整后的尺寸
                verbose=False
            )
        
        result = results[0]
        
        # 初始化结果结构
        processed_data = {
            'pose': {
                'boxes': [],
                'keypoints': [],
                'keypoints_conf': [],
                'skeleton_config': self.skeleton_connect
            }
        }
        stats = {
            'detection_count': 0,
            'avg_confidence': 0.0,
            'keypoint_count': 0,
            'num_people': 0
        }
        
        if result.boxes is None or result.keypoints is None:
            return self._create_success_result(frame, 'pose', processed_data, stats)
        
        # 提取检测结果
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints.xy is not None else []
        keypoints_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else []
        
        # 填充标准化数据
        processed_data['pose']['boxes'] = boxes.tolist() if len(boxes) > 0 else []
        processed_data['pose']['keypoints'] = keypoints.tolist() if len(keypoints) > 0 else []
        processed_data['pose']['keypoints_conf'] = keypoints_conf.tolist() if len(keypoints_conf) > 0 else []
        
        # 计算统计信息
        detection_count = len(boxes)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 计算可见关键点数量
        total_keypoints = 0
        for i in range(len(keypoints_conf)):
            if i < len(keypoints_conf):
                visible_keypoints = np.sum(keypoints_conf[i] > 0.1)
                total_keypoints += visible_keypoints
        
        stats['detection_count'] = detection_count
        stats['avg_confidence'] = float(avg_confidence)
        stats['keypoint_count'] = total_keypoints
        stats['num_people'] = len(boxes)
        
        return self._create_success_result(frame, 'pose', processed_data, stats)
    
    def _process_segmentation_data(self, frame: np.ndarray, inference_size: int) -> Dict[str, Any]:
        """
        处理分割检测数据
        
        Args:
            frame: 输入图像
            inference_size: 推理时使用的尺寸
            
        Returns:
            Dict: 标准化分割数据
        """
        with torch.no_grad():
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=inference_size,  # 使用智能调整后的尺寸
                verbose=False
            )
        
        result = results[0]
        
        # 初始化结果结构
        processed_data = {
            'segmentation': {
                'masks': [],
                'boxes': [],
                'class_ids': [],
                'confidences': []
            }
        }
        stats = {
            'detection_count': 0,
            'avg_confidence': 0.0,
            'class_distribution': {}
        }
        
        if result.masks is None:
            return self._create_success_result(frame, 'segmentation', processed_data, stats)
        
        # 提取分割结果
        masks = result.masks.data.cpu().numpy() if result.masks.data is not None else []
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
        confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        
        # 填充标准化数据（注意：masks可能是大数组，这里只存储引用或路径）
        processed_data['segmentation']['masks'] = masks.tolist() if len(masks) > 0 else []
        processed_data['segmentation']['boxes'] = boxes.tolist() if len(boxes) > 0 else []
        processed_data['segmentation']['class_ids'] = class_ids.tolist() if len(class_ids) > 0 else []
        processed_data['segmentation']['confidences'] = confidences.tolist() if len(confidences) > 0 else []
        
        # 计算统计信息
        detection_count = len(masks)
        avg_confidence = np.mean(confidences) if len(confidences) > 0 else 0.0
        
        # 类别分布
        class_distribution = {}
        for cls_id in class_ids:
            if hasattr(result, 'names') and cls_id < len(result.names):
                cls_name = result.names[cls_id]
            else:
                cls_name = f"class_{cls_id}"
            class_distribution[cls_name] = class_distribution.get(cls_name, 0) + 1
        
        stats['detection_count'] = detection_count
        stats['avg_confidence'] = float(avg_confidence)
        stats['class_distribution'] = class_distribution
        
        return self._create_success_result(frame, 'segmentation', processed_data, stats)
    
    def _create_success_result(self, frame: np.ndarray, data_type: str, 
                         processed_data: Dict, stats: Dict) -> Dict[str, Any]:
        """
        创建成功结果
        
        Args:
            frame: 原始图像
            data_type: 数据类型
            processed_data: 处理后的数据
            stats: 统计信息
            
        Returns:
            Dict: 成功结果字典
        """
        return {
            'success': True,
            'error': None,
            'raw_image': frame.copy(),  # 确保返回原始图像
            'data_type': data_type,
            'processed_data': processed_data,
            'stats': stats,
            'image': frame.copy()  # 添加image字段供渲染器使用
        }

    # 将render_detection方法移到类内部
    def render_detection(self, result: Dict[str, Any]) -> np.ndarray:
        """
        渲染检测结果
        
        Args:
            result: process_frame 返回的结果字典
            
        Returns:
            np.ndarray: 渲染后的图像
        """
        if not result.get('success', False):
            # 返回原始图像
            return result.get('raw_image', np.zeros((100, 100, 3), dtype=np.uint8))
        
        # 准备渲染器所需的数据格式
        analyzer_result = {
            'success': True,
            'raw_image': result.get('raw_image'),
            'data_type': result.get('data_type', 'detection'),
            'processed_data': result.get('processed_data', {}),
            'stats': result.get('stats', {}),
            'model_info': result.get('model_info', {})
        }
        
        # 使用 BaseDetect 渲染
        rendered_image = self.renderer.render(analyzer_result)
        
        # 可选：添加统计信息
        if 'stats' in result:
            rendered_image = self.renderer.draw_statistics(rendered_image, result['stats'])
        
        return rendered_image
    
    def update_params(self, conf_threshold=None, iou_threshold=None, img_size=None) -> bool:
        """
        实时更新推理参数
        
        Args:
            conf_threshold: 新的置信度阈值 (0.0-1.0)
            iou_threshold: 新的IOU阈值 (0.0-1.0)
            img_size: 新的输入图像尺寸
            
        Returns:
            bool: 参数更新是否成功
        """
        try:
            updated = False
            
            if conf_threshold is not None and 0.0 <= conf_threshold <= 1.0:
                self.conf = conf_threshold
                self.conf_threshold = conf_threshold
                print(f"🔄 置信度阈值更新为: {conf_threshold}")
                updated = True
            
            if iou_threshold is not None and 0.0 <= iou_threshold <= 1.0:
                self.iou = iou_threshold
                self.iou_threshold = iou_threshold
                print(f"🔄 IOU阈值更新为: {iou_threshold}")
                updated = True
            
            if img_size is not None and img_size > 0:
                self.img_size = img_size
                print(f"🔄 输入尺寸更新为: {img_size}")
                updated = True
            
            return updated
        except Exception as e:
            print(f"❌ 更新参数时出错: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict: 模型信息字典
        """
        if not self.model_info:
            self._collect_model_info()
        
        return self.model_info.copy()
    
    def __call__(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        使对象可调用
        
        Args:
            frame: 输入图像
            
        Returns:
            Dict: 处理结果
        """
        return self.process_frame(frame)
    
    # ==================== 静态方法 ====================
    
    @staticmethod
    def analyze_model_info(model_path: str) -> Dict[str, Any]:
        """
        静态方法：分析模型信息（不真正加载模型）
        
        用于在加载前预览模型信息
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            Dict: 模型信息字典
        """
        try:
            import os
            from pathlib import Path
            
            filename = Path(model_path).name
            file_size = os.path.getsize(model_path)
            
            # 根据文件名猜测模式（仅作为 fallback）
            if 'cls' in filename.lower() or 'classify' in filename.lower():
                fallback_task = 'classification'
                fallback_input_size = '224x224'
            elif 'pose' in filename.lower() or 'keypoint' in filename.lower():
                fallback_task = 'pose'
                fallback_input_size = '640x640'
            elif 'seg' in filename.lower():
                fallback_task = 'segmentation'
                fallback_input_size = '640x640'
            else:
                fallback_task = 'detection'
                fallback_input_size = '640x640'
            
            # 尝试获取更准确的模型信息
            try:
                with torch.no_grad():
                    model = YOLO(model_path)
                    
                    # 获取真实任务类型
                    real_task = getattr(model, 'task', fallback_task)
                    
                    # 建立映射关系 (YOLO task -> 系统 mode)
                    task_map = {
                        'pose': 'pose',
                        'detect': 'detection',
                        'classify': 'classification',
                        'segment': 'segmentation'
                    }
                    
                    # 转换为系统模式
                    task_type = task_map.get(real_task, fallback_task)
                    
                    model_info = {
                        'model_name': filename,
                        'task': real_task,
                        'task_type': task_type,  # 确保返回 task_type 键
                        'class_names': list(model.names.values()) if hasattr(model, 'names') else [],
                        'class_count': len(model.names) if hasattr(model, 'names') else 0,
                        'input_size': getattr(model, 'imgsz', fallback_input_size),
                        'file_size': f"{file_size/1024/1024:.1f} MB"
                    }
                    
                    # 获取关键点信息
                    if hasattr(model, 'nkpt'):
                        model_info['num_keypoints'] = model.nkpt
                    
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    print(f"🔍 静态分析: 模型真实任务类型: {real_task}, 系统模式: {task_type}")
                    return model_info
                    
            except Exception as e:
                print(f"⚠️ 详细模型信息获取失败: {e}")
                
                # 返回基本信息
                return {
                    'model_name': filename,
                    'task_type': fallback_task,
                    'input_size': fallback_input_size,
                    'file_size': f"{file_size/1024/1024:.1f} MB",
                    'class_count': '未知'
                }
                
        except Exception as e:
            print(f"❌ 模型信息分析失败: {e}")
            return None