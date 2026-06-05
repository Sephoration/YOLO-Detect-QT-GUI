# YOLO 多模型检测系统 v2.0

基于 **Qt6 + OpenCV + YOLO** 的实时目标检测可视化工具。  
支持检测、分类、姿态估计、分割四种任务，视频播放与检测完全解耦可独立开关。

---

## 功能特性

- **多任务支持** — 目标检测 / 图像分类 / 关键点检测 / 语义分割
- **独立视频播放** — 播放器与检测模块完全解耦，检测可随时开关不影响播放
- **自由导入模型** — 支持 `.pt` `.pth` `.onnx` `.engine`，自动识别任务类型
- **实时检测面板** — 三面板布局：显示区 + 检测表格/日志 + 参数/统计
- **检测参数可调** — IOU 阈值、置信度阈值、线宽、处理帧间隔实时调节
- **暗色主题** — 全暗色 Fusion 风格，护眼且专业
- **截图保存** — 一键保存当前画面
- **拖放支持** — 直接拖入图片/视频/模型文件
- **Python 推理后端** — 通过 JSON/stdin-stdout 与 Python YOLO 进程通信，模型缓存

---

## 系统要求

| 组件 | 版本 |
|------|------|
| C++ 编译器 | GCC ≥ 10 / MSVC 2019+ |
| CMake | ≥ 3.18 |
| Qt6 | Core, Gui, Widgets, Network |
| OpenCV | ≥ 4.5 |
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| Ultralytics | ≥ 8.0 |

---

## 编译

```bash
# 1. 克隆
git clone <repo-url>
cd YOLO-Viewer_C++

# 2. 配置 (指定 Qt6 路径)
cmake -B build -DCMAKE_PREFIX_PATH="C:/Qt/6.x.x/mingw_xx"

# 3. 编译
cmake --build build -j$(nproc)

# 编译完成后，python_bridge/yolo_service.py 会自动复制到可执行文件目录
```

### Windows (MinGW + Scoop)

```bash
# 安装依赖
scoop install mingw cmake qt opencv
pip install ultralytics torch torchvision opencv-python

# 编译
cmake -B build -G "MinGW Makefiles" -DCMAKE_PREFIX_PATH="C:/Users/<你>/scoop/apps/qt/current"
cmake --build build
```

---

## 使用

### 1. 启动 Python 推理服务

程序启动时自动启动 `yolo_service.py`，确保 Python 环境已安装：

```bash
pip install ultralytics torch torchvision opencv-python
```

### 2. 基本流程

1. **加载模型** — 点击 `📦 模型` 或 `文件 → 打开模型`，选择 `.pt` / `.onnx` 文件
2. **打开源** — 打开图片、视频或摄像头
3. **开始推理** — 点击 `开始推理` 按钮
4. **调节参数** — 实时调整 IOU、置信度、线宽等

### 3. 检测独立开关

- 左侧面板 **「检测:开/关」** 按钮可随时开关检测
- 开关检测 **不影响** 视频播放

### 4. 模式切换

右侧面板下拉框切换：
- **目标检测** — 绘制边界框 + 标签
- **图像分类** — 显示 Top-5 分类结果
- **关键点检测** — 绘制骨骼关键点
- **分割检测** — 绘制分割区域边界框

---

## 项目结构

```
├── CMakeLists.txt                  # CMake 构建配置 (Qt6 + OpenCV)
├── python_bridge/
│   └── yolo_service.py             # Python YOLO 推理服务 (JSON/stdin-stdout)
└── src/
    ├── main.cpp                    # 入口，暗色主题，Fusion 风格
    ├── Config.h/.cpp               # 全局配置中心
    ├── models/
    │   └── DetectionResult.h/.cpp  # 数据结构 + JSON 序列化
    ├── core/
    │   ├── VideoPlayerThread       # ★ 独立视频播放器 (检测可开关)
    │   ├── DetectorWorker          # 异步推理工作器
    │   ├── BaseDetectRenderer      # OpenCV 结果渲染
    │   ├── YoloBridge              # Python 进程通信桥
    │   └── MainController          # ★ 中枢控制器 (全部信号/槽)
    └── ui/
        ├── AspectRatioDisplayLabel # 保持宽高比 QLabel
        ├── LeftDisplayPanel        # 视频显示 + 播放控制 + 检测开关
        ├── RightControlPanel       # 参数 + 统计 + 模式选择
        ├── InspectionPanel         # ★ 检测目标表格 + 模型信息 + 日志
        └── MainWindow              # 三面板布局 + 菜单 + 拖放
```

---

## 架构设计

### 信号/槽链路

```
播放器(rawFrameReady) ──→ 检测器(onRawFrame) ──→ YoloBridge(requestInference)
                                                          ↓
UI(displayFrameReady) ←── 检测器(frameProcessed) ←── YoloBridge(inferenceResultReady)
                                                          ↓
                                                MainController(onDetectionStats)
                                                          ↓
                                                UI(RightControlPanel::updateStatistics)
```

### 视频播放与检测解耦

```
VideoPlayerThread
  ├── displayFrameReady(QImage)  ←── 始终发送，用于 UI 显示
  └── rawFrameReady(cv::Mat)     ←── 仅当 detectEnabled == true 时发送
                                         ↓
                                  DetectorWorker (可独立启停)
```

---

## 开发

### 添加新检测模式

1. 在 `InferenceResult` 中添加对应数据结构
2. 在 `yolo_service.py` 的 `_handle_inference` 中添加处理逻辑
3. 在 `BaseDetectRenderer` 中添加渲染方法
4. 在 `RightControlPanel` 的 `m_modeCombo` 中添加选项

### Python 桥协议

每行一个 JSON 对象：

```json
// 请求
{"action": "inference", "model_path": "...", "mode": "detect", "frame": {...}}
{"action": "analyze_model", "model_path": "..."}
{"action": "shutdown"}

// 响应
{"action": "inference_result", "result": {...}}
{"action": "model_info", "info": {...}}
{"action": "error", "message": "..."}
{"action": "status", "message": "..."}
```

---

## License

MIT
