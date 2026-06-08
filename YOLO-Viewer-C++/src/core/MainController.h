#pragma once

#include <QObject>
#include <QThread>
#include "ui/MainWindow.h"
#include "core/VideoPlayerThread.h"
#include "core/DetectorWorker.h"
#include "core/YoloBridge.h"

/**
 * @brief 主控制器 —— 串联所有模块
 *
 * 负责：
 * 1. 创建和管理各模块实例
 * 2. 连接所有信号/槽
 * 3. 协调播放器 ↔ 检测 ↔ UI 的数据流
 * 4. 提供"检测独立开关"：播放器不受检测影响
 */
class MainController : public QObject
{
    Q_OBJECT
public:
    explicit MainController(MainWindow *ui, QObject *parent = nullptr);
    ~MainController() override;

private slots:
    // ── 播放器反馈 ──
    void onDisplayFrame(const QImage &img, int frameId);
    void onPlayerStatus(const QString &msg);
    void onPlaybackFinished();
    void onPositionChanged(double ratio, int frame, int total);

    // ── 检测反馈 ──
    void onProcessedFrame(const QImage &img, int frameId);
    void onDetectionStats(const QVariantMap &stats);
    void onDetectorStatus(const QString &msg);
    void onDetectorError(const QString &err);

    // ── YOLO 桥 ──
    void onBridgeStatus(const QString &msg);
    void onBridgeError(const QString &err);
    void onModelInfo(const ModelInfo &info);

    // ── UI 操作 ──
    void handleLoadModel();
    void handleOpenImage();
    void handleOpenVideo();
    void handleOpenCamera();
    void handlePlayPause();
    void handleDetectToggle(bool on);
    void handleStartInference();
    void handleStopInference();
    void handleScreenshot();
    void handleSeek(double ratio);

    void handleIouChange(double v);
    void handleConfChange(double v);
    void handleDelayChange(int v);
    void handleLineWidthChange(int v);
    void handleModeChange(const QString &mode);

    void onAbout();

private:
    void initModules();
    void connectAll();
    void stopAll();

    bool loadModel(const QString &path);

    // ── 模块 ──
    MainWindow          *m_ui       = nullptr;
    VideoPlayerThread   *m_player   = nullptr;
    QThread             *m_playerThread = nullptr;
    DetectorWorker      *m_detector = nullptr;
    QThread             *m_detectorThread = nullptr;
    YoloBridge          *m_bridge   = nullptr;

    // ── 状态 ──
    bool m_modelLoaded  = false;
    bool m_isDetecting = false;
    QString m_modelPath;
    QString m_modelMode = "detection";
    QString m_currentSource;
    QImage  m_lastDisplayImage;
    cv::Mat m_lastRawImage;     // 图片模式下用于推理的原始帧

    // 缓存推理参数
    double m_confCache    = 0.5;
    double m_iouCache     = 0.45;
    int    m_lineWidthCache = 2;
    int    m_delayCache   = 1;

    // 缓存的模型骨架连接
    QVector<QPair<int,int>> m_skeletonConnections;
};
