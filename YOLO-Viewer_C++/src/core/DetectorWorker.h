#pragma once

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <QImage>
#include "models/DetectionResult.h"
#include "core/BaseDetectRenderer.h"
#include "core/YoloBridge.h"

/**
 * @brief 异步推理工作器
 *
 * 从帧队列取帧，通过 YoloBridge 推理，渲染结果后发射。
 * 运行在独立线程中。
 */
class DetectorWorker : public QObject
{
    Q_OBJECT
public:
    explicit DetectorWorker(YoloBridge *bridge, QObject *parent = nullptr);
    ~DetectorWorker() override;

    void setModelPath(const QString &path)  { m_modelPath = path; }
    void setMode(const QString &mode)       { m_mode = mode; }
    void setParameters(double conf, double iou, int lineWidth);
    void setProcessInterval(int interval);   // 每 N 帧处理一次
    void setRenderOptions(bool showLabels, bool showConf, bool showTrack);

    bool isProcessing() const;
    bool isPaused() const;

public slots:
    /** 从播放器接收原始帧 */
    void onRawFrame(const cv::Mat &frame, int frameId);
    /** 启动处理循环 */
    void start();
    /** 停止处理循环 */
    void stop();
    /** 暂停/继续 */
    void pause();
    void resume();

signals:
    void frameProcessed(QImage image, int frameId);
    void detectionStats(QVariantMap stats);
    void statusUpdated(const QString &status);
    void errorOccurred(const QString &error);
    void processingStarted();
    void processingStopped();

protected:
    void processLoop();

private:
    QImage matToQImage(const cv::Mat &mat) const;

    YoloBridge       *m_bridge     = nullptr;
    BaseDetectRenderer m_renderer;

    QString m_modelPath;
    QString m_mode;                // detection | classification | pose | segmentation
    double  m_confThreshold  = 0.5;
    double  m_iouThreshold   = 0.45;
    int     m_lineWidth      = 2;
    int     m_processInterval = 1;

    mutable QMutex m_mtx;
    QWaitCondition m_wait;
    QQueue<QPair<cv::Mat,int>> m_frameQueue;   // frame + frameId
    static constexpr int MAX_QUEUE_SIZE = 10;

    bool m_running    = false;
    bool m_paused     = false;
    bool m_stopReq    = false;
    int  m_frameCount = 0;
    int  m_processed  = 0;
    qint64 m_totalInferenceNs = 0;

    // 跨线程结果暂存
    InferenceResult m_lastResult;
    bool m_resultValid = false;
};
