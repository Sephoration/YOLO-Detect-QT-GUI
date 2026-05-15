#pragma once

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
#include <QQueue>
#include <QImage>
#include "models/DetectionResult.h"
#include "core/BaseDetectRenderer.h"
#include "core/YoloBridge.h"

class DetectorWorker : public QObject
{
    Q_OBJECT
public:
    explicit DetectorWorker(YoloBridge *bridge, QObject *parent = nullptr);
    ~DetectorWorker() override;

    void setModelPath(const QString &path)  { m_modelPath = path; }
    void setMode(const QString &mode)       { m_mode = mode; }
    void setParameters(double conf, double iou, int lineWidth);
    void setProcessInterval(int interval);
    void setSkeleton(const QVector<QPair<int,int>> &conn);
    void setRenderOptions(bool showLabels, bool showConf, bool showTrack);

    bool isProcessing() const;
    bool isPaused() const;

public slots:
    void onRawFrame(const cv::Mat &frame, int frameId);
    void start();
    void stop();
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
    void onInferenceResult(const InferenceResult &result);

    YoloBridge       *m_bridge     = nullptr;
    BaseDetectRenderer m_renderer;

    QString m_modelPath;
    QString m_mode;
    double  m_confThreshold  = 0.5;
    double  m_iouThreshold   = 0.45;
    int     m_lineWidth      = 2;
    int     m_processInterval = 1;

    mutable QMutex m_mtx;
    QWaitCondition m_wait;
    QQueue<QPair<cv::Mat,int>> m_frameQueue;
    static constexpr int MAX_QUEUE_SIZE = 10;

    bool m_running    = false;
    bool m_paused     = false;
    bool m_stopReq    = false;
    int  m_frameCount = 0;
    int  m_processed  = 0;
    qint64 m_totalInferenceNs = 0;

    // Async pipeline state
    cv::Mat m_pendingFrame;
    int     m_pendingFrameId = -1;
    bool    m_waitingForResult = false;
    qint64  m_inferenceStartMs = 0;
};
