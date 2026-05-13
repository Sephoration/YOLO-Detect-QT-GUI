#pragma once

#include <QObject>
#include <QProcess>
#include <QJsonObject>
#include <QJsonDocument>
#include <QQueue>
#include <QMutex>
#include <opencv2/core.hpp>
#include "models/DetectionResult.h"

/**
 * @brief Python YOLO 推理桥接
 *
 * 通过子进程启动 Python 推理服务，通过 JSON/stdin-stdout 通信。
 * 支持异步推理请求与回调。
 */
class YoloBridge : public QObject
{
    Q_OBJECT
public:
    explicit YoloBridge(QObject *parent = nullptr);
    ~YoloBridge() override;

    bool startService();
    void stopService();
    bool isReady() const;

    /** 异步推理 —— 结果通过 inferenceResultReady 返回 */
    void requestInference(const cv::Mat &frame,
                          const QString &modelPath,
                          const QString &mode,
                          double confThreshold,
                          double iouThreshold);

    /** 分析模型文件，获取元信息 */
    ModelInfo analyzeModel(const QString &modelPath);

signals:
    void inferenceResultReady(InferenceResult result);
    void modelInfoReady(ModelInfo info);
    void serviceError(const QString &error);
    void serviceStatus(const QString &status);

private slots:
    void onReadyReadStdout();
    void onReadyReadStderr();
    void onProcessError(QProcess::ProcessError err);
    void onProcessFinished(int exitCode, QProcess::ExitStatus status);

private:
    bool sendJson(const QJsonObject &obj);
    void handleResponse(const QJsonObject &obj);
    static QJsonObject matToJson(const cv::Mat &mat);

    QProcess *m_proc   = nullptr;
    QByteArray m_buf;
    bool m_ready       = false;

    // Pending callbacks for sync operations
    QMutex m_pendingMtx;
    InferenceResult m_pendingResult;
    bool m_pendingValid = false;
};
