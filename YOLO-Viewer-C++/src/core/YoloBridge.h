#pragma once

#include <QObject>
#include <QProcess>
#include <QJsonObject>
#include <QJsonDocument>
#include <QQueue>
#include <QMutex>
#include <opencv2/core.hpp>
#include "models/DetectionResult.h"

class YoloBridge : public QObject
{
    Q_OBJECT
public:
    explicit YoloBridge(QObject *parent = nullptr);
    ~YoloBridge() override;

    bool startService();
    void stopService();
    bool isReady() const;

    void requestInference(const cv::Mat &frame,
                          const QString &modelPath,
                          const QString &mode,
                          double confThreshold,
                          double iouThreshold);

    void analyzeModel(const QString &modelPath);

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

    QProcess *m_proc = nullptr;
    QByteArray m_buf;
    bool m_ready = false;
};
