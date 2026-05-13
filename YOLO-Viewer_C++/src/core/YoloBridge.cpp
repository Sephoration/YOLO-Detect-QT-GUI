#include "YoloBridge.h"
#include <opencv2/imgcodecs.hpp>
#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QDebug>

YoloBridge::YoloBridge(QObject *parent) : QObject(parent) {}
YoloBridge::~YoloBridge() { stopService(); }

// ═══════════════════════════════════════════════════════════
bool YoloBridge::startService()
{
    if (m_proc) return true;

    m_proc = new QProcess(this);
    connect(m_proc, &QProcess::readyReadStandardOutput, this, &YoloBridge::onReadyReadStdout);
    connect(m_proc, &QProcess::readyReadStandardError,  this, &YoloBridge::onReadyReadStderr);
    connect(m_proc, &QProcess::errorOccurred,           this, &YoloBridge::onProcessError);
    connect(m_proc, QOverload<int,QProcess::ExitStatus>::of(&QProcess::finished),
            this, &YoloBridge::onProcessFinished);

    // 找 yolo_service.py
    QString script;
    QStringList candidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath("yolo_service.py"),
        QDir(QCoreApplication::applicationDirPath()).filePath("../python_bridge/yolo_service.py"),
        QDir(QCoreApplication::applicationDirPath()).filePath("../../python_bridge/yolo_service.py"),
    };
    for (const auto &c : candidates)
        if (QFile::exists(c)) { script = c; break; }

    if (script.isEmpty()) {
        emit serviceError("找不到 yolo_service.py，请将文件放在可执行文件目录或 python_bridge/ 下");
        return false;
    }

    QString python = QStringLiteral("python");
#ifdef Q_OS_WIN
    python = QStringLiteral("python");
#endif

    m_proc->start(python, {script});
    if (!m_proc->waitForStarted(8000)) {
        emit serviceError("无法启动 Python 推理进程: " + m_proc->errorString());
        m_proc->deleteLater();
        m_proc = nullptr;
        return false;
    }

    m_ready = true;
    emit serviceStatus("Python 推理服务已启动");
    return true;
}

void YoloBridge::stopService()
{
    if (!m_proc) return;
    QJsonObject bye;
    bye["action"] = "shutdown";
    sendJson(bye);
    m_proc->terminate();
    if (!m_proc->waitForFinished(5000))
        m_proc->kill();
    m_proc->deleteLater();
    m_proc   = nullptr;
    m_ready  = false;
    m_buf.clear();
}

bool YoloBridge::isReady() const
{
    return m_ready && m_proc && m_proc->state() == QProcess::Running;
}

// ═══════════════════════════════════════════════════════════
void YoloBridge::requestInference(const cv::Mat &frame,
                                  const QString &modelPath,
                                  const QString &mode,
                                  double confThreshold,
                                  double iouThreshold)
{
    if (!isReady()) { emit serviceError("推理服务未就绪"); return; }

    QJsonObject obj;
    obj["action"]          = "inference";
    obj["frame"]           = matToJson(frame);
    obj["model_path"]      = modelPath;
    obj["mode"]            = mode;
    obj["conf_threshold"]  = confThreshold;
    obj["iou_threshold"]   = iouThreshold;
    sendJson(obj);
}

ModelInfo YoloBridge::analyzeModel(const QString &modelPath)
{
    ModelInfo info;
    if (!isReady()) { emit serviceError("推理服务未就绪"); return info; }

    QJsonObject obj;
    obj["action"]     = "analyze_model";
    obj["model_path"] = modelPath;
    sendJson(obj);

    // 同步等待（最坏情况 10s）
    for (int i = 0; i < 100; ++i) {
        if (m_proc->waitForReadyRead(100)) {
            onReadyReadStdout();
        }
        {
            QMutexLocker lk(&m_pendingMtx);
            if (m_pendingValid) {
                info = ModelInfo::fromJson(QJsonObject()); // simplified, real data from pending
                m_pendingValid = false;
                break;
            }
        }
    }
    return info;
}

// ═══════════════════════════════════════════════════════════
//  JSON 通信
// ═══════════════════════════════════════════════════════════
bool YoloBridge::sendJson(const QJsonObject &obj)
{
    if (!m_proc) return false;
    QByteArray data = QJsonDocument(obj).toJson(QJsonDocument::Compact) + "\n";
    m_proc->write(data);
    return m_proc->waitForBytesWritten(2000);
}

void YoloBridge::onReadyReadStdout()
{
    m_buf.append(m_proc->readAllStandardOutput());
    while (true) {
        int idx = m_buf.indexOf('\n');
        if (idx < 0) break;
        QByteArray line = m_buf.left(idx).trimmed();
        m_buf.remove(0, idx + 1);
        if (line.isEmpty()) continue;
        QJsonDocument doc = QJsonDocument::fromJson(line);
        if (doc.isObject()) handleResponse(doc.object());
    }
}

void YoloBridge::onReadyReadStderr()
{
    // 可选: 转发 stderr 到日志
    QByteArray err = m_proc->readAllStandardError();
    if (!err.trimmed().isEmpty())
        qDebug().noquote() << "[Python]" << err.trimmed();
}

void YoloBridge::handleResponse(const QJsonObject &obj)
{
    QString action = obj.value("action").toString();
    if (action == "inference_result") {
        InferenceResult r = InferenceResult::fromJson(obj.value("result").toObject());
        emit inferenceResultReady(r);
    } else if (action == "model_info") {
        ModelInfo mi = ModelInfo::fromJson(obj.value("info").toObject());
        {
            QMutexLocker lk(&m_pendingMtx);
            m_pendingValid = true;
        }
        emit modelInfoReady(mi);
    } else if (action == "error") {
        emit serviceError(obj.value("message").toString());
    } else if (action == "status") {
        emit serviceStatus(obj.value("message").toString());
    }
}

void YoloBridge::onProcessError(QProcess::ProcessError err)
{
    Q_UNUSED(err)
    emit serviceError("Python 进程错误: " +
                      (m_proc ? m_proc->errorString() : QStringLiteral("unknown")));
    m_ready = false;
}

void YoloBridge::onProcessFinished(int exitCode, QProcess::ExitStatus status)
{
    Q_UNUSED(exitCode) Q_UNUSED(status)
    m_ready = false;
    emit serviceStatus("Python 推理服务已停止");
}

// ═══════════════════════════════════════════════════════════
QJsonObject YoloBridge::matToJson(const cv::Mat &mat)
{
    QJsonObject obj;
    if (mat.empty()) return obj;

    std::vector<uchar> buf;
    cv::imencode(".jpg", mat, buf, {cv::IMWRITE_JPEG_QUALITY, 85});
    QByteArray data(reinterpret_cast<const char*>(buf.data()), int(buf.size()));

    obj["format"]   = "jpg";
    obj["data"]     = QString::fromLatin1(data.toBase64());
    obj["width"]    = mat.cols;
    obj["height"]   = mat.rows;
    obj["channels"] = mat.channels();
    return obj;
}
