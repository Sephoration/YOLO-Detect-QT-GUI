#include "DetectorWorker.h"
#include "utils/ImageUtils.h"
#include <QDateTime>
#include <QDebug>

DetectorWorker::DetectorWorker(YoloBridge *bridge, QObject *parent)
    : QObject(parent), m_bridge(bridge)
{
    if (m_bridge) {
        // DirectConnection — YoloBridge lives in main thread, DetectorWorker in worker thread.
        // The worker thread event loop is blocked by processLoop, so queued connections
        // would never be delivered.
        connect(m_bridge, &YoloBridge::inferenceResultReady,
                this, &DetectorWorker::onInferenceResult, Qt::DirectConnection);
    }
}

DetectorWorker::~DetectorWorker()
{
    stop();
}

void DetectorWorker::setParameters(double conf, double iou, int lineWidth)
{
    QMutexLocker lk(&m_mtx);
    m_confThreshold = conf;
    m_iouThreshold  = iou;
    m_lineWidth     = lineWidth;
    m_renderer.setLineWidth(lineWidth);
}

void DetectorWorker::setProcessInterval(int interval)
{
    QMutexLocker lk(&m_mtx);
    m_processInterval = qMax(1, interval);
}

void DetectorWorker::setSkeleton(const QVector<QPair<int,int>> &conn)
{
    QMutexLocker lk(&m_mtx);
    m_renderer.setSkeletonConnections(conn);
}

void DetectorWorker::setRenderOptions(bool showLabels, bool showConf, bool showTrack)
{
    QMutexLocker lk(&m_mtx);
    m_renderer.setShowLabels(showLabels);
    m_renderer.setShowConfidence(showConf);
    m_renderer.setShowTracking(showTrack);
}

bool DetectorWorker::isProcessing() const { QMutexLocker l(&m_mtx); return m_running; }
bool DetectorWorker::isPaused()     const { QMutexLocker l(&m_mtx); return m_paused; }

// ── Frame input (called from player thread) ──

void DetectorWorker::onRawFrame(const cv::Mat &frame, int frameId)
{
    QMutexLocker lk(&m_mtx);
    if (!m_running || m_stopReq || frame.empty()) return;

    m_frameCount++;

    // Interval check at enqueue time — skipped frames never enter queue
    if (m_frameCount % m_processInterval != 0) {
        // Emit original frame for display continuity
        lk.unlock();
        QImage qi = matToQImage(frame);
        if (!qi.isNull())
            emit frameProcessed(qi, frameId);
        return;
    }

    if (m_frameQueue.size() < MAX_QUEUE_SIZE) {
        m_frameQueue.enqueue({frame.clone(), frameId});
    }
    m_wait.wakeOne();
}

// ── Start / Stop ──

void DetectorWorker::start()
{
    QMutexLocker lk(&m_mtx);
    if (m_running) return;
    if (m_modelPath.isEmpty()) {
        emit errorOccurred("请先加载 YOLO 模型");
        return;
    }
    m_running      = true;
    m_stopReq      = false;
    m_paused       = false;
    m_frameQueue.clear();
    m_frameCount   = 0;
    m_processed    = 0;
    m_totalInferenceNs = 0;
    m_waitingForResult = false;
    lk.unlock();

    emit processingStarted();
    emit statusUpdated("推理服务已启动");

    QMetaObject::invokeMethod(this, "processLoop", Qt::QueuedConnection);
}

void DetectorWorker::stop()
{
    QMutexLocker lk(&m_mtx);
    if (!m_running) return;
    m_stopReq   = true;
    m_running   = false;
    m_waitingForResult = false;
    m_wait.wakeAll();
    m_frameQueue.clear();
    lk.unlock();
    emit statusUpdated("推理已停止");
    emit processingStopped();
}

void DetectorWorker::pause()
{
    QMutexLocker lk(&m_mtx);
    if (m_running && !m_paused) {
        m_paused = true;
        emit statusUpdated("推理已暂停");
    }
}

void DetectorWorker::resume()
{
    QMutexLocker lk(&m_mtx);
    if (m_running && m_paused) {
        m_paused = false;
        m_wait.wakeAll();
        emit statusUpdated("推理已恢复");
    }
}

// ── Async processing loop (runs in worker thread) ──

void DetectorWorker::processLoop()
{
    while (true) {
        {
            QMutexLocker lk(&m_mtx);
            if (m_stopReq) break;
        }

        {
            QMutexLocker lk(&m_mtx);
            if (m_paused) {
                m_wait.wait(&m_mtx, 100);
                continue;
            }
        }

        // Wait for a frame or a pending result to complete
        cv::Mat frame;
        int frameId = -1;
        {
            QMutexLocker lk(&m_mtx);
            // Don't send a new frame while waiting for previous result
            if (m_waitingForResult || m_frameQueue.isEmpty()) {
                m_wait.wait(&m_mtx, 10);
                continue;
            }
            auto pair = m_frameQueue.dequeue();
            frame   = pair.first;
            frameId = pair.second;
            m_pendingFrame   = frame;
            m_pendingFrameId = frameId;
            m_waitingForResult = true;
        }

        if (frame.empty()) continue;

        if (!m_bridge || !m_bridge->isReady()) {
            // Bridge not ready — emit original frame
            m_waitingForResult = false;
            QImage qi = matToQImage(frame);
            if (!qi.isNull())
                emit frameProcessed(qi, frameId);
            continue;
        }

        // Fire-and-forget: result comes back via onInferenceResult
        m_inferenceStartMs = QDateTime::currentMSecsSinceEpoch();
        m_bridge->requestInference(frame, m_modelPath, m_mode,
                                   m_confThreshold, m_iouThreshold);
    }

    emit processingStopped();
    emit statusUpdated("推理工作器已停止");
}

// ── Async result handler ──

// Called from main thread via DirectConnection — must be thread-safe.
void DetectorWorker::onInferenceResult(const InferenceResult &result)
{
    cv::Mat frame;
    int frameId;
    double dt;
    int lw;
    int processed;
    qint64 totalNs;
    {
        QMutexLocker lk(&m_mtx);
        if (!m_waitingForResult) return;
        frame = m_pendingFrame.clone();
        frameId = m_pendingFrameId;
        m_waitingForResult = false;
        dt = QDateTime::currentMSecsSinceEpoch() - m_inferenceStartMs;
        m_totalInferenceNs += qint64(dt * 1e6);
        m_processed++;
        processed = m_processed;
        totalNs   = m_totalInferenceNs;
        lw = m_lineWidth;
    }

    m_renderer.setLineWidth(lw);
    cv::Mat display = m_renderer.render(result, frame);

    QVariantMap stats;
    stats["detection_count"] = result.detectionCount;
    stats["avg_confidence"]  = result.avgConfidence;
    stats["inference_time"]  = dt;
    stats["fps"]             = dt > 0 ? 1000.0 / dt : 0;
    emit detectionStats(stats);

    QImage qi = matToQImage(display);
    if (!qi.isNull())
        emit frameProcessed(qi, frameId);

    if (processed % 30 == 0 && processed > 0) {
        double avgMs = totalNs / 1e6 / processed;
        emit statusUpdated(QStringLiteral("推理: %1 ms/帧, ~%2 FPS")
                           .arg(avgMs, 0, 'f', 1)
                           .arg(avgMs > 0 ? 1000.0 / avgMs : 0, 0, 'f', 1));
    }
}

// ── Utility ──

QImage DetectorWorker::matToQImage(const cv::Mat &mat) const
{
    return cvMatToQImage(mat);
}
