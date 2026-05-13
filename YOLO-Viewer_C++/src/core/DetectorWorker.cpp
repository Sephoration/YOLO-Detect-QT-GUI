#include "DetectorWorker.h"
#include <QElapsedTimer>
#include <QDateTime>
#include <QDebug>

DetectorWorker::DetectorWorker(YoloBridge *bridge, QObject *parent)
    : QObject(parent), m_bridge(bridge)
{
    if (m_bridge) {
        connect(m_bridge, &YoloBridge::inferenceResultReady,
                this, [this](const InferenceResult &r) {
                    QMutexLocker lk(&m_mtx);
                    m_lastResult  = r;
                    m_resultValid = true;
                });
    }
}

DetectorWorker::~DetectorWorker()
{
    stop();
}

// ────────────────────────────────
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

void DetectorWorker::setRenderOptions(bool showLabels, bool showConf, bool showTrack)
{
    QMutexLocker lk(&m_mtx);
    m_renderer.setShowLabels(showLabels);
    m_renderer.setShowConfidence(showConf);
    m_renderer.setShowTracking(showTrack);
}

bool DetectorWorker::isProcessing() const { QMutexLocker l(&m_mtx); return m_running; }
bool DetectorWorker::isPaused()     const { QMutexLocker l(&m_mtx); return m_paused; }

// ────────────────────────────────
void DetectorWorker::onRawFrame(const cv::Mat &frame, int frameId)
{
    QMutexLocker lk(&m_mtx);
    if (!m_running || m_stopReq || frame.empty()) return;
    if (m_frameQueue.size() < MAX_QUEUE_SIZE) {
        m_frameQueue.enqueue({frame.clone(), frameId});
        m_frameCount++;
    }
    m_wait.wakeOne();
}

void DetectorWorker::start()
{
    QMutexLocker lk(&m_mtx);
    if (m_running) return;
    if (m_modelPath.isEmpty()) {
        emit errorOccurred("请先加载 YOLO 模型");
        return;
    }
    m_running   = true;
    m_stopReq   = false;
    m_paused    = false;
    m_frameQueue.clear();
    m_frameCount = 0;
    m_processed  = 0;
    m_totalInferenceNs = 0;
    lk.unlock();

    emit processingStarted();
    emit statusUpdated("推理服务已启动");

    // 在同线程中启动循环（需要外部将本对象 moveToThread）
    QMetaObject::invokeMethod(this, "processLoop", Qt::QueuedConnection);
}

void DetectorWorker::stop()
{
    QMutexLocker lk(&m_mtx);
    if (!m_running) return;
    m_stopReq = true;
    m_running = false;
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

// ────────────────────────────────
// 处理循环（在 woker 线程中运行）
// ────────────────────────────────
void DetectorWorker::processLoop()
{
    while (true) {
        // 是否停止
        {
            QMutexLocker lk(&m_mtx);
            if (m_stopReq) break;
        }

        // 是否暂停
        {
            QMutexLocker lk(&m_mtx);
            if (m_paused) {
                m_wait.wait(&m_mtx, 100);
                continue;
            }
        }

        // 取帧
        cv::Mat frame;
        int frameId = -1;
        {
            QMutexLocker lk(&m_mtx);
            if (m_frameQueue.isEmpty()) {
                m_wait.wait(&m_mtx, 50);
                continue;
            }
            auto pair = m_frameQueue.dequeue();
            frame   = pair.first;
            frameId = pair.second;
        }
        if (frame.empty()) continue;

        // 间隔采样
        {
            QMutexLocker lk(&m_mtx);
            if (m_frameCount % m_processInterval != 0)
                continue;
        }

        // 推理
        if (!m_bridge || !m_bridge->isReady()) {
            // 桥未就绪，直接发原图
            QImage qimg = matToQImage(frame);
            if (!qimg.isNull())
                emit frameProcessed(qimg, frameId);
            continue;
        }

        // 异步推理 —— 等待结果
        QElapsedTimer timer;
        timer.start();

        m_bridge->requestInference(frame, m_modelPath, m_mode,
                                   m_confThreshold, m_iouThreshold);

        // 轮询等待结果（最多 5 秒）
        InferenceResult result;
        bool got = false;
        for (int i = 0; i < 50; ++i) {
            {
                QMutexLocker lk(&m_mtx);
                if (m_resultValid) {
                    result       = m_lastResult;
                    m_lastResult = {};
                    m_resultValid = false;
                    got = true;
                    break;
                }
            }
            QThread::msleep(100);
            {
                QMutexLocker lk(&m_mtx);
                if (m_stopReq) break;
            }
        }

        double dt = timer.elapsed();
        m_totalInferenceNs += qint64(dt * 1e6);
        m_processed++;

        // 渲染
        cv::Mat display;
        if (got) {
            m_renderer.setLineWidth(m_lineWidth);
            display = m_renderer.render(result, frame);

            QVariantMap stats;
            stats["detection_count"] = result.detectionCount;
            stats["avg_confidence"]  = result.avgConfidence;
            stats["inference_time"]  = dt;
            stats["fps"]             = dt > 0 ? 1000.0 / dt : 0;
            emit detectionStats(stats);
        } else {
            display = frame.clone();
        }

        QImage qi = matToQImage(display);
        if (!qi.isNull())
            emit frameProcessed(qi, frameId);

        // 定期状态
        if (m_processed % 30 == 0 && m_processed > 0) {
            double avgMs = m_totalInferenceNs / 1e6 / m_processed;
            emit statusUpdated(QStringLiteral("推理: %1 ms/帧, ~%2 FPS")
                               .arg(avgMs, 0, 'f', 1)
                               .arg(avgMs > 0 ? 1000.0 / avgMs : 0, 0, 'f', 1));
        }
    }

    emit processingStopped();
    emit statusUpdated("推理工作器已停止");
}

// ────────────────────────────────
QImage DetectorWorker::matToQImage(const cv::Mat &mat) const
{
    if (mat.empty()) return {};
    cv::Mat bgr;
    if (mat.channels() == 1)      cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    else if (mat.channels() == 4) cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    else                          bgr = mat;
    auto cont = bgr.isContinuous() ? bgr : bgr.clone();
    return QImage(cont.data, cont.cols, cont.rows, cont.cols*3, QImage::Format_BGR888).copy();
}
