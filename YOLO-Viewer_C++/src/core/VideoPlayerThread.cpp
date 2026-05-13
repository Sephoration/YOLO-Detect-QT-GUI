#include "VideoPlayerThread.h"
#include "Config.h"
#include <QDebug>
#include <opencv2/imgproc.hpp>

// ═══════════════════════════════════════════════════════════
VideoPlayerThread::VideoPlayerThread(QObject *parent)
    : QThread(parent)
{
}

VideoPlayerThread::~VideoPlayerThread()
{
    requestStop();
    wait(3000);
}

// ────────────────────────────
//  播放控制
// ────────────────────────────
void VideoPlayerThread::playVideo(const QString &videoPath)
{
    QMutexLocker lk(&m_mtx);
    requestStop();                  // 停止当前播放
    m_sourcePath = videoPath;
    m_camId      = -1;
    m_mode       = "video";
    m_ok         = true;
    if (!isRunning()) start(LowPriority);
}

void VideoPlayerThread::playCamera(int cameraId)
{
    QMutexLocker lk(&m_mtx);
    requestStop();
    m_sourcePath.clear();
    m_camId  = cameraId;
    m_mode   = "camera";
    m_ok     = true;
    if (!isRunning()) start(LowPriority);
}

void VideoPlayerThread::requestStop()
{
    QMutexLocker lk(&m_mtx);
    m_stopReq = true;
    m_ok      = false;
    if (m_cap.isOpened()) m_cap.release();
    lk.unlock();
    if (isRunning()) { quit(); wait(1500); }
}

void VideoPlayerThread::requestPause()
{
    QMutexLocker lk(&m_mtx);
    m_pauseReq = true;
}

void VideoPlayerThread::requestResume()
{
    QMutexLocker lk(&m_mtx);
    m_pauseReq = false;
}

void VideoPlayerThread::seekToFrame(int targetFrame)
{
    QMutexLocker lk(&m_mtx);
    if (!m_cap.isOpened() || m_mode != "video") return;
    int f = qBound(0, targetFrame, m_totalFrames - 1);
    m_cap.set(cv::CAP_PROP_POS_FRAMES, f);
    cv::Mat frame;
    if (m_cap.read(frame) && !frame.empty()) {
        m_currentFrame = f;
        QImage img = cvMatToQImage(frame);
        lk.unlock();
        int fid = ++m_frameIdCounter;
        emit displayFrameReady(img, fid);
        if (m_detectEn)
            emit rawFrameReady(frame.clone(), fid);
    }
}

void VideoPlayerThread::seekByRatio(double ratio)
{
    seekToFrame(int(ratio * m_totalFrames));
}

// ────────────────────────────
//  查询
// ────────────────────────────
bool VideoPlayerThread::isPaused()       const { QMutexLocker l(&m_mtx); return m_pauseReq; }
bool VideoPlayerThread::isRunningState() const { QMutexLocker l(&m_mtx); return m_ok; }
bool VideoPlayerThread::isCameraMode()   const { QMutexLocker l(&m_mtx); return m_mode == "camera"; }
int  VideoPlayerThread::currentFrameNumber() const { QMutexLocker l(&m_mtx); return m_currentFrame; }
int  VideoPlayerThread::totalFrameCount()    const { QMutexLocker l(&m_mtx); return m_totalFrames; }
double VideoPlayerThread::currentFps()       const { QMutexLocker l(&m_mtx); return m_fps; }
double VideoPlayerThread::playbackProgress() const {
    QMutexLocker l(&m_mtx);
    return m_totalFrames > 0 ? double(m_currentFrame) / m_totalFrames : 0.0;
}
QString VideoPlayerThread::sourcePath() const { QMutexLocker l(&m_mtx); return m_sourcePath; }
QString VideoPlayerThread::playMode()   const { QMutexLocker l(&m_mtx); return m_mode; }

// ────────────────────────────
//  检测开关
// ────────────────────────────
void VideoPlayerThread::setDetectEnabled(bool on)
{
    QMutexLocker lk(&m_mtx);
    m_detectEn = on;
}

bool VideoPlayerThread::isDetectEnabled() const
{
    QMutexLocker lk(&m_mtx);
    return m_detectEn;
}

// ────────────────────────────
//  打开视频源
// ────────────────────────────
bool VideoPlayerThread::openVideoSource()
{
    m_frameIdCounter = 0;
    m_currentFrame   = 0;

    if (m_mode == QLatin1String("video")) {
        m_cap.open(m_sourcePath.toStdString());
        if (!m_cap.isOpened()) {
            emit statusUpdated(QStringLiteral("无法打开视频: %1").arg(m_sourcePath));
            return false;
        }
        m_totalFrames = int(m_cap.get(cv::CAP_PROP_FRAME_COUNT));
        m_fps         = m_cap.get(cv::CAP_PROP_FPS);
        if (m_fps <= 0) m_fps = AppConfig::minFps();
        m_durationSec = m_totalFrames / m_fps;

        emit statusUpdated(QStringLiteral("打开视频: %1  [%2 帧, %3 FPS]")
                           .arg(m_sourcePath).arg(m_totalFrames).arg(m_fps, 0, 'f', 1));
        emit sourceChanged(m_sourcePath, m_mode);
    }
    else if (m_mode == QLatin1String("camera")) {
        m_cap.open(m_camId);
        if (!m_cap.isOpened()) {
            emit statusUpdated(QStringLiteral("无法打开摄像头 #%1").arg(m_camId));
            return false;
        }
        auto res = AppConfig::cameraResolution();
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH,  res.width());
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, res.height());
        m_cap.set(cv::CAP_PROP_FPS, AppConfig::targetFps());
        m_fps         = AppConfig::targetFps();
        m_totalFrames = 0;
        m_durationSec = 0;

        emit statusUpdated(QStringLiteral("摄像头 #%1 已启动 [%2x%3]")
                           .arg(m_camId).arg(res.width()).arg(res.height()));
        emit sourceChanged(QStringLiteral("摄像头 #%1").arg(m_camId), m_mode);
    }
    return true;
}

// ────────────────────────────
//  采集并发送一帧
// ────────────────────────────
bool VideoPlayerThread::grabAndSendFrame()
{
    cv::Mat frame;
    bool ret = m_cap.read(frame);
    if (!ret || frame.empty()) {
        if (m_mode == QLatin1String("video")) {
            // 循环
            m_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            ret = m_cap.read(frame);
            if (!ret || frame.empty()) return false;
        } else {
            return false;
        }
    }

    if (m_mode == QLatin1String("video"))
        m_currentFrame = int(m_cap.get(cv::CAP_PROP_POS_FRAMES));
    else
        ++m_currentFrame;

    int fid = ++m_frameIdCounter;

    // 1) 始终发送显示帧
    QImage displayImg = cvMatToQImage(frame);
    if (!displayImg.isNull())
        emit displayFrameReady(displayImg, fid);

    // 2) 检测开启时才发送原始帧
    if (m_detectEn)
        emit rawFrameReady(frame, fid);

    // 3) 进度
    if (m_mode == QLatin1String("video" ) && m_totalFrames > 0) {
        double ratio = double(m_currentFrame) / m_totalFrames;
        emit positionChanged(ratio, m_currentFrame, m_totalFrames);
    }

    return true;
}

// ────────────────────────────
//  线程主循环
// ────────────────────────────
void VideoPlayerThread::run()
{
    emit statusUpdated(QStringLiteral("播放器线程启动"));

    {   // 加锁打开
        QMutexLocker lk(&m_mtx);
        if (!openVideoSource()) { m_ok = false; return; }
    }

    int sleepMs = qMax(1, int(1000.0 / m_fps));
    QElapsedTimer timer;
    timer.start();

    while (true) {
        // 检查停止
        {
            QMutexLocker lk(&m_mtx);
            if (m_stopReq) break;
        }

        // 暂停
        {
            QMutexLocker lk(&m_mtx);
            if (m_pauseReq) {
                lk.unlock();
                msleep(AppConfig::pauseCheckInterval());
                continue;
            }
        }

        // 采集帧
        bool ok;
        {
            QMutexLocker lk(&m_mtx);
            if (!m_cap.isOpened()) break;
            ok = grabAndSendFrame();
            if (!ok && m_mode == QLatin1String("video")) break;
        }

        // 帧率控制
        int elapsed = timer.elapsed();
        int waitMs  = sleepMs - elapsed;
        if (waitMs > 0) msleep(waitMs);
        timer.restart();
    }

    {
        QMutexLocker lk(&m_mtx);
        if (m_cap.isOpened()) m_cap.release();
        m_ok = false;
    }

    emit playbackFinished();
    emit statusUpdated(QStringLiteral("播放器线程结束"));
}

// ────────────────────────────
//  cv::Mat → QImage
// ────────────────────────────
QImage VideoPlayerThread::cvMatToQImage(const cv::Mat &mat) const
{
    if (mat.empty()) return {};
    cv::Mat bgr;
    if (mat.channels() == 1)
        cv::cvtColor(mat, bgr, cv::COLOR_GRAY2BGR);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
    else
        bgr = mat;

    cv::Mat contiguous;
    if (!bgr.isContinuous()) contiguous = bgr.clone();
    else                     contiguous = bgr;

    return QImage(contiguous.data, contiguous.cols, contiguous.rows,
                  contiguous.cols * 3, QImage::Format_BGR888).copy();
}
