#include "VideoPlayerThread.h"
#include "Config.h"
#include "utils/ImageUtils.h"
#include <QDebug>
#include <opencv2/imgproc.hpp>

VideoPlayerThread::VideoPlayerThread(QObject *parent)
    : QThread(parent) {}

VideoPlayerThread::~VideoPlayerThread()
{
    requestStop();
    wait(3000);
}

// ── Play control ──

void VideoPlayerThread::playVideo(const QString &videoPath)
{
    // Signal stop without holding mutex (avoids deadlock)
    m_stopReq = true;
    if (isRunning()) { quit(); wait(1500); }

    // Now safe to take lock
    QMutexLocker lk(&m_mtx);
    if (m_cap.isOpened()) m_cap.release();
    m_sourcePath = videoPath;
    m_camId      = -1;
    m_mode       = "video";
    m_ok         = true;
    m_stopReq    = false;
    if (!isRunning()) start(LowPriority);
}

void VideoPlayerThread::playCamera(int cameraId)
{
    m_stopReq = true;
    if (isRunning()) { quit(); wait(1500); }

    QMutexLocker lk(&m_mtx);
    if (m_cap.isOpened()) m_cap.release();
    m_sourcePath.clear();
    m_camId  = cameraId;
    m_mode   = "camera";
    m_ok     = true;
    m_stopReq = false;
    if (!isRunning()) start(LowPriority);
}

void VideoPlayerThread::requestStop()
{
    m_stopReq = true;
    {
        QMutexLocker lk(&m_mtx);
        m_ok = false;
        if (m_cap.isOpened()) m_cap.release();
    }
    if (isRunning()) { quit(); wait(1500); }
}

void VideoPlayerThread::requestPause()
{
    m_pauseReq = true;
}

void VideoPlayerThread::requestResume()
{
    m_pauseReq = false;
}

void VideoPlayerThread::seekToFrame(int targetFrame)
{
    QImage displayImg;
    cv::Mat rawFrame;
    int fid = 0;
    bool detectOn = false;

    {
        QMutexLocker lk(&m_mtx);
        if (!m_cap.isOpened() || m_mode != "video") return;
        int f = qBound(0, targetFrame, m_totalFrames - 1);
        m_cap.set(cv::CAP_PROP_POS_FRAMES, f);
        cv::Mat frame;
        if (m_cap.read(frame) && !frame.empty()) {
            m_currentFrame = f;
            displayImg = cvMatToQImage(frame);
            fid = ++m_frameIdCounter;
            rawFrame = frame.clone();
            detectOn = m_detectEn;
        }
    }

    if (!displayImg.isNull()) {
        emit displayFrameReady(displayImg, fid);
        if (detectOn)
            emit rawFrameReady(rawFrame, fid);
    }
}

void VideoPlayerThread::seekByRatio(double ratio)
{
    QMutexLocker lk(&m_mtx);
    int total = m_totalFrames;
    lk.unlock();
    seekToFrame(int(ratio * total));
}

// ── Queries ──

bool VideoPlayerThread::isPaused()       const { return m_pauseReq; }
bool VideoPlayerThread::isRunningState() const { QMutexLocker l(&m_mtx); return m_ok; }
bool VideoPlayerThread::isCameraMode()   const { QMutexLocker l(&m_mtx); return m_mode == "camera"; }
int  VideoPlayerThread::currentFrameNumber() const { QMutexLocker l(&m_mtx); return m_currentFrame; }
int  VideoPlayerThread::totalFrameCount()    const { QMutexLocker l(&m_mtx); return m_totalFrames; }
double VideoPlayerThread::currentFps()       const { QMutexLocker l(&m_mtx); return m_fps; }

double VideoPlayerThread::playbackProgress() const
{
    QMutexLocker l(&m_mtx);
    return m_totalFrames > 0 ? double(m_currentFrame) / m_totalFrames : 0.0;
}

double VideoPlayerThread::durationSec() const
{
    QMutexLocker l(&m_mtx);
    return m_durationSec;
}

QString VideoPlayerThread::sourcePath() const { QMutexLocker l(&m_mtx); return m_sourcePath; }
QString VideoPlayerThread::playMode()   const { QMutexLocker l(&m_mtx); return m_mode; }

// ── Detect toggle ──

void VideoPlayerThread::setDetectEnabled(bool on)
{
    m_detectEn = on;
}

bool VideoPlayerThread::isDetectEnabled() const
{
    return m_detectEn;
}

// ── Open source ──

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

        // Warmup reads (cameras often need a few frames to stabilize)
        cv::Mat warmupFrame;
        for (int i = 0; i < AppConfig::warmupFrames(); ++i) {
            m_cap.read(warmupFrame);
        }

        emit statusUpdated(QStringLiteral("摄像头 #%1 已启动 [%2x%3]")
                           .arg(m_camId).arg(res.width()).arg(res.height()));
        emit sourceChanged(QStringLiteral("摄像头 #%1").arg(m_camId), m_mode);
    }
    return true;
}

// ── Grab frame ──

bool VideoPlayerThread::grabAndSendFrame()
{
    int total, current, fid;
    double fps;
    QString mode;
    bool detectOn;

    cv::Mat frame;
    {
        QMutexLocker lk(&m_mtx);
        bool ret = m_cap.read(frame);
        // Camera warmup: retry a few times on empty frames
        if (!ret || frame.empty()) {
            if (m_mode == QLatin1String("video")) {
                // Video: try restarting from beginning
                m_cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                ret = m_cap.read(frame);
                if (!ret || frame.empty()) return false;
            } else {
                // Camera: retry up to 10 times for warmup
                for (int warmup = 0; warmup < 10 && (frame.empty()); ++warmup) {
                    ret = m_cap.read(frame);
                    if (ret && !frame.empty()) break;
                    QThread::msleep(50);
                }
                if (frame.empty()) return false;
            }
        }

        if (m_mode == QLatin1String("video"))
            m_currentFrame = int(m_cap.get(cv::CAP_PROP_POS_FRAMES));
        else
            ++m_currentFrame;

        fid = ++m_frameIdCounter;
        total   = m_totalFrames;
        current = m_currentFrame;
        fps     = m_fps;
        mode    = m_mode;
    }

    detectOn = m_detectEn;

    // Always send display frame
    QImage displayImg = cvMatToQImage(frame);
    if (!displayImg.isNull())
        emit displayFrameReady(displayImg, fid);

    // Send raw frame only when detection is enabled
    if (detectOn)
        emit rawFrameReady(frame, fid);

    // Progress
    if (mode == QLatin1String("video") && total > 0) {
        double ratio = double(current) / total;
        emit positionChanged(ratio, current, total);
    }

    return true;
}

// ── Main loop ──

void VideoPlayerThread::run()
{
    emit statusUpdated(QStringLiteral("播放器线程启动"));

    {
        QMutexLocker lk(&m_mtx);
        if (!openVideoSource()) { m_ok = false; return; }
    }

    double fps;
    QString mode;
    {
        QMutexLocker lk(&m_mtx);
        fps = m_fps;
        mode = m_mode;
    }
    // For cameras, cv::VideoCapture::read() blocks — no need to sleep
    int sleepMs = (mode == QLatin1String("camera")) ? 1 : qMax(1, int(1000.0 / fps));
    QElapsedTimer timer;
    timer.start();

    while (!m_stopReq) {
        if (m_pauseReq) {
            msleep(AppConfig::pauseCheckInterval());
            timer.restart();
            continue;
        }

        {
            QMutexLocker lk(&m_mtx);
            if (!m_cap.isOpened()) break;
        }

        bool ok = grabAndSendFrame();
        if (!ok) break;

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
