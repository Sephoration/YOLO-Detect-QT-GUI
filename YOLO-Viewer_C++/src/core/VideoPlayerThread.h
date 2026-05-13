#pragma once

#include <QThread>
#include <QImage>
#include <QMutex>
#include <QElapsedTimer>
#include <opencv2/videoio.hpp>

/**
 * @brief 独立视频播放器模块
 *
 * 播放视频或摄像头，与检测模块解耦。
 * - 播放本身不需要任何检测。
 * - 检测可自由开关，不影响播放。
 * - 外部通过 setDetectEnabled(bool) 控制是否向检测模块发送帧。
 */
class VideoPlayerThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoPlayerThread(QObject *parent = nullptr);
    ~VideoPlayerThread() override;

    // ── 播放控制 ──
    void playVideo(const QString &videoPath);
    void playCamera(int cameraId = 0);
    void requestStop();
    void requestPause();
    void requestResume();
    void seekToFrame(int targetFrame);
    void seekByRatio(double ratio);      // 0..1

    // ── 查询 ──
    bool isPaused()           const;
    bool isRunningState()     const;
    bool isCameraMode()       const;

    int  currentFrameNumber() const;
    int  totalFrameCount()    const;
    double currentFps()       const;
    double playbackProgress() const;    // 0..1
    QString sourcePath()      const;
    QString playMode()        const;    // "video" | "camera" | "none"

    // ── 检测开关（自由控制） ──
    void setDetectEnabled(bool on);
    bool isDetectEnabled()    const;

signals:
    // @{ 显示帧 —— 用于 UI 显示，播放器无论如何都会发 }
    void displayFrameReady(QImage image, int frameId);

    // @{ 原始帧 —— 仅当 detectEnabled == true 时发送 }
    void rawFrameReady(const cv::Mat &frame, int frameId);

    void statusUpdated(const QString &status);
    void playbackFinished();
    void positionChanged(double ratio, int frame, int total);
    void sourceChanged(const QString &path, const QString &mode);

protected:
    void run() override;

private:
    bool openVideoSource();
    bool grabAndSendFrame();

    QImage cvMatToQImage(const cv::Mat &mat) const;

    mutable QMutex m_mtx;

    // ── 状态 ──
    bool   m_ok          = false;
    bool   m_pauseReq    = false;
    bool   m_stopReq     = false;
    bool   m_detectEn    = true;        // 默认开启检测

    // ── 播放源 ──
    cv::VideoCapture m_cap;
    QString  m_sourcePath;
    int      m_camId     = -1;
    QString  m_mode;          // "video" | "camera" | "none"

    // ── 统计 ──
    int     m_totalFrames    = 0;
    int     m_currentFrame   = 0;
    double  m_fps            = 30.0;
    double  m_durationSec    = 0.0;
    double  m_detectionScale = 1.0;
    int     m_frameIdCounter = 0;
};
