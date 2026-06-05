#pragma once

#include <QThread>
#include <QImage>
#include <QMutex>
#include <QElapsedTimer>
#include <atomic>
#include <opencv2/videoio.hpp>

class VideoPlayerThread : public QThread
{
    Q_OBJECT
public:
    explicit VideoPlayerThread(QObject *parent = nullptr);
    ~VideoPlayerThread() override;

    void playVideo(const QString &videoPath);
    void playCamera(int cameraId = 0);
    void requestStop();
    void requestPause();
    void requestResume();
    void seekToFrame(int targetFrame);
    void seekByRatio(double ratio);

    bool isPaused()       const;
    bool isRunningState() const;
    bool isCameraMode()   const;

    int  currentFrameNumber() const;
    int  totalFrameCount()    const;
    double currentFps()       const;
    double playbackProgress() const;
    double durationSec()      const;
    QString sourcePath()      const;
    QString playMode()        const;

    void setDetectEnabled(bool on);
    bool isDetectEnabled() const;

signals:
    void displayFrameReady(QImage image, int frameId);
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

    // Mutex protects: m_cap, m_mode, m_sourcePath, m_camId, m_totalFrames,
    //                 m_fps, m_durationSec, m_currentFrame, m_ok
    mutable QMutex m_mtx;

    // Atomic flags — lock-free access from any thread
    std::atomic<bool> m_stopReq{false};
    std::atomic<bool> m_pauseReq{false};
    std::atomic<bool> m_detectEn{true};

    // Playback source (guarded by m_mtx)
    cv::VideoCapture m_cap;
    QString  m_sourcePath;
    int      m_camId  = -1;
    QString  m_mode;

    // Stats (guarded by m_mtx)
    int     m_totalFrames  = 0;
    int     m_currentFrame = 0;
    double  m_fps          = 30.0;
    double  m_durationSec  = 0.0;
    bool    m_ok           = false;

    int     m_frameIdCounter = 0;
};
