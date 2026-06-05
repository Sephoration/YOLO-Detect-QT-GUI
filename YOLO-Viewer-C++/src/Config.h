#pragma once

#include <QString>
#include <QMap>
#include <QVector>
#include <QSize>
#include <QStringList>

/// Central configuration — all tunable constants in one place.
struct AppConfig
{
    // ── App info ──
    static constexpr const char* APP_NAME     = "YOLO 多模型检测系统";
    static constexpr const char* APP_VERSION  = "2.0.0";
    static constexpr const char* APP_AUTHOR   = "Sephoration";

    // ── Camera defaults ──
    static int       defaultCameraId();
    static QSize     cameraResolution();
    static int       targetFps();
    static int       cameraBufferSize();
    static int       warmupFrames();
    static int       retryDelayMs();
    static int       msleepDuration();

    // ── Video / Player ──
    static int       videoBufferSize();
    static int       minFps();
    static int       maxFps();
    static bool      seekAccuracy();
    static int       frameIntervalMs();
    static bool      useGrabMethod();
    static int       pauseCheckInterval();
    static int       progressUpdateThreshold();

    // ── Threads ──
    static int       threadTimeoutMs();
    static bool      useOpencvThreads();
    static int       opencvNumThreads();

    // ── YOLO inference ──
    static double    defaultConfidence();
    static double    defaultIou();
    static int       defaultLineWidth();
    static int       inferenceBatchSize();
    static int       warmupIterations();
    static QStringList supportedModelFormats();

    // ── UI layout ──
    static QSize     displayRatio();
    static QSize     panelRatio();
    static int       progressRange();
    static QString   timeFormat();
    static int       statusUpdateDelay();

    // ── Performance ──
    static bool      enableFrameBuffer();
    static int       frameBufferSize();
    static bool      skipDuplicateFrames();
    static bool      downsampleLargeFrames();
    static QSize     maxFrameSize();

    // ── File dialogs ──
    static QString   modelFileFilter();
    static QString   imageFileFilter();
    static QString   videoFileFilter();
    static QString   screenshotFileFilter();
    static QString   defaultSaveFormat();
    static int       maxRecentFiles();

    // ── Task display ──
    static QMap<QString, QString> taskDisplayMap();
    static QString defaultInputSize();

    // ── Debug ──
    static QString   logLevel();
    static bool      logToFile();
    static QString   logFile();
    static bool      profilePerformance();
    static bool      showFps();

    // ── PT model paths (recently imported) ──
    static QStringList recentModelPaths();
    static void      setRecentModelPaths(const QStringList &paths);
    static void      addRecentModelPath(const QString &path);
};
