#include "Config.h"
#include <QSettings>
#include <QStandardPaths>

// ============================================================
//  Camera
// ============================================================
int   AppConfig::defaultCameraId()               { return 0; }
QSize AppConfig::cameraResolution()              { return {640, 480}; }
int   AppConfig::targetFps()                     { return 30; }
int   AppConfig::cameraBufferSize()              { return 1; }
int   AppConfig::warmupFrames()                  { return 5; }
int   AppConfig::retryDelayMs()                  { return 10; }
int   AppConfig::msleepDuration()                { return 50; }

// ============================================================
//  Video / Player
// ============================================================
int   AppConfig::videoBufferSize()               { return 1; }
int   AppConfig::minFps()                        { return 10; }
int   AppConfig::maxFps()                        { return 60; }
bool  AppConfig::seekAccuracy()                  { return true; }
int   AppConfig::frameIntervalMs()               { return 33; }
bool  AppConfig::useGrabMethod()                 { return true; }
int   AppConfig::pauseCheckInterval()            { return 50; }
int   AppConfig::progressUpdateThreshold()       { return 10; }

// ============================================================
//  Threads
// ============================================================
int   AppConfig::threadTimeoutMs()               { return 3000; }
bool  AppConfig::useOpencvThreads()              { return false; }
int   AppConfig::opencvNumThreads()              { return 0; }

// ============================================================
//  YOLO inference
// ============================================================
double AppConfig::defaultConfidence()            { return 0.5; }
double AppConfig::defaultIou()                   { return 0.45; }
int    AppConfig::defaultLineWidth()             { return 2; }
int    AppConfig::inferenceBatchSize()           { return 1; }
int    AppConfig::warmupIterations()             { return 10; }
QStringList AppConfig::supportedModelFormats()   { return {"pt", "pth", "onnx", "engine"}; }

// ============================================================
//  UI layout
// ============================================================
QSize  AppConfig::displayRatio()                 { return {16, 9}; }
QSize  AppConfig::panelRatio()                   { return {4, 3}; }
int    AppConfig::progressRange()                { return 1000; }
QString AppConfig::timeFormat()                  { return "mm:ss"; }
int    AppConfig::statusUpdateDelay()            { return 1000; }

// ============================================================
//  Performance
// ============================================================
bool   AppConfig::enableFrameBuffer()            { return true; }
int    AppConfig::frameBufferSize()              { return 3; }
bool   AppConfig::skipDuplicateFrames()          { return true; }
bool   AppConfig::downsampleLargeFrames()        { return true; }
QSize  AppConfig::maxFrameSize()                 { return {1920, 1080}; }

// ============================================================
//  File dialogs
// ============================================================
QString AppConfig::modelFileFilter()
{
    return QObject::tr("YOLO 模型 (*.pt *.pth *.onnx *.engine);;PyTorch 模型 (*.pt *.pth);;ONNX 模型 (*.onnx);;TensorRT (*.engine);;所有文件 (*.*)");
}
QString AppConfig::imageFileFilter()
{
    return QObject::tr("图片文件 (*.png *.jpg *.jpeg *.bmp *.webp *.tiff);;所有文件 (*.*)");
}
QString AppConfig::videoFileFilter()
{
    return QObject::tr("视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.webm);;所有文件 (*.*)");
}
QString AppConfig::screenshotFileFilter()
{
    return QObject::tr("PNG 图片 (*.png);;JPEG 图片 (*.jpg *.jpeg);;BMP (*.bmp);;所有文件 (*.*)");
}
QString AppConfig::defaultSaveFormat()          { return "png"; }
int     AppConfig::maxRecentFiles()             { return 20; }

// ============================================================
//  Task display
// ============================================================
QMap<QString, QString> AppConfig::taskDisplayMap()
{
    return {
        {"detection",      "目标检测"},
        {"classification", "图像分类"},
        {"pose",           "关键点检测"},
        {"segmentation",   "分割检测"},
        {"obb",            "旋转框检测"},
    };
}
QString AppConfig::defaultInputSize()            { return "640"; }

// ============================================================
//  Debug
// ============================================================
QString AppConfig::logLevel()                    { return "DEBUG"; }
bool    AppConfig::logToFile()                   { return false; }
QString AppConfig::logFile()                     { return "yolo_viewer.log"; }
bool    AppConfig::profilePerformance()          { return false; }
bool    AppConfig::showFps()                     { return true; }

// ============================================================
//  PT model paths persistence
// ============================================================
QStringList AppConfig::recentModelPaths()
{
    QSettings s;
    return s.value("recent_models").toStringList();
}
void AppConfig::setRecentModelPaths(const QStringList &paths)
{
    QSettings s;
    s.setValue("recent_models", paths);
}
void AppConfig::addRecentModelPath(const QString &path)
{
    QStringList cur = recentModelPaths();
    cur.removeAll(path);
    cur.prepend(path);
    while (cur.size() > maxRecentFiles())
        cur.removeLast();
    setRecentModelPaths(cur);
}
