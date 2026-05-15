#include "MainController.h"
#include "Config.h"
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QStandardPaths>
#include <QApplication>
#include <QClipboard>
#include <QStatusBar>

MainController::MainController(MainWindow *ui, QObject *parent)
    : QObject(parent), m_ui(ui)
{
    initModules();
    connectAll();
    m_ui->statusBar()->showMessage(tr("系统就绪 — 加载模型后开始检测"));

    if (m_bridge)
        m_bridge->startService();
}

MainController::~MainController()
{
    stopAll();
}

// ── Init modules ──

void MainController::initModules()
{
    m_bridge = new YoloBridge(this);

    m_player = new VideoPlayerThread;
    m_playerThread = new QThread(this);
    m_player->moveToThread(m_playerThread);
    connect(m_playerThread, &QThread::finished, m_player, &QObject::deleteLater);
    m_playerThread->start();

    m_detector = new DetectorWorker(m_bridge);
    m_detectorThread = new QThread(this);
    m_detector->moveToThread(m_detectorThread);
    connect(m_detectorThread, &QThread::finished, m_detector, &QObject::deleteLater);
    m_detectorThread->start();
}

// ── Connect all signals ──

void MainController::connectAll()
{
    // Player → UI
    connect(m_player, &VideoPlayerThread::displayFrameReady,
            this, &MainController::onDisplayFrame);
    connect(m_player, &VideoPlayerThread::statusUpdated,
            this, &MainController::onPlayerStatus);
    connect(m_player, &VideoPlayerThread::playbackFinished,
            this, &MainController::onPlaybackFinished);
    connect(m_player, &VideoPlayerThread::positionChanged,
            this, &MainController::onPositionChanged);

    // Player → Detector
    connect(m_player, &VideoPlayerThread::rawFrameReady,
            m_detector, &DetectorWorker::onRawFrame);

    // Detector → UI
    connect(m_detector, &DetectorWorker::frameProcessed,
            this, &MainController::onProcessedFrame);
    connect(m_detector, &DetectorWorker::detectionStats,
            this, &MainController::onDetectionStats);
    connect(m_detector, &DetectorWorker::statusUpdated,
            this, &MainController::onDetectorStatus);
    connect(m_detector, &DetectorWorker::errorOccurred,
            this, &MainController::onDetectorError);

    // YOLO bridge
    connect(m_bridge, &YoloBridge::serviceStatus,
            this, &MainController::onBridgeStatus);
    connect(m_bridge, &YoloBridge::serviceError,
            this, &MainController::onBridgeError);
    connect(m_bridge, &YoloBridge::modelInfoReady,
            this, &MainController::onModelInfo);

    // UI menu/toolbar
    connect(m_ui, &MainWindow::modelLoadRequested,  this, &MainController::handleLoadModel);
    connect(m_ui, &MainWindow::imageOpenRequested,  this, &MainController::handleOpenImage);
    connect(m_ui, &MainWindow::videoOpenRequested,  this, &MainController::handleOpenVideo);
    connect(m_ui, &MainWindow::cameraOpenRequested, this, &MainController::handleOpenCamera);
    connect(m_ui, &MainWindow::aboutRequested,      this, &MainController::onAbout);
    connect(m_ui, &MainWindow::playPauseClicked,    this, &MainController::handlePlayPause);
    connect(m_ui, &MainWindow::detectToggled,       this, &MainController::handleDetectToggle);
    connect(m_ui, &MainWindow::screenshotClicked,   this, &MainController::handleScreenshot);
    connect(m_ui, &MainWindow::seekRequested,       this, &MainController::handleSeek);

    // Right panel
    auto *rp = m_ui->rightPanel();
    connect(rp, &RightControlPanel::loadModelClicked,  this, &MainController::handleLoadModel);
    connect(rp, &RightControlPanel::loadImageClicked,  this, &MainController::handleOpenImage);
    connect(rp, &RightControlPanel::loadVideoClicked,  this, &MainController::handleOpenVideo);
    connect(rp, &RightControlPanel::loadCameraClicked, this, &MainController::handleOpenCamera);
    connect(rp, &RightControlPanel::iouChanged,        this, &MainController::handleIouChange);
    connect(rp, &RightControlPanel::confidenceChanged, this, &MainController::handleConfChange);
    connect(rp, &RightControlPanel::delayChanged,      this, &MainController::handleDelayChange);
    connect(rp, &RightControlPanel::lineWidthChanged,  this, &MainController::handleLineWidthChange);
    connect(rp, &RightControlPanel::modelModeChanged,  this, &MainController::handleModeChange);
    connect(rp, &RightControlPanel::startInference,    this, &MainController::handleStartInference);
    connect(rp, &RightControlPanel::stopInference,     this, &MainController::handleStopInference);
}

// ── Stop all ──

void MainController::stopAll()
{
    m_detector->stop();
    m_player->requestStop();
    m_bridge->stopService();

    if (m_detectorThread) { m_detectorThread->quit(); m_detectorThread->wait(2000); }
    if (m_playerThread)   { m_playerThread->quit();   m_playerThread->wait(2000); }
}

// ── Player feedback ──

void MainController::onDisplayFrame(const QImage &img, int frameId)
{
    m_lastDisplayImage = img;
    m_ui->leftPanel()->setDisplayImage(QPixmap::fromImage(img), frameId);
}

void MainController::onPlayerStatus(const QString &msg)
{
    m_ui->statusBar()->showMessage(msg, 5000);
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog("[播放] " + msg);
}

void MainController::onPlaybackFinished()
{
    m_ui->leftPanel()->setPlayState(false);
    m_ui->statusBar()->showMessage(tr("播放结束"), 5000);
}

void MainController::onPositionChanged(double ratio, int frame, int total)
{
    Q_UNUSED(ratio)
    m_ui->leftPanel()->setProgress(frame, total);
}

// ── Detector feedback ──

void MainController::onProcessedFrame(const QImage &img, int frameId)
{
    m_lastDisplayImage = img;
    m_ui->leftPanel()->setDisplayImage(QPixmap::fromImage(img), frameId);
}

void MainController::onDetectionStats(const QVariantMap &stats)
{
    m_ui->rightPanel()->updateStatistics(
        stats.value("detection_count").toInt(),
        stats.value("avg_confidence").toDouble(),
        stats.value("inference_time").toDouble(),
        stats.value("fps").toDouble());
}

void MainController::onDetectorStatus(const QString &msg)
{
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog("[检测] " + msg);
}

void MainController::onDetectorError(const QString &err)
{
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog("[错误] " + err);
    MainWindow::error(m_ui, tr("检测错误"), err);
}

// ── YOLO bridge ──

void MainController::onBridgeStatus(const QString &msg)
{
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog("[YOLO] " + msg);
}

void MainController::onBridgeError(const QString &err)
{
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog("[YOLO错误] " + err);
}

void MainController::onModelInfo(const ModelInfo &info)
{
    m_ui->rightPanel()->updateModelInfo(
        QFileInfo(info.modelPath).fileName(),
        info.taskType, info.inputSize,
        QString::number(info.classCount));
    m_ui->inspectionPanel()->updateModelInfo(info);
    m_ui->statusBar()->showMessage(tr("模型已加载: %1 — 任务类型: %2")
                                   .arg(QFileInfo(info.modelPath).fileName(), info.taskType));

    // Auto-detect mode from model task type
    static const QMap<QString, QString> taskModeMap = {
        {"detect",  "detection"},
        {"classify", "classification"},
        {"pose",    "pose"},
        {"segment", "segmentation"},
    };
    QString mode = taskModeMap.value(info.taskType, "detection");
    m_modelMode = mode;
    m_detector->setMode(mode);
    m_ui->rightPanel()->setMode(mode);

    // Store skeleton for renderer
    m_skeletonConnections = info.skeletonConnections;
    m_detector->setSkeleton(m_skeletonConnections);

    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("自动识别模型任务: %1").arg(info.taskType));
}

// ── UI actions ──

void MainController::handleLoadModel()
{
    QString dir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    QString path = QFileDialog::getOpenFileName(m_ui, tr("选择 YOLO 模型"),
                                                 dir,
                                                 AppConfig::modelFileFilter());
    if (path.isEmpty()) return;
    loadModel(path);
}

bool MainController::loadModel(const QString &path)
{
    if (!QFileInfo::exists(path)) {
        MainWindow::error(m_ui, tr("模型错误"), tr("文件不存在: %1").arg(path));
        return false;
    }

    m_modelPath   = path;
    m_modelLoaded = true;

    AppConfig::addRecentModelPath(path);

    m_detector->setModelPath(path);

    if (m_bridge && m_bridge->isReady()) {
        m_bridge->analyzeModel(path);
    }

    m_ui->statusBar()->showMessage(tr("模型已选择: %1").arg(QFileInfo(path).fileName()));
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("已加载模型: %1").arg(path));

    return true;
}

void MainController::handleOpenImage()
{
    QString dir = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString path = QFileDialog::getOpenFileName(m_ui, tr("打开图片"),
                                                 dir,
                                                 AppConfig::imageFileFilter());
    if (path.isEmpty()) return;

    // Fully stop previous source
    m_detector->stop();
    m_player->requestStop();
    m_isDetecting = false;
    m_currentSource = path;

    cv::Mat mat = cv::imread(path.toStdString());
    if (mat.empty()) {
        MainWindow::error(m_ui, tr("打开失败"), tr("无法读取图片: %1").arg(path));
        return;
    }

    // Convert for display
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    QPixmap pix = QPixmap::fromImage(img.copy());

    m_ui->leftPanel()->clearDisplay();
    m_ui->leftPanel()->setDisplayImage(pix);
    m_ui->leftPanel()->updateInfo(QFileInfo(path).fileName(), "image");
    m_ui->statusBar()->showMessage(tr("已打开图片: %1").arg(QFileInfo(path).fileName()));
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("打开图片: %1").arg(path));

    // Auto-inference if model is loaded
    if (m_modelLoaded) {
        m_lastRawImage = mat.clone();
        m_detector->setMode(m_modelMode);
        m_detector->setParameters(m_confCache, m_iouCache, m_lineWidthCache);
        m_detector->start();
        m_detector->onRawFrame(m_lastRawImage, 0);
        m_isDetecting = true;
        m_ui->rightPanel()->setControlState(true);
        m_ui->leftPanel()->setDetectButtonState(true);
    }
}

void MainController::handleOpenVideo()
{
    QString dir = QStandardPaths::writableLocation(QStandardPaths::MoviesLocation);
    QString path = QFileDialog::getOpenFileName(m_ui, tr("打开视频"),
                                                 dir,
                                                 AppConfig::videoFileFilter());
    if (path.isEmpty()) return;

    // Fully stop previous source
    m_detector->stop();
    m_isDetecting = false;
    m_player->requestStop();
    m_ui->leftPanel()->clearDisplay();
    m_ui->rightPanel()->setControlState(false);
    m_ui->leftPanel()->setDetectButtonState(false);

    m_currentSource = path;
    m_player->playVideo(path);
    m_ui->leftPanel()->setDuration(m_player->durationSec());

    m_ui->leftPanel()->updateInfo(QFileInfo(path).fileName(), "video");
    m_ui->statusBar()->showMessage(tr("打开视频: %1").arg(QFileInfo(path).fileName()));
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("打开视频: %1").arg(path));
}

void MainController::handleOpenCamera()
{
    // Fully stop previous source
    m_detector->stop();
    m_isDetecting = false;
    m_player->requestStop();
    m_ui->leftPanel()->clearDisplay();
    m_ui->leftPanel()->updateInfo(QString(), "camera");
    m_ui->rightPanel()->setControlState(false);
    m_ui->leftPanel()->setDetectButtonState(false);

    m_currentSource = "camera";
    m_player->playCamera(AppConfig::defaultCameraId());

    m_ui->statusBar()->showMessage(tr("摄像头已打开"));
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("打开摄像头 #%1").arg(AppConfig::defaultCameraId()));
}

void MainController::handlePlayPause()
{
    if (m_player->isPaused())
        m_player->requestResume();
    else
        m_player->requestPause();
    m_ui->leftPanel()->setPlayState(!m_player->isPaused());
}

void MainController::handleDetectToggle(bool on)
{
    m_player->setDetectEnabled(on);

    if (!on) {
        m_detector->stop();
        m_isDetecting = false;
        m_ui->rightPanel()->setControlState(false);
    } else if (m_modelLoaded) {
        handleStartInference();
    }

    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("检测 %1").arg(on ? tr("开启") : tr("关闭")));
}

void MainController::handleStartInference()
{
    if (!m_modelLoaded) {
        MainWindow::warn(m_ui, tr("提示"), tr("请先加载 YOLO 模型！"));
        return;
    }

    m_detector->setParameters(m_confCache, m_iouCache, m_lineWidthCache);
    m_detector->setMode(m_modelMode);
    m_player->setDetectEnabled(true);

    m_detector->start();
    m_isDetecting = true;
    m_ui->rightPanel()->setControlState(true);
    m_ui->leftPanel()->setDetectButtonState(true);
}

void MainController::handleStopInference()
{
    m_detector->stop();
    m_isDetecting = false;
    m_ui->rightPanel()->setControlState(false);
}

void MainController::handleScreenshot()
{
    if (m_lastDisplayImage.isNull()) {
        MainWindow::warn(m_ui, tr("提示"), tr("没有可保存的图像"));
        return;
    }

    QString dir = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString path = QFileDialog::getSaveFileName(m_ui, tr("保存截图"),
                                                 dir + tr("/yolo_screenshot.png"),
                                                 AppConfig::screenshotFileFilter());
    if (path.isEmpty()) return;

    bool ok = m_lastDisplayImage.save(path);
    if (ok) {
        m_ui->statusBar()->showMessage(tr("截图已保存: %1").arg(path), 5000);
        auto *ip = m_ui->inspectionPanel();
        if (ip) ip->appendLog(tr("截图保存至: %1").arg(path));
    } else {
        MainWindow::error(m_ui, tr("保存失败"), tr("无法保存截图到: %1").arg(path));
    }
}

void MainController::handleSeek(double ratio)
{
    m_player->seekByRatio(ratio);
}

void MainController::handleIouChange(double v)
{
    m_iouCache = v;
    m_detector->setParameters(m_confCache, v, m_lineWidthCache);
}

void MainController::handleConfChange(double v)
{
    m_confCache = v;
    m_detector->setParameters(v, m_iouCache, m_lineWidthCache);
}

void MainController::handleDelayChange(int v)
{
    m_delayCache = v;
    m_detector->setProcessInterval(v);
}

void MainController::handleLineWidthChange(int v)
{
    m_lineWidthCache = v;
    m_detector->setParameters(m_confCache, m_iouCache, v);
}

void MainController::handleModeChange(const QString &mode)
{
    m_modelMode = mode;
    m_detector->setMode(mode);
    auto *ip = m_ui->inspectionPanel();
    if (ip) ip->appendLog(tr("检测模式切换: %1").arg(mode));
}

void MainController::onAbout()
{
    QString html = QStringLiteral(
        "<h3>YOLO 多模型检测系统</h3>"
        "<p>基于 Qt6 + OpenCV + YOLO 的实时目标检测可视化工具</p>"
        "<p><b>版本:</b> %1</p>"
        "<p><b>开发者:</b> %2</p>"
        "<p>支持检测 | 分类 | 姿态 | 分割 多种任务</p>"
    ).arg(AppConfig::APP_VERSION, AppConfig::APP_AUTHOR);
    QMessageBox::about(m_ui, tr("关于"), html);
}
