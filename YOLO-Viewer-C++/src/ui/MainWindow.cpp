#include "MainWindow.h"
#include "Config.h"
#include <QMenuBar>
#include <QMenu>
#include <QToolBar>
#include <QStatusBar>
#include <QMessageBox>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QFileInfo>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle(tr("YOLO 多模型检测系统 v%1").arg(AppConfig::APP_VERSION));
    setGeometry(100, 100, 1400, 800);
    setAcceptDrops(true);
    initUi();
    setupMenuBar();
    setupStyleSheet();
}

void MainWindow::initUi()
{
    auto *central = new QWidget(this);
    setCentralWidget(central);

    m_splitter = new QSplitter(Qt::Horizontal, central);

    m_leftPanel = new LeftDisplayPanel(m_splitter);
    m_inspectionPanel = new InspectionPanel(m_splitter);
    m_rightPanel = new RightControlPanel(m_splitter);

    m_splitter->addWidget(m_leftPanel);
    m_splitter->addWidget(m_inspectionPanel);
    m_splitter->addWidget(m_rightPanel);

    // 比例: 左 5 : 中 3 : 右 2
    m_splitter->setStretchFactor(0, 5);
    m_splitter->setStretchFactor(1, 3);
    m_splitter->setStretchFactor(2, 2);
    m_splitter->setSizes({700, 350, 300});

    auto *lay = new QHBoxLayout(central);
    lay->setContentsMargins(0,0,0,0);
    lay->addWidget(m_splitter);

    // 状态栏
    statusBar()->showMessage(tr("就绪"));

    // ── 转发子面板信号 ──
    connect(m_leftPanel, &LeftDisplayPanel::playPauseClicked,
            this, &MainWindow::playPauseClicked);
    connect(m_leftPanel, &LeftDisplayPanel::detectToggled,
            this, &MainWindow::detectToggled);
    connect(m_leftPanel, &LeftDisplayPanel::screenshotClicked,
            this, &MainWindow::screenshotClicked);
    connect(m_leftPanel, &LeftDisplayPanel::seekChanged,
            this, &MainWindow::seekRequested);
}

void MainWindow::setupMenuBar()
{
    // ── 文件 ──
    auto *fileMenu = menuBar()->addMenu(tr("文件(&F)"));
    auto *openModelAct = fileMenu->addAction(tr("打开模型..."));
    openModelAct->setShortcut(QKeySequence::Open);
    connect(openModelAct, &QAction::triggered, this, &MainWindow::modelLoadRequested);
    fileMenu->addSeparator();
    fileMenu->addAction(tr("打开图片..."),   this, &MainWindow::imageOpenRequested);
    fileMenu->addAction(tr("打开视频..."),   this, &MainWindow::videoOpenRequested);
    fileMenu->addAction(tr("打开摄像头..."), this, &MainWindow::cameraOpenRequested);
    fileMenu->addSeparator();
    auto *exitAct = fileMenu->addAction(tr("退出"));
    exitAct->setShortcut(QKeySequence::Quit);
    connect(exitAct, &QAction::triggered, this, &QMainWindow::close);

    // ── 视图 ──
    auto *viewMenu = menuBar()->addMenu(tr("视图(&V)"));
    auto *toggleInsp = viewMenu->addAction(tr("检查面板"));
    toggleInsp->setCheckable(true);
    toggleInsp->setChecked(true);
    connect(toggleInsp, &QAction::toggled, this, [this](bool on) {
        m_inspectionPanel->setVisible(on);
    });

    // ── 帮助 ──
    auto *helpMenu = menuBar()->addMenu(tr("帮助(&H)"));
    helpMenu->addAction(tr("关于"),   this, &MainWindow::aboutRequested);
    helpMenu->addAction(tr("使用说明"), this, &MainWindow::helpRequested);
}

void MainWindow::setupStyleSheet()
{
    setStyleSheet(QStringLiteral(R"(
        QMainWindow { background-color: #252525; }
        QMenuBar { background-color: #2d2d2d; color: #ddd; border-bottom: 1px solid #444; }
        QMenuBar::item:selected { background-color: #3a3a3a; }
        QMenu { background-color: #2d2d2d; color: #ddd; border: 1px solid #444; }
        QMenu::item:selected { background-color: #3a4a6d; }
        QStatusBar { background-color: #2d2d2d; color: #999; border-top: 1px solid #444; font-size: 10px; }
        QSplitter::handle { background-color: #3a3a3a; width: 2px; }
    )"));
}

// ═══════════════════════════════════════════════════════════
//  拖放支持
// ═══════════════════════════════════════════════════════════
void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event)
{
    const auto urls = event->mimeData()->urls();
    if (urls.isEmpty()) return;
    QString path = urls.first().toLocalFile();
    if (path.isEmpty()) return;

    QString ext = QFileInfo(path).suffix().toLower();
    static const QStringList imgExt = {"png","jpg","jpeg","bmp","webp","tiff"};
    static const QStringList vidExt = {"mp4","avi","mov","mkv","flv","webm"};
    static const QStringList mdlExt = {"pt","pth","onnx","engine"};

    if (mdlExt.contains(ext))      emit modelLoadRequested(); // signal with path
    else if (imgExt.contains(ext)) emit imageOpenRequested();
    else if (vidExt.contains(ext)) emit videoOpenRequested();
}

// ═══════════════════════════════════════════════════════════
//  静态对话框
// ═══════════════════════════════════════════════════════════
bool MainWindow::confirm(QWidget *parent, const QString &title, const QString &msg)
{
    return QMessageBox::question(parent, title, msg,
                                 QMessageBox::Yes|QMessageBox::No,
                                 QMessageBox::No) == QMessageBox::Yes;
}

void MainWindow::info(QWidget *parent, const QString &title, const QString &msg)
{
    QMessageBox::information(parent, title, msg);
}

void MainWindow::warn(QWidget *parent, const QString &title, const QString &msg)
{
    QMessageBox::warning(parent, title, msg);
}

void MainWindow::error(QWidget *parent, const QString &title, const QString &msg)
{
    QMessageBox::critical(parent, title, msg);
}
