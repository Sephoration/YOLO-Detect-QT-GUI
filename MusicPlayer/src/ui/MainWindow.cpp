#include "MainWindow.h"
#include "utils/SettingsManager.h"
#include <QApplication>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSplitter>
#include <QStackedWidget>
#include <QFileDialog>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QMessageBox>
#include <QMediaMetaData>
#include <QMenuBar>
#include <QFileInfo>
#include <QLabel>
#include <QBuffer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    setupUI();
    setupMenu();
    connectSignals();
    restoreAppState();
}

MainWindow::~MainWindow() {
    saveAppState();
}

void MainWindow::setupUI() {
    setWindowTitle("Music Player");
    setMinimumSize(960, 640);
    resize(1200, 800);

    auto *central = new QWidget(this);
    setCentralWidget(central);
    auto *mainLayout = new QHBoxLayout(central);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    m_sidebar = new NavigationSidebar(this);
    mainLayout->addWidget(m_sidebar);

    auto *centerWidget = new QWidget(this);
    auto *centerLayout = new QVBoxLayout(centerWidget);
    centerLayout->setContentsMargins(0, 0, 0, 0);
    centerLayout->setSpacing(0);

    m_stack = new QStackedWidget(this);
    m_model = new PlaylistModel(this);
    m_player = new PlayerCore(m_model, this);

    m_playlistView = new PlaylistView(m_model, this);
    m_stack->addWidget(m_playlistView);

    m_libraryPage = new QWidget(this);
    m_stack->addWidget(m_libraryPage);

    m_favoritesPage = new QWidget(this);
    m_stack->addWidget(m_favoritesPage);

    auto *rightLayout = new QVBoxLayout();
    rightLayout->setContentsMargins(12, 12, 12, 12);
    rightLayout->setSpacing(12);

    m_albumCover = new AlbumCover(this);
    m_albumCover->setMinimumHeight(200);
    rightLayout->addWidget(m_albumCover, 2);

    m_lyricView = new LyricView(this);
    rightLayout->addWidget(m_lyricView, 3);

    auto *contentSplitter = new QSplitter(Qt::Horizontal, this);
    contentSplitter->addWidget(m_stack);
    contentSplitter->addWidget(new QWidget(this)); // placeholder for right panel container

    auto *rightContainer = new QWidget(this);
    rightContainer->setLayout(rightLayout);
    rightContainer->setMinimumWidth(260);
    rightContainer->setMaximumWidth(380);

    contentSplitter->addWidget(rightContainer);
    contentSplitter->setStretchFactor(0, 3);
    contentSplitter->setStretchFactor(1, 1);
    contentSplitter->setSizes({800, 320});

    centerLayout->addWidget(contentSplitter, 1);

    m_controls = new PlayerControls(m_player, this);
    m_controls->setFixedHeight(80);
    centerLayout->addWidget(m_controls);

    mainLayout->addWidget(centerWidget, 1);

    // Style
    setStyleSheet(
        "QMainWindow { background: #0f0f16; }"
        "QWidget { font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; }"
    );

    m_scanner = new MusicScanner(this);
}

void MainWindow::setupMenu() {
    auto *fileMenu = menuBar()->addMenu("文件");
    auto *openFolder = fileMenu->addAction("打开文件夹...");
    auto *addFiles = fileMenu->addAction("添加文件...");
    fileMenu->addSeparator();
    auto *exitAct = fileMenu->addAction("退出");

    connect(openFolder, &QAction::triggered, this, &MainWindow::onScanFolder);
    connect(addFiles, &QAction::triggered, this, [this]() {
        QStringList files = QFileDialog::getOpenFileNames(this, "添加音乐文件", SettingsManager::instance()->lastFolder(),
            "音频文件 (*.mp3 *.flac *.wav *.aac *.ogg *.m4a *.wma)");
        if (files.isEmpty()) return;
        QList<MusicItem> items;
        for (const QString &f : files) {
            MusicItem item;
            item.filePath = f;
            item.url = QUrl::fromLocalFile(f);
            item.title = QFileInfo(f).completeBaseName();
            items.append(item);
        }
        m_model->addItems(items);
    });
    connect(exitAct, &QAction::triggered, qApp, &QApplication::quit);

    auto *playMenu = menuBar()->addMenu("播放");
    auto *playAct = playMenu->addAction("播放/暂停");
    auto *nextAct = playMenu->addAction("下一首");
    auto *prevAct = playMenu->addAction("上一首");

    connect(playAct, &QAction::triggered, m_player, &PlayerCore::togglePlayPause);
    connect(nextAct, &QAction::triggered, m_player, &PlayerCore::next);
    connect(prevAct, &QAction::triggered, m_player, &PlayerCore::previous);
}

void MainWindow::connectSignals() {
    connect(m_sidebar, &NavigationSidebar::pageChanged, this, &MainWindow::onPageChanged);
    connect(m_playlistView, &PlaylistView::activated, m_player, qOverload<int>(&PlayerCore::play));
    connect(m_playlistView, &PlaylistView::removeRequested, this, [this](int row) {
        m_model->removeRows(row, 1);
    });
    connect(m_player, &PlayerCore::currentIndexChanged, this, &MainWindow::onCurrentIndexChanged);
    connect(m_player, &PlayerCore::mediaStatusChanged, this, &MainWindow::onMediaStatusChanged);
    connect(m_player->mediaPlayer(), &QMediaPlayer::metaDataChanged, this, [this]() {
        // Qt6: metadata is available after media is loaded
        auto *p = m_player->mediaPlayer();
        if (!p) return;
        // Extract cover
        QMediaMetaData md = p->metaData();
        QVariant cover = md.value(QMediaMetaData::ThumbnailImage);
        if (cover.isValid()) {
            QImage img = cover.value<QImage>();
            if (!img.isNull()) {
                m_albumCover->setCover(QPixmap::fromImage(img));
                // Also store back to model
                if (m_player->currentIndex() >= 0) {
                    QByteArray ba;
                    QBuffer buf(&ba);
                    buf.open(QIODevice::WriteOnly);
                    img.save(&buf, "JPEG");
                }
            }
        }
    });
    connect(m_scanner, &MusicScanner::itemFound, this, &MainWindow::onItemFound);
    connect(m_scanner, &MusicScanner::scanFinished, this, &MainWindow::onScanFinished);
}

void MainWindow::restoreAppState() {
    auto *s = SettingsManager::instance();
    restoreGeometry(s->windowGeometry());
    restoreState(s->windowState());
    m_player->setVolume(s->volume());
    m_player->setMuted(s->muted());
    m_player->setPlayMode(s->playMode());
    m_controls->volumeButton()->setVolume(s->volume());
    m_controls->volumeButton()->setMuted(s->muted());
}

void MainWindow::saveAppState() {
    auto *s = SettingsManager::instance();
    s->setWindowGeometry(saveGeometry());
    s->setWindowState(saveState());
    s->setVolume(m_player->volume());
    s->setMuted(m_player->isMuted());
    s->setPlayMode(m_player->playMode());
    s->setLastSongIndex(m_player->currentIndex());
    s->setLastPosition(m_player->position());
}

void MainWindow::closeEvent(QCloseEvent *event) {
    saveAppState();
    event->accept();
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event) {
    if (event->mimeData()->hasUrls())
        event->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent *event) {
    QList<MusicItem> items;
    QStringList folders;
    for (const QUrl &url : event->mimeData()->urls()) {
        QString path = url.toLocalFile();
        QFileInfo info(path);
        if (info.isDir()) {
            folders.append(path);
        } else if (info.isFile()) {
            MusicItem item;
            item.filePath = path;
            item.url = url;
            item.title = info.completeBaseName();
            items.append(item);
        }
    }
    if (!items.isEmpty())
        m_model->addItems(items);
    if (!folders.isEmpty())
        m_scanner->scanFolders(folders);
}

void MainWindow::onScanFolder() {
    QString folder = QFileDialog::getExistingDirectory(this, "选择音乐文件夹", SettingsManager::instance()->lastFolder());
    if (folder.isEmpty()) return;
    loadFolder(folder);
}

void MainWindow::loadFolder(const QString &folder) {
    SettingsManager::instance()->setLastFolder(folder);
    m_model->clear();
    m_scanner->scanFolder(folder);
}

void MainWindow::onScanFinished(int count) {
    Q_UNUSED(count)
    // Optionally show a toast or status message
}

void MainWindow::onItemFound(const MusicItem &item) {
    m_model->addItem(item);
}

void MainWindow::onCurrentIndexChanged(int index) {
    m_playlistView->setCurrentRow(index);
    MusicItem item = m_model->itemAt(index);

    // Load lyric
    m_lyricView->loadFromFile(item.filePath);

    // Clear cover first (will be updated by metaDataChanged signal)
    m_albumCover->clear();

    // Try to extract metadata using QMediaPlayer if not already loaded
    if (item.title.isEmpty() || item.artist.isEmpty() || item.durationMs == 0) {
        // Lazy metadata loading could be done here with a temporary QMediaPlayer
        // For now we rely on the playing player's metadataChanged signal to update UI
    }
}

void MainWindow::onPageChanged(NavigationSidebar::Page page) {
    m_stack->setCurrentIndex(static_cast<int>(page));
}

void MainWindow::onMediaStatusChanged(QMediaPlayer::MediaStatus status) {
    if (status == QMediaPlayer::LoadedMedia || status == QMediaPlayer::BufferedMedia) {
        // Update playlist item duration/title if missing
        QMediaPlayer *p = m_player->mediaPlayer();
        if (!p) return;
        QMediaMetaData md = p->metaData();
        QString title = md.stringValue(QMediaMetaData::Title);
        QString artist = md.stringValue(QMediaMetaData::ContributingArtist);
        QString album = md.stringValue(QMediaMetaData::AlbumTitle);
        qint64 dur = p->duration();

        int idx = m_player->currentIndex();
        if (idx < 0 || idx >= m_model->rowCount()) return;

        // We cannot easily edit model data here because PlaylistModel doesn't have setData.
        // For a fully polished app, add setData support.
        Q_UNUSED(title) Q_UNUSED(artist) Q_UNUSED(album) Q_UNUSED(dur)
    }
}
