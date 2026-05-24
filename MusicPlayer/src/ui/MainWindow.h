#pragma once
#include <QMainWindow>
#include <QStackedWidget>
#include "core/PlayerCore.h"
#include "core/PlaylistModel.h"
#include "ui/PlayerControls.h"
#include "ui/PlaylistView.h"
#include "ui/LyricView.h"
#include "ui/AlbumCover.h"
#include "ui/NavigationSidebar.h"
#include "utils/MusicScanner.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private slots:
    void onScanFolder();
    void onScanFinished(int count);
    void onItemFound(const MusicItem &item);
    void onCurrentIndexChanged(int index);
    void onPageChanged(NavigationSidebar::Page page);
    void onMediaStatusChanged(QMediaPlayer::MediaStatus status);

private:
    PlaylistModel *m_model;
    PlayerCore *m_player;
    MusicScanner *m_scanner;

    NavigationSidebar *m_sidebar;
    QStackedWidget *m_stack;
    PlaylistView *m_playlistView;
    QWidget *m_libraryPage;
    QWidget *m_favoritesPage;

    AlbumCover *m_albumCover;
    LyricView *m_lyricView;
    PlayerControls *m_controls;

    void setupUI();
    void setupMenu();
    void connectSignals();
    void restoreAppState();
    void saveAppState();
    void loadFolder(const QString &folder);
};
