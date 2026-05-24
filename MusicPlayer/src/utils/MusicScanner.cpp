#include "MusicScanner.h"
#include <QDirIterator>
#include <QMediaPlayer>
#include <QEventLoop>
#include <QMimeDatabase>
#include <QFileInfo>

MusicScanner::MusicScanner(QObject *parent) : QObject(parent) {}

void MusicScanner::scanFolder(const QString &folderPath) {
    scanFolders(QStringList() << folderPath);
}

void MusicScanner::scanFolders(const QStringList &folderPaths) {
    m_stopped = false;
    int count = 0;

    for (const QString &folder : folderPaths) {
        if (m_stopped) break;

        QDirIterator it(folder, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            if (m_stopped) break;

            QString filePath = it.next();
            if (it.fileInfo().isDir()) continue;
            if (!isAudioFile(filePath)) continue;

            MusicItem item = extractMetadata(filePath);
            if (item.isValid()) {
                emit itemFound(item);
                ++count;
            }
        }
    }

    emit scanFinished(count);
}

void MusicScanner::stop() {
    m_stopped = true;
}

bool MusicScanner::isAudioFile(const QString &filePath) {
    static const QStringList exts = {"mp3", "flac", "wav", "aac", "ogg", "m4a", "wma", "ape"};
    return exts.contains(QFileInfo(filePath).suffix().toLower());
}

MusicItem MusicScanner::extractMetadata(const QString &filePath) {
    MusicItem item;
    item.filePath = filePath;
    item.url = QUrl::fromLocalFile(filePath);

    QFileInfo info(filePath);
    item.title = info.completeBaseName();

    // Use QMediaPlayer to get metadata asynchronously... but for scanning we need sync.
    // Qt6 does not provide a synchronous metadata reader easily.
    // We'll do a quick async fetch with event loop for each file (not ideal for thousands).
    // For better performance, consider TagLib or leaving duration to be loaded lazily.

    // Lazy approach: fill only path and title now, let PlayerCore load metadata on play.
    // But let's try to get basic info from Qt if possible.

    return item;
}
