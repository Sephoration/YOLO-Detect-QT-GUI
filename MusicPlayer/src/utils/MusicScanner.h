#pragma once
#include <QObject>
#include <QList>
#include "core/MusicItem.h"

class MusicScanner : public QObject {
    Q_OBJECT
public:
    explicit MusicScanner(QObject *parent = nullptr);

    void scanFolder(const QString &folderPath);
    void scanFolders(const QStringList &folderPaths);
    void stop();

signals:
    void itemFound(const MusicItem &item);
    void scanFinished(int count);
    void scanError(const QString &error);

private:
    bool m_stopped = false;
    static bool isAudioFile(const QString &filePath);
    static MusicItem extractMetadata(const QString &filePath);
};
