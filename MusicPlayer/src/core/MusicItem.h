#pragma once
#include <QString>
#include <QUrl>
#include <QMetaType>

struct MusicItem {
    QString title;
    QString artist;
    QString album;
    QString filePath;
    QUrl url;
    qint64 durationMs = 0;
    QString durationStr;
    QByteArray coverData;

    bool isValid() const { return !filePath.isEmpty(); }
};

Q_DECLARE_METATYPE(MusicItem)
