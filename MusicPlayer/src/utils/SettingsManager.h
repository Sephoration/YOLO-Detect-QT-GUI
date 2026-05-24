#pragma once
#include <QObject>
#include <QSettings>
#include <QByteArray>
#include "core/PlayMode.h"

class SettingsManager : public QObject {
    Q_OBJECT
public:
    static SettingsManager *instance();

    float volume() const;
    void setVolume(float volume);

    bool muted() const;
    void setMuted(bool muted);

    PlayMode playMode() const;
    void setPlayMode(PlayMode mode);

    QString lastFolder() const;
    void setLastFolder(const QString &folder);

    int lastSongIndex() const;
    void setLastSongIndex(int index);

    qint64 lastPosition() const;
    void setLastPosition(qint64 pos);

    QByteArray windowGeometry() const;
    void setWindowGeometry(const QByteArray &geometry);

    QByteArray windowState() const;
    void setWindowState(const QByteArray &state);

private:
    explicit SettingsManager(QObject *parent = nullptr);
    QSettings *m_settings;
};
