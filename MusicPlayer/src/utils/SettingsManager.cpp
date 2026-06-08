#include "SettingsManager.h"

SettingsManager *SettingsManager::instance() {
    static SettingsManager inst;
    return &inst;
}

SettingsManager::SettingsManager(QObject *parent)
    : QObject(parent), m_settings(new QSettings("MusicPlayer", "MusicPlayer", this)) {}

float SettingsManager::volume() const {
    return m_settings->value("volume", 0.8f).toFloat();
}

void SettingsManager::setVolume(float volume) {
    m_settings->setValue("volume", volume);
}

bool SettingsManager::muted() const {
    return m_settings->value("muted", false).toBool();
}

void SettingsManager::setMuted(bool muted) {
    m_settings->setValue("muted", muted);
}

PlayMode SettingsManager::playMode() const {
    return static_cast<PlayMode>(m_settings->value("playMode", 0).toInt());
}

void SettingsManager::setPlayMode(PlayMode mode) {
    m_settings->setValue("playMode", static_cast<int>(mode));
}

QString SettingsManager::lastFolder() const {
    return m_settings->value("lastFolder").toString();
}

void SettingsManager::setLastFolder(const QString &folder) {
    m_settings->setValue("lastFolder", folder);
}

int SettingsManager::lastSongIndex() const {
    return m_settings->value("lastSongIndex", -1).toInt();
}

void SettingsManager::setLastSongIndex(int index) {
    m_settings->setValue("lastSongIndex", index);
}

qint64 SettingsManager::lastPosition() const {
    return m_settings->value("lastPosition", 0).toLongLong();
}

void SettingsManager::setLastPosition(qint64 pos) {
    m_settings->setValue("lastPosition", pos);
}

QByteArray SettingsManager::windowGeometry() const {
    return m_settings->value("geometry").toByteArray();
}

void SettingsManager::setWindowGeometry(const QByteArray &geometry) {
    m_settings->setValue("geometry", geometry);
}

QByteArray SettingsManager::windowState() const {
    return m_settings->value("windowState").toByteArray();
}

void SettingsManager::setWindowState(const QByteArray &state) {
    m_settings->setValue("windowState", state);
}
