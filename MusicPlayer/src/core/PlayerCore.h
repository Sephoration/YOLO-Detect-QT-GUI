#pragma once
#include <QObject>
#include <QMediaPlayer>
#include <QAudioOutput>
#include "PlayMode.h"
#include "MusicItem.h"

class PlaylistModel;

class PlayerCore : public QObject {
    Q_OBJECT
public:
    explicit PlayerCore(PlaylistModel *model, QObject *parent = nullptr);

    void play(int index);
    void play();
    void pause();
    void togglePlayPause();
    void stop();
    void next();
    void previous();

    void setPosition(qint64 ms);
    void setVolume(float volume);
    void setMuted(bool muted);
    void setPlayMode(PlayMode mode);

    int currentIndex() const { return m_currentIndex; }
    bool isPlaying() const;
    bool isPaused() const;
    qint64 position() const;
    qint64 duration() const;
    float volume() const;
    bool isMuted() const;
    PlayMode playMode() const { return m_playMode; }
    MusicItem currentItem() const;

    QMediaPlayer::PlaybackState playbackState() const;
    QMediaPlayer::MediaStatus mediaStatus() const;
    QMediaPlayer *mediaPlayer() const { return m_player; }

signals:
    void currentIndexChanged(int index);
    void positionChanged(qint64 position);
    void durationChanged(qint64 duration);
    void playbackStateChanged(QMediaPlayer::PlaybackState state);
    void volumeChanged(float volume);
    void mutedChanged(bool muted);
    void playModeChanged(PlayMode mode);
    void mediaStatusChanged(QMediaPlayer::MediaStatus status);
    void errorOccurred(const QString &error);

private slots:
    void onMediaStatusChanged(QMediaPlayer::MediaStatus status);
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState state);
    void onDurationChanged(qint64 duration);
    void onErrorOccurred(QMediaPlayer::Error error, const QString &errorString);

private:
    QMediaPlayer *m_player;
    QAudioOutput *m_audioOutput;
    PlaylistModel *m_model;
    int m_currentIndex = -1;
    PlayMode m_playMode = PlayMode::Sequential;
    QList<int> m_shuffleHistory;
    bool m_ignoreAutoNext = false;

    void doPlay(int index);
    void updateShuffleHistory(int index);
};
