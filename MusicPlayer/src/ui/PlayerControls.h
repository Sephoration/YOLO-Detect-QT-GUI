#pragma once
#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include "core/PlayerCore.h"
#include "ui/VolumeButton.h"

class PlayerControls : public QWidget {
    Q_OBJECT
public:
    explicit PlayerControls(PlayerCore *player, QWidget *parent = nullptr);

public slots:
    void onCurrentIndexChanged(int index);
    void onPositionChanged(qint64 position);
    void onDurationChanged(qint64 duration);
    void onPlaybackStateChanged(QMediaPlayer::PlaybackState state);
    void onPlayModeChanged(PlayMode mode);

    VolumeButton *volumeButton() const { return m_volumeBtn; }

private:
    PlayerCore *m_player;

    QLabel *m_coverLabel;
    QLabel *m_titleLabel;
    QLabel *m_artistLabel;
    QPushButton *m_playPauseBtn;
    QPushButton *m_prevBtn;
    QPushButton *m_nextBtn;
    QPushButton *m_modeBtn;
    QSlider *m_progressSlider;
    QLabel *m_timeLabel;
    VolumeButton *m_volumeBtn;

    bool m_sliderPressed = false;

    void setupUI();
    void updatePlayButton(bool playing);
    void updateModeButton();
    static QString formatTime(qint64 ms);
};
