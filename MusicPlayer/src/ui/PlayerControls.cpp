#include "PlayerControls.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QFileInfo>
#include <QPushButton>
#include <QMediaPlayer>

PlayerControls::PlayerControls(PlayerCore *player, QWidget *parent)
    : QWidget(parent), m_player(player) {
    setupUI();
}

void PlayerControls::setupUI() {
    setFixedHeight(80);
    setStyleSheet("background: #12121a; border-top: 1px solid #2a2a3a;");

    auto *mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(16, 8, 16, 8);
    mainLayout->setSpacing(16);

    // Left: cover + info
    auto *infoLayout = new QHBoxLayout();
    infoLayout->setSpacing(10);

    m_coverLabel = new QLabel(this);
    m_coverLabel->setFixedSize(48, 48);
    m_coverLabel->setStyleSheet("background: #1e1e2a; border-radius: 4px;");
    m_coverLabel->setAlignment(Qt::AlignCenter);
    m_coverLabel->setText("🎵");

    auto *textLayout = new QVBoxLayout();
    textLayout->setSpacing(2);
    m_titleLabel = new QLabel("未在播放", this);
    m_titleLabel->setStyleSheet("color: #fff; font-size: 13px; font-weight: 500;");
    m_artistLabel = new QLabel("--", this);
    m_artistLabel->setStyleSheet("color: #888; font-size: 11px;");
    textLayout->addWidget(m_titleLabel);
    textLayout->addWidget(m_artistLabel);

    infoLayout->addWidget(m_coverLabel);
    infoLayout->addLayout(textLayout);
    infoLayout->addStretch();
    mainLayout->addLayout(infoLayout, 2);

    // Center: controls + progress
    auto *centerLayout = new QVBoxLayout();
    centerLayout->setSpacing(6);
    centerLayout->setAlignment(Qt::AlignCenter);

    auto *btnLayout = new QHBoxLayout();
    btnLayout->setSpacing(12);
    btnLayout->setAlignment(Qt::AlignCenter);

    auto createBtn = [this](const QString &text) -> QPushButton* {
        auto *btn = new QPushButton(text, this);
        btn->setFixedSize(32, 32);
        btn->setFlat(true);
        btn->setStyleSheet(
            "QPushButton { color: #aaa; font-size: 14px; }"
            "QPushButton:hover { color: #fff; }"
            "QPushButton:pressed { color: #00d4ff; }"
        );
        return btn;
    };

    m_modeBtn = createBtn("➡️");
    m_prevBtn = createBtn("⏮");
    m_playPauseBtn = createBtn("▶");
    m_playPauseBtn->setFixedSize(40, 40);
    m_playPauseBtn->setStyleSheet(
        "QPushButton { background: #00d4ff; color: #000; border-radius: 20px; font-size: 16px; }"
        "QPushButton:hover { background: #33ddff; }"
    );
    m_nextBtn = createBtn("⏭");

    btnLayout->addWidget(m_modeBtn);
    btnLayout->addWidget(m_prevBtn);
    btnLayout->addWidget(m_playPauseBtn);
    btnLayout->addWidget(m_nextBtn);

    auto *progLayout = new QHBoxLayout();
    progLayout->setSpacing(8);
    m_timeLabel = new QLabel("0:00 / 0:00", this);
    m_timeLabel->setStyleSheet("color: #888; font-size: 11px; font-family: monospace;");
    m_timeLabel->setFixedWidth(90);

    m_progressSlider = new QSlider(Qt::Horizontal, this);
    m_progressSlider->setRange(0, 0);
    m_progressSlider->setStyleSheet(
        "QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }"
        "QSlider::sub-page:horizontal { background: #00d4ff; border-radius: 2px; }"
        "QSlider::handle:horizontal { width: 12px; background: #fff; border-radius: 6px; margin: -4px 0; }"
        "QSlider::handle:horizontal:hover { background: #00d4ff; }"
    );

    progLayout->addWidget(m_timeLabel);
    progLayout->addWidget(m_progressSlider);

    centerLayout->addLayout(btnLayout);
    centerLayout->addLayout(progLayout);
    mainLayout->addLayout(centerLayout, 5);

    // Right: volume
    m_volumeBtn = new VolumeButton(this);
    mainLayout->addWidget(m_volumeBtn, 1);

    // Connections
    connect(m_playPauseBtn, &QPushButton::clicked, m_player, &PlayerCore::togglePlayPause);
    connect(m_prevBtn, &QPushButton::clicked, m_player, &PlayerCore::previous);
    connect(m_nextBtn, &QPushButton::clicked, m_player, &PlayerCore::next);
    connect(m_modeBtn, &QPushButton::clicked, this, [this]() {
        PlayMode m = m_player->playMode();
        int next = (static_cast<int>(m) + 1) % 4;
        m_player->setPlayMode(static_cast<PlayMode>(next));
    });

    connect(m_progressSlider, &QSlider::sliderPressed, this, [this]() { m_sliderPressed = true; });
    connect(m_progressSlider, &QSlider::sliderReleased, this, [this]() {
        m_sliderPressed = false;
        m_player->setPosition(m_progressSlider->value());
    });
    connect(m_progressSlider, &QSlider::valueChanged, this, [this](int val) {
        if (m_sliderPressed) {
            m_timeLabel->setText(QString("%1 / %2")
                .arg(formatTime(val))
                .arg(formatTime(m_player->duration())));
        }
    });

    connect(m_volumeBtn, &VolumeButton::volumeChanged, m_player, &PlayerCore::setVolume);
    connect(m_volumeBtn, &VolumeButton::muteToggled, m_player, &PlayerCore::setMuted);

    connect(m_player, &PlayerCore::positionChanged, this, &PlayerControls::onPositionChanged);
    connect(m_player, &PlayerCore::durationChanged, this, &PlayerControls::onDurationChanged);
    connect(m_player, &PlayerCore::playbackStateChanged, this, &PlayerControls::onPlaybackStateChanged);
    connect(m_player, &PlayerCore::currentIndexChanged, this, &PlayerControls::onCurrentIndexChanged);
    connect(m_player, &PlayerCore::playModeChanged, this, &PlayerControls::onPlayModeChanged);
}

void PlayerControls::onCurrentIndexChanged(int index) {
    MusicItem item = m_player->currentItem();
    if (index < 0 || !item.isValid()) {
        m_titleLabel->setText("未在播放");
        m_artistLabel->setText("--");
        m_coverLabel->setText("🎵");
        return;
    }
    m_titleLabel->setText(item.title.isEmpty() ? QFileInfo(item.filePath).fileName() : item.title);
    m_artistLabel->setText(item.artist.isEmpty() ? "未知艺术家" : item.artist);
}

void PlayerControls::onPositionChanged(qint64 position) {
    if (!m_sliderPressed) {
        m_progressSlider->setValue(static_cast<int>(position));
        m_timeLabel->setText(QString("%1 / %2")
            .arg(formatTime(position))
            .arg(formatTime(m_player->duration())));
    }
}

void PlayerControls::onDurationChanged(qint64 duration) {
    m_progressSlider->setRange(0, static_cast<int>(duration));
    m_timeLabel->setText(QString("%1 / %2")
        .arg(formatTime(m_player->position()))
        .arg(formatTime(duration)));
}

void PlayerControls::onPlaybackStateChanged(QMediaPlayer::PlaybackState state) {
    updatePlayButton(state == QMediaPlayer::PlayingState);
}

void PlayerControls::onPlayModeChanged(PlayMode mode) {
    switch (mode) {
    case PlayMode::Sequential: m_modeBtn->setText("➡️"); break;
    case PlayMode::Loop: m_modeBtn->setText("🔁"); break;
    case PlayMode::SingleLoop: m_modeBtn->setText("🔂"); break;
    case PlayMode::Shuffle: m_modeBtn->setText("🔀"); break;
    }
}

void PlayerControls::updatePlayButton(bool playing) {
    m_playPauseBtn->setText(playing ? "⏸" : "▶");
}

QString PlayerControls::formatTime(qint64 ms) {
    if (ms < 0) return "0:00";
    int totalSec = static_cast<int>(ms / 1000);
    int min = totalSec / 60;
    int sec = totalSec % 60;
    return QString("%1:%2").arg(min).arg(sec, 2, 10, QLatin1Char('0'));
}
