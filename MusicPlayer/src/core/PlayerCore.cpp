#include "PlayerCore.h"
#include "PlaylistModel.h"
#include <QFileInfo>

PlayerCore::PlayerCore(PlaylistModel *model, QObject *parent)
    : QObject(parent), m_model(model), m_currentIndex(-1) {
    m_audioOutput = new QAudioOutput(this);
    m_player = new QMediaPlayer(this);
    m_player->setAudioOutput(m_audioOutput);

    connect(m_player, &QMediaPlayer::mediaStatusChanged,
            this, &PlayerCore::onMediaStatusChanged);
    connect(m_player, &QMediaPlayer::playbackStateChanged,
            this, &PlayerCore::onPlaybackStateChanged);
    connect(m_player, &QMediaPlayer::positionChanged,
            this, &PlayerCore::positionChanged);
    connect(m_player, &QMediaPlayer::durationChanged,
            this, &PlayerCore::onDurationChanged);
    connect(m_player, &QMediaPlayer::errorOccurred,
            this, &PlayerCore::onErrorOccurred);
}

void PlayerCore::play(int index) {
    if (index < 0 || index >= m_model->rowCount()) return;
    doPlay(index);
}

void PlayerCore::play() {
    if (m_currentIndex < 0 && m_model->rowCount() > 0)
        doPlay(0);
    else
        m_player->play();
}

void PlayerCore::pause() {
    m_player->pause();
}

void PlayerCore::togglePlayPause() {
    if (m_player->playbackState() == QMediaPlayer::PlayingState)
        pause();
    else
        play();
}

void PlayerCore::stop() {
    m_player->stop();
}

void PlayerCore::next() {
    if (m_model->rowCount() == 0) return;

    bool loop = (m_playMode == PlayMode::Loop || m_playMode == PlayMode::Shuffle);
    bool shuffle = (m_playMode == PlayMode::Shuffle);
    int nextIdx = m_model->nextIndex(m_currentIndex, loop, shuffle, nullptr);
    if (nextIdx >= 0)
        doPlay(nextIdx);
}

void PlayerCore::previous() {
    if (m_model->rowCount() == 0) return;

    bool shuffle = (m_playMode == PlayMode::Shuffle);
    int prevIdx = m_model->previousIndex(m_currentIndex, shuffle, m_shuffleHistory);
    if (prevIdx >= 0)
        doPlay(prevIdx);
}

void PlayerCore::setPosition(qint64 ms) {
    m_player->setPosition(ms);
}

void PlayerCore::setVolume(float volume) {
    m_audioOutput->setVolume(volume);
    emit volumeChanged(volume);
}

void PlayerCore::setMuted(bool muted) {
    m_audioOutput->setMuted(muted);
    emit mutedChanged(muted);
}

void PlayerCore::setPlayMode(PlayMode mode) {
    if (m_playMode == mode) return;
    m_playMode = mode;
    emit playModeChanged(mode);
}

bool PlayerCore::isPlaying() const {
    return m_player->playbackState() == QMediaPlayer::PlayingState;
}

bool PlayerCore::isPaused() const {
    return m_player->playbackState() == QMediaPlayer::PausedState;
}

qint64 PlayerCore::position() const {
    return m_player->position();
}

qint64 PlayerCore::duration() const {
    return m_player->duration();
}

float PlayerCore::volume() const {
    return m_audioOutput->volume();
}

bool PlayerCore::isMuted() const {
    return m_audioOutput->isMuted();
}

MusicItem PlayerCore::currentItem() const {
    return m_model->itemAt(m_currentIndex);
}

QMediaPlayer::PlaybackState PlayerCore::playbackState() const {
    return m_player->playbackState();
}

QMediaPlayer::MediaStatus PlayerCore::mediaStatus() const {
    return m_player->mediaStatus();
}

void PlayerCore::onMediaStatusChanged(QMediaPlayer::MediaStatus status) {
    emit mediaStatusChanged(status);

    if (status == QMediaPlayer::EndOfMedia) {
        if (m_playMode == PlayMode::SingleLoop) {
            m_player->setPosition(0);
            m_player->play();
        } else {
            next();
        }
    }
}

void PlayerCore::onPlaybackStateChanged(QMediaPlayer::PlaybackState state) {
    emit playbackStateChanged(state);
}

void PlayerCore::onDurationChanged(qint64 duration) {
    emit durationChanged(duration);
}

void PlayerCore::onErrorOccurred(QMediaPlayer::Error error, const QString &errorString) {
    Q_UNUSED(error)
    emit errorOccurred(errorString);
}

void PlayerCore::doPlay(int index) {
    MusicItem item = m_model->itemAt(index);
    if (!item.isValid()) return;

    bool sameMedia = (m_currentIndex == index);
    m_currentIndex = index;

    if (!sameMedia) {
        m_player->setSource(item.url);
    }
    m_player->play();
    updateShuffleHistory(index);
    emit currentIndexChanged(index);
}

void PlayerCore::updateShuffleHistory(int index) {
    m_shuffleHistory.append(index);
    if (m_shuffleHistory.size() > 100)
        m_shuffleHistory.removeFirst();
}
