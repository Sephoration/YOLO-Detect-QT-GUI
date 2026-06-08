#pragma once
#include <QWidget>
#include <QSlider>

class VolumeButton : public QWidget {
    Q_OBJECT
public:
    explicit VolumeButton(QWidget *parent = nullptr);

    float volume() const;

signals:
    void volumeChanged(float volume);
    void muteToggled(bool muted);

public slots:
    void setVolume(float volume);
    void setMuted(bool muted);

private:
    QSlider *m_slider;
    float m_volume = 1.0f;
    bool m_muted = false;
};
