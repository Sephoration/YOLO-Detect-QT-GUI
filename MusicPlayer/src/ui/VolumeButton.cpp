#include "VolumeButton.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QPushButton>

VolumeButton::VolumeButton(QWidget *parent)
    : QWidget(parent) {
    auto *layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);

    auto *btn = new QPushButton(this);
    btn->setFixedSize(24, 24);
    btn->setText("🔊");
    btn->setFlat(true);
    btn->setStyleSheet("QPushButton { color: #ccc; font-size: 14px; } QPushButton:hover { color: #fff; }");

    m_slider = new QSlider(Qt::Horizontal, this);
    m_slider->setRange(0, 100);
    m_slider->setValue(80);
    m_slider->setFixedWidth(80);
    m_slider->setStyleSheet(
        "QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }"
        "QSlider::sub-page:horizontal { background: #00d4ff; border-radius: 2px; }"
        "QSlider::handle:horizontal { width: 10px; background: #fff; border-radius: 5px; margin: -3px 0; }"
    );

    layout->addWidget(btn);
    layout->addWidget(m_slider);

    connect(btn, &QPushButton::clicked, this, [this, btn]() {
        m_muted = !m_muted;
        btn->setText(m_muted ? "🔇" : (m_volume < 0.3f ? "🔈" : "🔊"));
        emit muteToggled(m_muted);
    });

    connect(m_slider, &QSlider::valueChanged, this, [this, btn](int value) {
        m_volume = value / 100.0f;
        btn->setText(m_muted ? "🔇" : (m_volume < 0.3f ? "🔈" : "🔊"));
        emit volumeChanged(m_volume);
    });
}

float VolumeButton::volume() const {
    return m_volume;
}

void VolumeButton::setVolume(float volume) {
    m_volume = qBound(0.0f, volume, 1.0f);
    m_slider->setValue(static_cast<int>(m_volume * 100));
}

void VolumeButton::setMuted(bool muted) {
    m_muted = muted;
}
