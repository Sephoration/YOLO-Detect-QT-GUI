#include "musicpage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QScrollArea>
#include <QWidget>

MusicPage::MusicPage(QWidget *parent) : QWidget(parent) {
    setupUI();
}

void MusicPage::setupUI() {
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(24, 16, 24, 16);
    mainLayout->setSpacing(32);

    // 左侧专辑区
    QWidget *left = new QWidget(this);
    left->setFixedWidth(300);
    QVBoxLayout *leftLayout = new QVBoxLayout(left);
    leftLayout->setAlignment(Qt::AlignCenter);

    QWidget *album = new QWidget(this);
    album->setFixedSize(240, 240);
    album->setStyleSheet(
        "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a3e, stop:0.5 #2d1b4e, stop:1 #1a1a3e);"
        "border-radius: 20px;"
    );
    QVBoxLayout *al = new QVBoxLayout(album);
    al->setAlignment(Qt::AlignCenter);
    QLabel *note = new QLabel("🎵", this);
    note->setAlignment(Qt::AlignCenter);
    note->setStyleSheet("font-size: 64px; border: none; background: transparent;");
    al->addWidget(note);
    leftLayout->addWidget(album);

    QLabel *track = new QLabel("夜曲", this);
    track->setAlignment(Qt::AlignCenter);
    track->setStyleSheet("font-size: 22px; font-weight: 500; border: none; background: transparent;");
    leftLayout->addWidget(track);

    QLabel *artist = new QLabel("周杰伦", this);
    artist->setAlignment(Qt::AlignCenter);
    artist->setStyleSheet("font-size: 14px; color: #8b8b9e; border: none; background: transparent;");
    leftLayout->addWidget(artist);
    leftLayout->addStretch();
    mainLayout->addWidget(left);

    // 右侧控制区
    QWidget *right = new QWidget(this);
    QVBoxLayout *rightLayout = new QVBoxLayout(right);
    rightLayout->setAlignment(Qt::AlignVCenter);
    rightLayout->setSpacing(24);

    // 进度条
    QWidget *progWidget = new QWidget(this);
    QVBoxLayout *progLayout = new QVBoxLayout(progWidget);
    progLayout->setContentsMargins(0, 0, 0, 0);
    m_progress = new QSlider(Qt::Horizontal, this);
    m_progress->setRange(0, 252);
    m_progress->setValue(112);
    m_progress->setStyleSheet(Styles::sliderStyle());
    connect(m_progress, &QSlider::sliderMoved, this, &MusicPage::onSliderMoved);
    progLayout->addWidget(m_progress);

    QHBoxLayout *timeLayout = new QHBoxLayout();
    m_timeCurrent = new QLabel("1:52", this);
    m_timeCurrent->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
    m_timeTotal = new QLabel("4:12", this);
    m_timeTotal->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
    timeLayout->addWidget(m_timeCurrent);
    timeLayout->addStretch();
    timeLayout->addWidget(m_timeTotal);
    progLayout->addLayout(timeLayout);
    rightLayout->addWidget(progWidget);

    // 控制按钮
    QHBoxLayout *ctrlLayout = new QHBoxLayout();
    ctrlLayout->setAlignment(Qt::AlignCenter);
    ctrlLayout->setSpacing(20);
    QString icons[] = {"🔀", "⏮", "▶", "⏭", "🔁"};
    for (int i = 0; i < 5; ++i) {
        QPushButton *btn = new QPushButton(icons[i], this);
        if (i == 2) {
            m_playBtn = btn;
            btn->setFixedSize(72, 72);
            btn->setStyleSheet(
                "background: #00d4ff; border-radius: 36px; color: #000; font-size: 24px;"
            );
            connect(btn, &QPushButton::clicked, this, &MusicPage::togglePlay);
        } else {
            btn->setFixedSize(48, 48);
            btn->setStyleSheet(
                "background: transparent; border: none; color: #8b8b9e; font-size: 18px;"
            );
        }
        ctrlLayout->addWidget(btn);
    }
    rightLayout->addLayout(ctrlLayout);

    // 播放列表
    QWidget *listWidget = new QWidget(this);
    QVBoxLayout *listLayout = new QVBoxLayout(listWidget);
    listLayout->setContentsMargins(0, 0, 0, 0);
    listLayout->setSpacing(4);

    struct Track { QString num, title, artist, dur; bool active; };
    Track tracks[] = {
        {"1", "夜曲", "周杰伦", "4:12", true},
        {"2", "晴天", "周杰伦", "4:29", false},
        {"3", "稻香", "周杰伦", "3:43", false},
        {"4", "青花瓷", "周杰伦", "3:57", false}
    };
    for (int i = 0; i < 4; ++i) {
        QPushButton *item = new QPushButton(this);
        item->setStyleSheet(Styles::playlistItemStyle(tracks[i].active));
        item->setFixedHeight(48);
        QHBoxLayout *il = new QHBoxLayout(item);
        il->setContentsMargins(12, 0, 12, 0);
        QLabel *num = new QLabel(tracks[i].num, this);
        num->setFixedWidth(24);
        num->setAlignment(Qt::AlignCenter);
        num->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
        QVBoxLayout *info = new QVBoxLayout();
        QLabel *t = new QLabel(tracks[i].title, this);
        t->setStyleSheet("font-size: 13px; border: none; background: transparent;");
        QLabel *a = new QLabel(tracks[i].artist, this);
        a->setStyleSheet("font-size: 11px; color: #8b8b9e; border: none; background: transparent;");
        info->addWidget(t);
        info->addWidget(a);
        QLabel *dur = new QLabel(tracks[i].dur, this);
        dur->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
        il->addWidget(num);
        il->addLayout(info, 1);
        il->addWidget(dur);
        listLayout->addWidget(item);
    }
    rightLayout->addWidget(listWidget);
    rightLayout->addStretch();
    mainLayout->addWidget(right, 1);
}

void MusicPage::togglePlay() {
    m_playing = !m_playing;
    m_playBtn->setText(m_playing ? "⏸" : "▶");
}

void MusicPage::onSliderMoved(int val) {
    int min = val / 60;
    int sec = val % 60;
    m_timeCurrent->setText(QString("%1:%2").arg(min).arg(sec, 2, 10, QChar('0')));
}
