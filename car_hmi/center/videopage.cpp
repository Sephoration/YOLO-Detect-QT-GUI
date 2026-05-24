#include "videopage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>

VideoPage::VideoPage(QWidget *parent) : QWidget(parent) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(16);

    // 视频主区域
    QWidget *videoMain = new QWidget(this);
    videoMain->setStyleSheet("background: #000000; border-radius: 16px;");
    QVBoxLayout *vm = new QVBoxLayout(videoMain);
    vm->setAlignment(Qt::AlignCenter);

    QWidget *placeholder = new QWidget(this);
    placeholder->setFixedSize(100, 100);
    QVBoxLayout *ph = new QVBoxLayout(placeholder);
    ph->setAlignment(Qt::AlignCenter);
    QPushButton *playBtn = new QPushButton("▶", this);
    playBtn->setFixedSize(80, 80);
    playBtn->setStyleSheet(
        "background: rgba(255,255,255,0.1); border-radius: 40px; color: #ffffff; font-size: 28px;"
    );
    ph->addWidget(playBtn);
    vm->addWidget(placeholder);

    QLabel *hint = new QLabel("行车中视频播放已禁用\n停车后可观看", this);
    hint->setAlignment(Qt::AlignCenter);
    hint->setStyleSheet("color: #8b8b9e; font-size: 14px; border: none; background: transparent;");
    vm->addWidget(hint);

    QLabel *title = new QLabel("行车记录  ▪  2024-05-20 14:20", this);
    title->setStyleSheet("color: #ffffff; font-size: 14px; font-weight: 500; border: none; background: transparent;");
    vm->addWidget(title);
    mainLayout->addWidget(videoMain, 1);

    // 缩略图列表
    QWidget *thumbArea = new QWidget(this);
    QHBoxLayout *thumbLayout = new QHBoxLayout(thumbArea);
    thumbLayout->setSpacing(12);

    struct Thumb { QString emoji, title; bool active; QColor c1, c2; };
    Thumb thumbs[] = {
        {"🎥", "行车记录仪 01", true, QColor(26,26,62), QColor(45,27,78)},
        {"🎥", "行车记录仪 02", false, QColor(26,62,26), QColor(27,78,45)},
        {"🎥", "停车监控 01", false, QColor(62,26,26), QColor(78,45,27)},
        {"🎬", "本地影片", false, QColor(26,26,26), QColor(45,45,45)}
    };
    for (int i = 0; i < 4; ++i) {
        QPushButton *btn = new QPushButton(this);
        btn->setFixedSize(160, 100);
        QString border = thumbs[i].active ? "#00d4ff" : "transparent";
        btn->setStyleSheet(QString(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 %1, stop:1 %2);"
            "border: 2px solid %3; border-radius: 12px;"
        ).arg(thumbs[i].c1.name()).arg(thumbs[i].c2.name()).arg(border));
        QVBoxLayout *bl = new QVBoxLayout(btn);
        bl->setAlignment(Qt::AlignCenter);
        QLabel *ico = new QLabel(thumbs[i].emoji, this);
        ico->setAlignment(Qt::AlignCenter);
        ico->setStyleSheet("font-size: 22px; border: none; background: transparent;");
        QLabel *tt = new QLabel(thumbs[i].title, this);
        tt->setAlignment(Qt::AlignCenter);
        tt->setStyleSheet("font-size: 11px; color: #ffffff; border: none; background: transparent;");
        bl->addWidget(ico);
        bl->addWidget(tt);
        thumbLayout->addWidget(btn);
    }
    thumbLayout->addStretch();
    mainLayout->addWidget(thumbArea);
}
