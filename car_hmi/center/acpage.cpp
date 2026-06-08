#include "acpage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>

ACPage::ACPage(QWidget *parent) : QWidget(parent) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(24, 16, 24, 16);
    mainLayout->setAlignment(Qt::AlignCenter);
    mainLayout->setSpacing(32);

    // 温度控制区
    QHBoxLayout *tempLayout = new QHBoxLayout();
    tempLayout->setSpacing(48);
    tempLayout->setAlignment(Qt::AlignCenter);

    for (int side = 0; side < 2; ++side) {
        QVBoxLayout *sideLayout = new QVBoxLayout();
        sideLayout->setAlignment(Qt::AlignCenter);
        sideLayout->setSpacing(12);

        QPushButton *plus = new QPushButton("+", this);
        plus->setStyleSheet(Styles::tempButtonStyle());
        connect(plus, &QPushButton::clicked, this, [this, side]() { adjustTemp(side * 1000 + 1); });
        sideLayout->addWidget(plus, 0, Qt::AlignCenter);

        QWidget *knob = new QWidget(this);
        knob->setFixedSize(160, 160);
        knob->setStyleSheet(
            "background: #1a1a24; border: 3px solid #2a2a3a; border-radius: 80px;"
        );
        QVBoxLayout *kl = new QVBoxLayout(knob);
        kl->setAlignment(Qt::AlignCenter);
        m_tempLabels[side] = new QLabel("22", this);
        m_tempLabels[side]->setAlignment(Qt::AlignCenter);
        m_tempLabels[side]->setStyleSheet("font-size: 44px; font-weight: 200; border: none; background: transparent;");
        kl->addWidget(m_tempLabels[side]);
        QLabel *unit = new QLabel("°C", this);
        unit->setAlignment(Qt::AlignCenter);
        unit->setStyleSheet("color: #8b8b9e; font-size: 14px; border: none; background: transparent;");
        kl->addWidget(unit);
        sideLayout->addWidget(knob, 0, Qt::AlignCenter);

        QPushButton *minus = new QPushButton("−", this);
        minus->setStyleSheet(Styles::tempButtonStyle());
        connect(minus, &QPushButton::clicked, this, [this, side]() { adjustTemp(side * 1000 - 1); });
        sideLayout->addWidget(minus, 0, Qt::AlignCenter);

        QLabel *name = new QLabel(side == 0 ? "主驾" : "副驾", this);
        name->setAlignment(Qt::AlignCenter);
        name->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
        sideLayout->addWidget(name);

        tempLayout->addLayout(sideLayout);
    }
    mainLayout->addLayout(tempLayout);

    // 模式选择
    QHBoxLayout *modeLayout = new QHBoxLayout();
    modeLayout->setSpacing(16);
    modeLayout->setAlignment(Qt::AlignCenter);
    QString modes[][2] = {{"❄", "制冷"}, {"☀", "制热"}, {"💨", "送风"}, {"🔄", "内循环"}, {"💺", "座椅加热"}};
    for (int i = 0; i < 5; ++i) {
        QPushButton *btn = new QPushButton(this);
        btn->setFixedSize(80, 80);
        btn->setStyleSheet(Styles::modeButtonStyle(i == 0));
        QVBoxLayout *bl = new QVBoxLayout(btn);
        bl->setAlignment(Qt::AlignCenter);
        bl->setSpacing(4);
        QLabel *ico = new QLabel(modes[i][0], this);
        ico->setAlignment(Qt::AlignCenter);
        ico->setStyleSheet("font-size: 22px; border: none; background: transparent;");
        QLabel *lab = new QLabel(modes[i][1], this);
        lab->setAlignment(Qt::AlignCenter);
        lab->setStyleSheet("font-size: 10px; border: none; background: transparent;");
        bl->addWidget(ico);
        bl->addWidget(lab);
        connect(btn, &QPushButton::clicked, this, [this, i]() { setMode(i); });
        modeLayout->addWidget(btn);
    }
    mainLayout->addLayout(modeLayout);

    // 风量控制
    QWidget *fanWidget = new QWidget(this);
    fanWidget->setMaximumWidth(500);
    QVBoxLayout *fanLayout = new QVBoxLayout(fanWidget);
    fanLayout->setContentsMargins(0, 0, 0, 0);
    QHBoxLayout *fanLabels = new QHBoxLayout();
    QLabel *fl1 = new QLabel("风量", this);
    fl1->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
    QLabel *fl2 = new QLabel("3级", this);
    fl2->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
    fanLabels->addWidget(fl1);
    fanLabels->addStretch();
    fanLabels->addWidget(fl2);
    fanLayout->addLayout(fanLabels);

    QHBoxLayout *fanBar = new QHBoxLayout();
    fanBar->setSpacing(6);
    for (int i = 0; i < 7; ++i) {
        QWidget *seg = new QWidget(this);
        seg->setFixedHeight(40);
        bool active = i < 3;
        seg->setStyleSheet(active
            ? "background: #00d4ff; border-radius: 6px;"
            : "background: rgba(255,255,255,0.05); border-radius: 6px;");
        fanBar->addWidget(seg);
    }
    fanLayout->addLayout(fanBar);
    mainLayout->addWidget(fanWidget, 0, Qt::AlignCenter);
    mainLayout->addStretch();
}

void ACPage::setMode(int mode) {
    m_currentMode = mode;
    // 实际应用中可以更新UI状态
}

void ACPage::adjustTemp(int code) {
    int side = code >= 1000 ? 1 : 0;
    int delta = code >= 1000 ? code - 1000 : code;
    m_temps[side] = qBound(16, m_temps[side] + delta, 30);
    m_tempLabels[side]->setText(QString::number(m_temps[side]));
}
