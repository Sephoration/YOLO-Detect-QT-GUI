#include "settingspage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>

SettingsPage::SettingsPage(QWidget *parent) : QWidget(parent) {
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(20);

    // 左侧导航
    QWidget *nav = new QWidget(this);
    nav->setFixedWidth(220);
    QVBoxLayout *nl = new QVBoxLayout(nav);
    nl->setContentsMargins(0, 0, 0, 0);
    nl->setAlignment(Qt::AlignTop);
    nl->setSpacing(4);

    struct NavItem { QString icon, label; bool active; };
    NavItem items[] = {
        {"🖥", "显示", true},
        {"🔊", "声音", false},
        {"🔗", "连接", false},
        {"🚗", "驾驶", false},
        {"🔒", "隐私", false},
        {"ℹ", "关于", false}
    };
    for (int i = 0; i < 6; ++i) {
        QPushButton *btn = new QPushButton(QString("%1  %2").arg(items[i].icon).arg(items[i].label), this);
        btn->setStyleSheet(items[i].active
            ? "background: #1a1a24; color: #ffffff; border-left: 3px solid #00d4ff; border-top: none; border-right: none; border-bottom: none; border-radius: 0px; text-align: left; padding: 12px 16px; font-size: 13px;"
            : "background: transparent; color: #8b8b9e; border: none; text-align: left; padding: 12px 16px; font-size: 13px;"
        );
        nl->addWidget(btn);
    }
    nl->addStretch();
    mainLayout->addWidget(nav);

    // 右侧内容
    QWidget *content = new QWidget(this);
    content->setStyleSheet(Styles::widgetCardStyle());
    QVBoxLayout *cl = new QVBoxLayout(content);
    cl->setContentsMargins(24, 20, 24, 20);
    cl->setSpacing(20);

    QLabel *title = new QLabel("显示设置", this);
    title->setStyleSheet("font-size: 16px; font-weight: 500; border: none; background: transparent;");
    cl->addWidget(title);

    struct Setting { QString title, desc; bool on; };
    Setting settings[] = {
        {"自动亮度", "根据环境光自动调节屏幕亮度", true},
        {"深色模式", "使用深色主题减少夜间眩光", true},
        {"护眼模式", "降低蓝光，保护视力", false}
    };
    for (int i = 0; i < 3; ++i) {
        QWidget *item = new QWidget(this);
        item->setStyleSheet(Styles::settingItemStyle());
        QHBoxLayout *il = new QHBoxLayout(item);
        il->setContentsMargins(16, 12, 16, 12);
        QVBoxLayout *info = new QVBoxLayout();
        QLabel *t = new QLabel(settings[i].title, this);
        t->setStyleSheet("font-size: 14px; font-weight: 500; border: none; background: transparent;");
        QLabel *d = new QLabel(settings[i].desc, this);
        d->setStyleSheet("font-size: 11px; color: #8b8b9e; border: none; background: transparent;");
        info->addWidget(t);
        info->addWidget(d);
        il->addLayout(info, 1);
        QPushButton *toggle = new QPushButton(this);
        toggle->setFixedSize(52, 28);
        toggle->setStyleSheet(Styles::toggleStyle(settings[i].on));
        toggle->setCheckable(true);
        toggle->setChecked(settings[i].on);
        connect(toggle, &QPushButton::toggled, this, [toggle](bool checked) {
            toggle->setStyleSheet(Styles::toggleStyle(checked));
        });
        il->addWidget(toggle);
        cl->addWidget(item);
    }

    // 亮度滑块
    QWidget *bright = new QWidget(this);
    QVBoxLayout *bl = new QVBoxLayout(bright);
    bl->setContentsMargins(0, 8, 0, 0);
    QLabel *blabel = new QLabel("屏幕亮度", this);
    blabel->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
    bl->addWidget(blabel);
    QSlider *slider = new QSlider(Qt::Horizontal, this);
    slider->setRange(0, 100);
    slider->setValue(70);
    slider->setStyleSheet(Styles::sliderStyle());
    bl->addWidget(slider);
    cl->addWidget(bright);
    cl->addStretch();
    mainLayout->addWidget(content, 1);
}
