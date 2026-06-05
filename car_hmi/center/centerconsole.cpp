#include "centerconsole.h"
#include "homepage.h"
#include "musicpage.h"
#include "videopage.h"
#include "acpage.h"
#include "vehiclepage.h"
#include "navpage.h"
#include "settingspage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QDateTime>
#include <QTimer>

class HeaderBar : public QWidget {
public:
    explicit HeaderBar(QWidget *parent = nullptr) : QWidget(parent) {
        setFixedHeight(56);
        setStyleSheet("border-bottom: 1px solid #2a2a3a; background: transparent;");
        QHBoxLayout *layout = new QHBoxLayout(this);
        layout->setContentsMargins(24, 0, 24, 0);

        QWidget *left = new QWidget(this);
        QHBoxLayout *ll = new QHBoxLayout(left);
        ll->setContentsMargins(0, 0, 0, 0);
        ll->setSpacing(8);
        m_time = new QLabel(this);
        m_time->setStyleSheet("font-size: 26px; font-weight: 300; color: #ffffff; border: none; background: transparent;");
        ll->addWidget(m_time);
        m_date = new QLabel(this);
        m_date->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent; padding-top: 6px;");
        ll->addWidget(m_date);
        layout->addWidget(left);
        layout->addStretch();

        QWidget *right = new QWidget(this);
        QHBoxLayout *rl = new QHBoxLayout(right);
        rl->setContentsMargins(0, 0, 0, 0);
        rl->setSpacing(16);
        QString icons[] = {"📶", "🔵", "☀ 24°C 北京", "👤"};
        for (int i = 0; i < 4; ++i) {
            QLabel *l = new QLabel(icons[i], this);
            l->setStyleSheet("color: #8b8b9e; font-size: 13px; border: none; background: transparent;");
            rl->addWidget(l);
        }
        layout->addWidget(right);

        QTimer *t = new QTimer(this);
        connect(t, &QTimer::timeout, this, [this]() {
            QDateTime now = QDateTime::currentDateTime();
            m_time->setText(now.toString("HH:mm"));
            m_date->setText(now.toString("M月d日 dddd"));
        });
        t->start(1000);
        m_time->setText(QDateTime::currentDateTime().toString("HH:mm"));
        m_date->setText(QDateTime::currentDateTime().toString("M月d日 dddd"));
    }
private:
    QLabel *m_time = nullptr;
    QLabel *m_date = nullptr;
};

CenterConsole::CenterConsole(QWidget *parent) : QWidget(parent) {
    setupUI();
}

void CenterConsole::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // 顶部状态栏
    HeaderBar *header = new HeaderBar(this);
    mainLayout->addWidget(header);

    // 页面栈
    m_stack = new QStackedWidget(this);
    m_stack->setStyleSheet("background: #0a0a0f; border: none;");

    HomePage *home = new HomePage(this);
    connect(home, &HomePage::requestSwitchPage, this, &CenterConsole::switchPage);
    m_stack->addWidget(home);
    m_stack->addWidget(new NavPage(this));
    m_stack->addWidget(new MusicPage(this));
    m_stack->addWidget(new VideoPage(this));
    m_stack->addWidget(new ACPage(this));
    m_stack->addWidget(new VehiclePage(this));
    m_stack->addWidget(new SettingsPage(this));

    mainLayout->addWidget(m_stack, 1);

    // Dock
    QWidget *dock = new QWidget(this);
    dock->setFixedHeight(80);
    dock->setStyleSheet("background: rgba(10,10,15,0.95); border-top: 1px solid #2a2a3a;");
    QHBoxLayout *dockLayout = new QHBoxLayout(dock);
    dockLayout->setSpacing(8);
    dockLayout->setAlignment(Qt::AlignCenter);

    for (int i = 0; i < 7; ++i) {
        m_dockButtons[i] = new QPushButton(this);
        m_dockButtons[i]->setFixedSize(72, 64);
        m_dockButtons[i]->setProperty("index", i);
        m_dockButtons[i]->setStyleSheet(Styles::dockButtonStyle(i == 0));
        QVBoxLayout *bl = new QVBoxLayout(m_dockButtons[i]);
        bl->setAlignment(Qt::AlignCenter);
        bl->setSpacing(2);
        QLabel *ico = new QLabel(m_dockIcons[i], this);
        ico->setAlignment(Qt::AlignCenter);
        ico->setStyleSheet("font-size: 20px; border: none; background: transparent;");
        QLabel *lab = new QLabel(m_dockLabels[i], this);
        lab->setAlignment(Qt::AlignCenter);
        lab->setStyleSheet("font-size: 10px; border: none; background: transparent;");
        bl->addWidget(ico);
        bl->addWidget(lab);
        connect(m_dockButtons[i], &QPushButton::clicked, this, &CenterConsole::onDockClicked);
        dockLayout->addWidget(m_dockButtons[i]);
    }
    mainLayout->addWidget(dock);
}

void CenterConsole::switchPage(const QString &page) {
    int idx = m_pageNames.indexOf(page);
    if (idx >= 0 && m_stack) {
        m_stack->setCurrentIndex(idx);
        for (int i = 0; i < 7; ++i) {
            m_dockButtons[i]->setStyleSheet(Styles::dockButtonStyle(i == idx));
        }
    }
}

void CenterConsole::onDockClicked() {
    QPushButton *btn = qobject_cast<QPushButton*>(sender());
    if (!btn) return;
    int idx = btn->property("index").toInt();
    m_stack->setCurrentIndex(idx);
    for (int i = 0; i < 7; ++i) {
        m_dockButtons[i]->setStyleSheet(Styles::dockButtonStyle(i == idx));
    }
}
