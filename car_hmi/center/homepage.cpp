#include "homepage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPainter>

class MapPreview : public QWidget {
public:
    explicit MapPreview(QWidget *parent = nullptr) : QWidget(parent) {
        setMinimumHeight(280);
    }
protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), QColor(13, 27, 42));

        // 绘制道路网格
        p.setPen(QPen(QColor(27, 58, 92), 2));
        int step = 80;
        for (int x = 0; x < width(); x += step) p.drawLine(x, 0, x, height());
        for (int y = 0; y < height(); y += step) p.drawLine(0, y, width(), y);

        // 导航箭头
        int cx = width() / 2, cy = height() / 2;
        p.setBrush(Colors::accent);
        p.setPen(Qt::NoPen);
        p.drawEllipse(cx - 20, cy - 20, 40, 40);
        p.setPen(QPen(Qt::black, 2));
        p.drawLine(cx, cy - 10, cx, cy + 10);
        p.drawLine(cx, cy - 10, cx - 6, cy + 2);
        p.drawLine(cx, cy - 10, cx + 6, cy + 2);

        // 导航信息卡片
        QRect cardRect(16, height() - 80, width() - 32, 64);
        p.fillRect(cardRect, QColor(0, 0, 0, 180));
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0, 0, 0, 180));
        p.drawRoundedRect(cardRect, 12, 12);

        p.setPen(Colors::textPrimary);
        p.setFont(QFont("Microsoft YaHei", 13, QFont::Medium));
        p.drawText(cardRect.adjusted(16, 10, -16, 0), "前方 200米 右转");
        p.setPen(Colors::accent);
        p.setFont(QFont("Microsoft YaHei", 11));
        p.drawText(cardRect.adjusted(16, 34, -16, -10), "进入建国路  ▪  剩余 3.2km");
    }
};

HomePage::HomePage(QWidget *parent) : QWidget(parent) {
    QGridLayout *grid = new QGridLayout(this);
    grid->setContentsMargins(16, 16, 16, 16);
    grid->setSpacing(16);

    // 导航小窗
    QWidget *navWidget = new QWidget(this);
    navWidget->setStyleSheet(Styles::widgetCardStyle());
    QVBoxLayout *navLayout = new QVBoxLayout(navWidget);
    navLayout->setContentsMargins(16, 12, 16, 16);
    QLabel *navTitle = new QLabel("导航", this);
    navTitle->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 1px; border: none; background: transparent;");
    navLayout->addWidget(navTitle);
    MapPreview *map = new MapPreview(this);
    navLayout->addWidget(map, 1);
    grid->addWidget(navWidget, 0, 0, 2, 1);

    // 车辆状态
    QWidget *statusWidget = new QWidget(this);
    statusWidget->setStyleSheet(Styles::widgetCardStyle());
    QVBoxLayout *statusLayout = new QVBoxLayout(statusWidget);
    statusLayout->setContentsMargins(16, 12, 16, 16);
    QLabel *statusTitle = new QLabel("车辆状态", this);
    statusTitle->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 1px; border: none; background: transparent;");
    statusLayout->addWidget(statusTitle);

    QGridLayout *statusGrid = new QGridLayout();
    statusGrid->setSpacing(12);
    struct Item { QString icon, value, label; };
    Item items[] = {
        {"🔋", "78%", "电池电量"},
        {"⚡", "420km", "续航里程"},
        {"🌡", "68°C", "电池温度"},
        {"💨", "2.5bar", "平均胎压"}
    };
    for (int i = 0; i < 4; ++i) {
        QWidget *item = new QWidget(this);
        item->setStyleSheet("background: rgba(255,255,255,0.05); border-radius: 12px; border: none;");
        QHBoxLayout *il = new QHBoxLayout(item);
        il->setContentsMargins(12, 12, 12, 12);
        QLabel *ico = new QLabel(items[i].icon, this);
        ico->setFixedSize(40, 40);
        ico->setAlignment(Qt::AlignCenter);
        ico->setStyleSheet("background: rgba(0,212,255,0.1); border-radius: 10px; font-size: 16px; border: none;");
        il->addWidget(ico);
        QVBoxLayout *vl = new QVBoxLayout();
        QLabel *val = new QLabel(items[i].value, this);
        val->setStyleSheet("font-size: 16px; font-weight: 500; color: #ffffff; border: none; background: transparent;");
        QLabel *lab = new QLabel(items[i].label, this);
        lab->setStyleSheet("font-size: 11px; color: #8b8b9e; border: none; background: transparent;");
        vl->addWidget(val);
        vl->addWidget(lab);
        il->addLayout(vl);
        statusGrid->addWidget(item, i / 2, i % 2);
    }
    statusLayout->addLayout(statusGrid);
    grid->addWidget(statusWidget, 0, 1);

    // 快捷功能
    QWidget *quickWidget = new QWidget(this);
    quickWidget->setStyleSheet(Styles::widgetCardStyle());
    QVBoxLayout *quickLayout = new QVBoxLayout(quickWidget);
    quickLayout->setContentsMargins(16, 12, 16, 16);
    QLabel *quickTitle = new QLabel("快捷功能", this);
    quickTitle->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 1px; border: none; background: transparent;");
    quickLayout->addWidget(quickTitle);

    QHBoxLayout *actions = new QHBoxLayout();
    actions->setSpacing(12);
    struct Act { QString icon, label, page; };
    Act acts[] = {
        {"🎵", "音乐", "music"},
        {"🎬", "视频", "video"},
        {"❄", "空调", "ac"},
        {"🚗", "车辆", "vehicle"}
    };
    for (int i = 0; i < 4; ++i) {
        QPushButton *btn = new QPushButton(this);
        btn->setStyleSheet(Styles::actionButtonStyle());
        btn->setFixedHeight(90);
        QVBoxLayout *bl = new QVBoxLayout(btn);
        bl->setAlignment(Qt::AlignCenter);
        QLabel *ico = new QLabel(acts[i].icon, this);
        ico->setAlignment(Qt::AlignCenter);
        ico->setStyleSheet("font-size: 22px; border: none; background: transparent;");
        QLabel *lab = new QLabel(acts[i].label, this);
        lab->setAlignment(Qt::AlignCenter);
        lab->setStyleSheet("color: #8b8b9e; font-size: 12px; border: none; background: transparent;");
        bl->addWidget(ico);
        bl->addWidget(lab);
        connect(btn, &QPushButton::clicked, this, [this, p = acts[i].page]() { emit requestSwitchPage(p); });
        actions->addWidget(btn);
    }
    quickLayout->addLayout(actions);
    grid->addWidget(quickWidget, 1, 1);
}
