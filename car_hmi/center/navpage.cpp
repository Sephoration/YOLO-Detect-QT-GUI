#include "navpage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>

class MapWidget : public QWidget {
public:
    explicit MapWidget(QWidget *parent = nullptr) : QWidget(parent) {}
protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.fillRect(rect(), QColor(13, 27, 42));
        // 道路
        p.setPen(QPen(QColor(27, 58, 92), 3));
        int step = 100;
        for (int x = 0; x < width(); x += step) p.drawLine(x, 0, x, height());
        for (int y = 0; y < height(); y += step) p.drawLine(0, y, width(), y);
        // 路线高亮
        p.setPen(QPen(Colors::accent, 5, Qt::DashLine));
        p.drawLine(width()/2, height()-60, width()/2, 60);
    }
};

NavPage::NavPage(QWidget *parent) : QWidget(parent) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(16);

    QWidget *mapContainer = new QWidget(this);
    mapContainer->setStyleSheet("background: #0d1b2a; border-radius: 16px;");
    QVBoxLayout *mc = new QVBoxLayout(mapContainer);
    mc->setContentsMargins(0, 0, 0, 0);

    MapWidget *map = new MapWidget(this);
    mc->addWidget(map, 1);

    // 搜索栏（叠加在地图上）
    QLineEdit *search = new QLineEdit(this);
    search->setPlaceholderText("搜索目的地...");
    search->setStyleSheet(
        "background: rgba(0,0,0,0.7); border: 1px solid #2a2a3a; border-radius: 12px;"
        "padding: 10px 16px; color: #ffffff; font-size: 15px;"
    );
    // 使用绝对定位模拟叠加效果
    mc->addWidget(search);
    mainLayout->addWidget(mapContainer, 1);

    // 底部信息
    QWidget *info = new QWidget(this);
    info->setStyleSheet("background: rgba(0,0,0,0.8); border-radius: 16px; border: none;");
    QVBoxLayout *il = new QVBoxLayout(info);
    il->setContentsMargins(20, 16, 20, 16);
    QLabel *dest = new QLabel("前往: 三里屯太古里", this);
    dest->setStyleSheet("font-size: 18px; font-weight: 500; border: none; background: transparent;");
    il->addWidget(dest);
    QLabel *detail = new QLabel("预计到达 15:08  ▪  剩余 8.5km  ▪  红绿灯 3个", this);
    detail->setStyleSheet("color: #8b8b9e; font-size: 13px; border: none; background: transparent;");
    il->addWidget(detail);
    QHBoxLayout *opts = new QHBoxLayout();
    opts->setSpacing(12);
    QString optTexts[] = {"避开拥堵", "高速优先"};
    for (int i = 0; i < 2; ++i) {
        QLabel *opt = new QLabel(optTexts[i], this);
        opt->setStyleSheet(
            "background: rgba(255,255,255,0.05); border-radius: 10px; padding: 8px 14px;"
            "font-size: 12px; color: #ffffff;"
        );
        opts->addWidget(opt);
    }
    opts->addStretch();
    il->addLayout(opts);
    mainLayout->addWidget(info);
}
