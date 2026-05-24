#include "vehiclepage.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QPainter>

class CarDiagram : public QWidget {
public:
    explicit CarDiagram(QWidget *parent = nullptr) : QWidget(parent) {
        setMinimumSize(260, 420);
    }
protected:
    void paintEvent(QPaintEvent *) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        int cx = width() / 2, cy = height() / 2;

        // 车身轮廓
        p.setPen(QPen(Colors::border, 2));
        p.setBrush(QColor(255, 255, 255, 5));
        QRect body(cx - 100, cy - 180, 200, 360);
        p.drawRoundedRect(body, 40, 40);

        // 车门
        auto drawDoor = [&](int x, int y, bool left) {
            QRect dr(left ? x : x - 50, y, 52, 100);
            p.setPen(QPen(Colors::accentSuccess, 2));
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(dr, 6, 6);
        };
        drawDoor(cx - 100, cy - 120, true);
        drawDoor(cx + 100, cy - 120, false);
        drawDoor(cx - 100, cy + 20, true);
        drawDoor(cx + 100, cy + 20, false);

        // 轮胎
        auto drawTire = [&](int x, int y) {
            p.setPen(QPen(Colors::textSecondary, 2));
            p.setBrush(QColor(26, 26, 26));
            p.drawRoundedRect(x - 14, y - 22, 28, 44, 6, 6);
        };
        drawTire(cx - 60, cy - 140);
        drawTire(cx + 60, cy - 140);
        drawTire(cx - 60, cy + 140);
        drawTire(cx + 60, cy + 140);

        // 胎压
        p.setPen(Colors::accentSuccess);
        p.setFont(QFont("Segoe UI", 10, QFont::Medium));
        p.drawText(cx - 90, cy - 155, "2.5");
        p.drawText(cx + 70, cy - 155, "2.5");
        p.drawText(cx - 90, cy + 165, "2.4");
        p.drawText(cx + 70, cy + 165, "2.4");
    }
};

VehiclePage::VehiclePage(QWidget *parent) : QWidget(parent) {
    QHBoxLayout *mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(16, 16, 16, 16);
    mainLayout->setSpacing(16);

    // 左侧车辆图
    QWidget *left = new QWidget(this);
    left->setStyleSheet(Styles::widgetCardStyle());
    QVBoxLayout *ll = new QVBoxLayout(left);
    ll->setContentsMargins(16, 12, 16, 16);
    QLabel *lt = new QLabel("车辆状态", this);
    lt->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 1px; border: none; background: transparent;");
    ll->addWidget(lt);
    CarDiagram *diagram = new CarDiagram(this);
    ll->addWidget(diagram, 1, Qt::AlignCenter);
    mainLayout->addWidget(left, 1);

    // 右侧设置
    QWidget *right = new QWidget(this);
    QVBoxLayout *rl = new QVBoxLayout(right);
    rl->setContentsMargins(0, 0, 0, 0);
    rl->setSpacing(12);
    QLabel *rt = new QLabel("控制设置", this);
    rt->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 1px; border: none; background: transparent;");
    rl->addWidget(rt);

    struct Setting { QString title, desc; bool on; };
    Setting settings[] = {
        {"自动驻车", "停车时自动施加制动", true},
        {"车道保持辅助", "偏离车道时自动修正方向", true},
        {"自动大灯", "根据光线自动开启/关闭大灯", false},
        {"哨兵模式", "停车后监测周围环境并录像", false}
    };
    for (int i = 0; i < 4; ++i) {
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
        rl->addWidget(item);
    }
    rl->addStretch();
    mainLayout->addWidget(right, 1);
}
