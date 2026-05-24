#include "mainwindow.h"
#include "dashboard/dashboardwidget.h"
#include "center/centerconsole.h"
#include "common/constants.h"
#include "common/styles.h"
#include <QHBoxLayout>
#include <QWidget>
#include <QScreen>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setWindowTitle("车载智能座舱系统");
    setStyleSheet(Styles::globalStyle());

    QWidget *central = new QWidget(this);
    QHBoxLayout *layout = new QHBoxLayout(central);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // 仪表盘
    DashboardWidget *dashboard = new DashboardWidget(this);
    layout->addWidget(dashboard);

    // 中控
    CenterConsole *console = new CenterConsole(this);
    layout->addWidget(console, 1);

    setCentralWidget(central);

    // 设置窗口大小
    resize(Dimens::cockpitWidth, Dimens::cockpitHeight);

    // 居中显示
    QScreen *screen = QApplication::primaryScreen();
    if (screen) {
        QRect geo = screen->geometry();
        move((geo.width() - width()) / 2, (geo.height() - height()) / 2);
    }
}
