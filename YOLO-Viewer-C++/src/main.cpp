#include <QApplication>
#include <QStyleFactory>
#include <QDebug>
#include "Config.h"
#include "ui/MainWindow.h"
#include "core/MainController.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    // ── 应用元信息 ──
    app.setApplicationName(AppConfig::APP_NAME);
    app.setApplicationVersion(AppConfig::APP_VERSION);
    app.setOrganizationName("Sephoration");

    // ── 主题 (Fusion 跨平台一致) ──
    app.setStyle(QStyleFactory::create("Fusion"));

    // ── 全局暗色 palette ──
    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window,          QColor(53,53,53));
    darkPalette.setColor(QPalette::WindowText,       Qt::white);
    darkPalette.setColor(QPalette::Base,             QColor(35,35,35));
    darkPalette.setColor(QPalette::AlternateBase,    QColor(53,53,53));
    darkPalette.setColor(QPalette::ToolTipBase,      QColor(25,25,25));
    darkPalette.setColor(QPalette::ToolTipText,      Qt::white);
    darkPalette.setColor(QPalette::Text,             Qt::white);
    darkPalette.setColor(QPalette::Button,           QColor(53,53,53));
    darkPalette.setColor(QPalette::ButtonText,       Qt::white);
    darkPalette.setColor(QPalette::BrightText,       Qt::red);
    darkPalette.setColor(QPalette::Link,             QColor(42,130,218));
    darkPalette.setColor(QPalette::Highlight,        QColor(42,130,218));
    darkPalette.setColor(QPalette::HighlightedText,  Qt::black);
    darkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(127,127,127));
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(127,127,127));
    app.setPalette(darkPalette);

    // ── 主窗口 ──
    MainWindow mainWindow;
    mainWindow.show();

    // ── 控制器（自动串联所有模块） ──
    MainController controller(&mainWindow);

    return app.exec();
}
