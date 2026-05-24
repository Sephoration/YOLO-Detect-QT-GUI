#ifndef CENTERCONSOLE_H
#define CENTERCONSOLE_H

#include <QWidget>

class QStackedWidget;
class QPushButton;

class CenterConsole : public QWidget {
    Q_OBJECT
public:
    explicit CenterConsole(QWidget *parent = nullptr);
    void switchPage(const QString &page);

private slots:
    void onDockClicked();

private:
    void setupUI();
    void createDock();

    QStackedWidget *m_stack = nullptr;
    QPushButton *m_dockButtons[7] = {};
    QStringList m_pageNames = {"home", "nav", "music", "video", "ac", "vehicle", "settings"};
    QStringList m_dockLabels = {"主页", "导航", "音乐", "视频", "空调", "车辆", "设置"};
    QStringList m_dockIcons = {"🏠", "🧭", "🎵", "🎬", "❄", "🚗", "⚙"};
};

#endif // CENTERCONSOLE_H
