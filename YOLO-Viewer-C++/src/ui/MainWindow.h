#pragma once

#include <QMainWindow>
#include <QSplitter>
#include <QHBoxLayout>
#include <QFileInfo>
#include <QDragEnterEvent>
#include <QDropEvent>
#include "ui/LeftDisplayPanel.h"
#include "ui/RightControlPanel.h"
#include "ui/InspectionPanel.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);

    LeftDisplayPanel*  leftPanel()      const { return m_leftPanel; }
    RightControlPanel* rightPanel()     const { return m_rightPanel; }
    InspectionPanel*   inspectionPanel() const { return m_inspectionPanel; }

    // ── 对话框 ──
    static bool confirm(QWidget *parent, const QString &title, const QString &msg);
    static void info(QWidget *parent, const QString &title, const QString &msg);
    static void warn(QWidget *parent, const QString &title, const QString &msg);
    static void error(QWidget *parent, const QString &title, const QString &msg);

signals:
    void modelLoadRequested();
    void imageOpenRequested();
    void videoOpenRequested();
    void cameraOpenRequested();
    void aboutRequested();
    void helpRequested();

    // 传递子面板的信号
    void playPauseClicked();
    void detectToggled(bool on);
    void screenshotClicked();
    void seekRequested(double ratio);

protected:
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;

private:
    void initUi();
    void setupMenuBar();
    void setupStyleSheet();

    LeftDisplayPanel  *m_leftPanel       = nullptr;
    RightControlPanel *m_rightPanel      = nullptr;
    InspectionPanel   *m_inspectionPanel = nullptr;
    QSplitter         *m_splitter        = nullptr;
};
