#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class QTimer;

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget() override;

protected:
    // 响应窗口拖拽大小变化
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void updateLayoutInfo();
    void onStopTimerTimeout();

private:
    Ui::Widget *ui;
    QTimer *m_updateTimer;   // 拖拽期间定时刷新 (100ms)
    QTimer *m_stopTimer;     // 检测拖拽结束 (500ms)
    int m_index = 0;
};

#endif // WIDGET_H
