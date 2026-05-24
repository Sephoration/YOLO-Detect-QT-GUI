#ifndef DASHBOARDWIDGET_H
#define DASHBOARDWIDGET_H

#include <QWidget>

class Gauge;
class GearPanel;
class QLabel;
class QTimer;

class DashboardWidget : public QWidget {
    Q_OBJECT
public:
    explicit DashboardWidget(QWidget *parent = nullptr);

private slots:
    void updateClock();
    void simulateData();

private:
    void setupUI();

    Gauge *m_powerGauge = nullptr;
    Gauge *m_speedGauge = nullptr;
    GearPanel *m_gearPanel = nullptr;
    QLabel *m_speedLabel = nullptr;
    QLabel *m_odoLabel = nullptr;
    QLabel *m_rangeLabel = nullptr;
    QLabel *m_adasLabel = nullptr;
    QTimer *m_dataTimer = nullptr;
    QTimer *m_clockTimer = nullptr;
    double m_currentSpeed = 78;
};

#endif // DASHBOARDWIDGET_H
