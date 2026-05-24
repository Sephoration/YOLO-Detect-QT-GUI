#include "dashboardwidget.h"
#include "gauge.h"
#include "gearpanel.h"
#include "common/constants.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QDateTime>
#include <QRandomGenerator>
#include <QtMath>

DashboardWidget::DashboardWidget(QWidget *parent) : QWidget(parent) {
    setFixedWidth(Dimens::dashboardWidth);
    setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0c0c14, stop:1 #080810);"
                  "border-right: 1px solid #2a2a3a;");
    setupUI();

    m_clockTimer = new QTimer(this);
    connect(m_clockTimer, &QTimer::timeout, this, &DashboardWidget::updateClock);
    m_clockTimer->start(1000);
    updateClock();

    m_dataTimer = new QTimer(this);
    connect(m_dataTimer, &QTimer::timeout, this, &DashboardWidget::simulateData);
    m_dataTimer->start(2000);
    simulateData();
}

void DashboardWidget::setupUI() {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    mainLayout->setSpacing(0);

    // 顶部状态栏
    QWidget *topBar = new QWidget(this);
    topBar->setFixedHeight(44);
    QHBoxLayout *topLayout = new QHBoxLayout(topBar);
    topLayout->setContentsMargins(24, 0, 24, 0);

    QLabel *tempLabel = new QLabel("室外 24°C | 晴", this);
    tempLabel->setStyleSheet("color: #8b8b9e; font-size: 13px; border: none; background: transparent;");
    topLayout->addStretch();
    topLayout->addWidget(tempLabel);
    mainLayout->addWidget(topBar);

    // 主仪表区
    QWidget *mainArea = new QWidget(this);
    QHBoxLayout *mainAreaLayout = new QHBoxLayout(mainArea);
    mainAreaLayout->setContentsMargins(16, 0, 16, 0);
    mainAreaLayout->setSpacing(8);

    // 左仪表
    QWidget *leftGauge = new QWidget(this);
    QVBoxLayout *leftLayout = new QVBoxLayout(leftGauge);
    leftLayout->setAlignment(Qt::AlignCenter);
    m_powerGauge = new Gauge(this);
    m_powerGauge->setMaxValue(100);
    m_powerGauge->setValue(45);
    m_powerGauge->setUnit("kW");
    m_powerGauge->setColor(Colors::accent);
    m_powerGauge->setMinimumSize(200, 200);
    leftLayout->addWidget(m_powerGauge);
    QLabel *powerLabel = new QLabel("功率 (kW)", this);
    powerLabel->setAlignment(Qt::AlignCenter);
    powerLabel->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 2px; border: none; background: transparent;");
    leftLayout->addWidget(powerLabel);
    mainAreaLayout->addWidget(leftGauge);

    // 中央信息
    QWidget *centerInfo = new QWidget(this);
    QVBoxLayout *centerLayout = new QVBoxLayout(centerInfo);
    centerLayout->setAlignment(Qt::AlignCenter);
    centerLayout->setSpacing(12);

    m_speedLabel = new QLabel("78", this);
    m_speedLabel->setAlignment(Qt::AlignCenter);
    m_speedLabel->setStyleSheet(
        "font-size: 88px; font-weight: 200; color: #ffffff;"
        "border: none; background: transparent;"
    );
    centerLayout->addWidget(m_speedLabel);

    QLabel *unitLabel = new QLabel("km/h", this);
    unitLabel->setAlignment(Qt::AlignCenter);
    unitLabel->setStyleSheet("color: #8b8b9e; font-size: 16px; border: none; background: transparent;");
    centerLayout->addWidget(unitLabel);

    m_gearPanel = new GearPanel(this);
    m_gearPanel->setGear("D");
    m_gearPanel->setMode("运动模式");
    centerLayout->addWidget(m_gearPanel);

    mainAreaLayout->addWidget(centerInfo);

    // 右仪表
    QWidget *rightGauge = new QWidget(this);
    QVBoxLayout *rightLayout = new QVBoxLayout(rightGauge);
    rightLayout->setAlignment(Qt::AlignCenter);
    m_speedGauge = new Gauge(this);
    m_speedGauge->setMaxValue(8);
    m_speedGauge->setValue(3.2);
    m_speedGauge->setUnit("x1000");
    m_speedGauge->setColor(Colors::accentSuccess);
    m_speedGauge->setMinimumSize(200, 200);
    rightLayout->addWidget(m_speedGauge);
    QLabel *rpmLabel = new QLabel("转速 (x1000)", this);
    rpmLabel->setAlignment(Qt::AlignCenter);
    rpmLabel->setStyleSheet("color: #8b8b9e; font-size: 12px; letter-spacing: 2px; border: none; background: transparent;");
    rightLayout->addWidget(rpmLabel);
    mainAreaLayout->addWidget(rightGauge);

    mainLayout->addWidget(mainArea, 1);

    // 底部信息栏
    QWidget *bottomBar = new QWidget(this);
    bottomBar->setFixedHeight(52);
    bottomBar->setStyleSheet("border-top: 1px solid #2a2a3a; background: transparent;");
    QHBoxLayout *bottomLayout = new QHBoxLayout(bottomBar);
    bottomLayout->setContentsMargins(24, 0, 24, 0);

    m_odoLabel = new QLabel("总里程: <span style='color:#ffffff;font-weight:500;'>12,580</span> km", this);
    m_odoLabel->setStyleSheet("color: #8b8b9e; font-size: 13px; border: none; background: transparent;");
    bottomLayout->addWidget(m_odoLabel);

    bottomLayout->addStretch();

    QWidget *adasWidget = new QWidget(this);
    QHBoxLayout *adasLayout = new QHBoxLayout(adasWidget);
    adasLayout->setSpacing(8);
    adasLayout->setContentsMargins(0, 0, 0, 0);
    QLabel *dot = new QLabel(this);
    dot->setFixedSize(8, 8);
    dot->setStyleSheet("background: #00e676; border-radius: 4px;");
    adasLayout->addWidget(dot);
    m_adasLabel = new QLabel("辅助驾驶已开启", this);
    m_adasLabel->setStyleSheet("color: #8b8b9e; font-size: 13px; border: none; background: transparent;");
    adasLayout->addWidget(m_adasLabel);
    bottomLayout->addWidget(adasWidget);

    mainLayout->addWidget(bottomBar);
}

void DashboardWidget::updateClock() {
    // 仪表盘不需要显示时间，留给中控
}

void DashboardWidget::simulateData() {
    m_currentSpeed += (QRandomGenerator::global()->bounded(100) / 25.0 - 2.0);
    m_currentSpeed = qBound(0.0, m_currentSpeed, 180.0);
    m_speedLabel->setText(QString::number(qRound(m_currentSpeed)));

    double power = m_currentSpeed * 0.6;
    m_powerGauge->setValue(power);

    double rpm = 0.8 + (m_currentSpeed / 180.0) * 5.5;
    m_speedGauge->setValue(rpm);
}
