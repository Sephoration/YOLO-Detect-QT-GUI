#include "gearpanel.h"
#include "common/constants.h"
#include <QHBoxLayout>
#include <QLabel>
#include <QVBoxLayout>

GearPanel::GearPanel(QWidget *parent) : QWidget(parent) {
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(12);
    mainLayout->setAlignment(Qt::AlignCenter);

    QHBoxLayout *gearLayout = new QHBoxLayout();
    gearLayout->setSpacing(8);

    for (int i = 0; i < 4; ++i) {
        m_labels[i] = new QLabel(m_gears[i], this);
        m_labels[i]->setFixedSize(40, 40);
        m_labels[i]->setAlignment(Qt::AlignCenter);
        m_labels[i]->setStyleSheet(QString(
            "background: rgba(255,255,255,0.05);"
            "border: 1px solid #2a2a3a;"
            "border-radius: 10px;"
            "color: #8b8b9e;"
            "font-size: 15px;"
            "font-weight: 600;"
        ));
        gearLayout->addWidget(m_labels[i]);
    }

    mainLayout->addLayout(gearLayout);

    m_modeLabel = new QLabel("舒适模式", this);
    m_modeLabel->setAlignment(Qt::AlignCenter);
    m_modeLabel->setStyleSheet(
        "background: rgba(255,255,255,0.05);"
        "border: 1px solid #2a2a3a;"
        "border-radius: 20px;"
        "color: #8b8b9e;"
        "font-size: 12px;"
        "padding: 6px 16px;"
    );
    mainLayout->addWidget(m_modeLabel, 0, Qt::AlignCenter);
    setLayout(mainLayout);
}

void GearPanel::setGear(const QString &gear) {
    int idx = m_gears.indexOf(gear.toUpper());
    for (int i = 0; i < 4; ++i) {
        if (i == idx) {
            m_labels[i]->setStyleSheet(
                "background: #00d4ff;"
                "border: 1px solid #00d4ff;"
                "border-radius: 10px;"
                "color: #000000;"
                "font-size: 15px;"
                "font-weight: 600;"
            );
        } else {
            m_labels[i]->setStyleSheet(
                "background: rgba(255,255,255,0.05);"
                "border: 1px solid #2a2a3a;"
                "border-radius: 10px;"
                "color: #8b8b9e;"
                "font-size: 15px;"
                "font-weight: 600;"
            );
        }
    }
}

void GearPanel::setMode(const QString &mode) {
    m_modeLabel->setText(mode);
    if (mode.contains("运动")) {
        m_modeLabel->setStyleSheet(
            "background: rgba(255, 51, 102, 0.1);"
            "border: 1px solid #ff3366;"
            "border-radius: 20px;"
            "color: #ff3366;"
            "font-size: 12px;"
            "padding: 6px 16px;"
        );
    } else if (mode.contains("经济")) {
        m_modeLabel->setStyleSheet(
            "background: rgba(0, 230, 118, 0.1);"
            "border: 1px solid #00e676;"
            "border-radius: 20px;"
            "color: #00e676;"
            "font-size: 12px;"
            "padding: 6px 16px;"
        );
    } else {
        m_modeLabel->setStyleSheet(
            "background: rgba(255,255,255,0.05);"
            "border: 1px solid #2a2a3a;"
            "border-radius: 20px;"
            "color: #8b8b9e;"
            "font-size: 12px;"
            "padding: 6px 16px;"
        );
    }
}
