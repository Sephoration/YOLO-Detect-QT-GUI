#include "gauge.h"
#include "common/constants.h"
#include <QPainter>
#include <QtMath>

Gauge::Gauge(QWidget *parent) : QWidget(parent) {
    setMinimumSize(180, 180);
    m_animValue = m_value;
}

void Gauge::setValue(double val) {
    val = qBound(0.0, val, m_maxValue);
    if (!qFuzzyCompare(m_value, val)) {
        m_value = val;
        emit valueChanged(m_value);
        update();
    }
}

void Gauge::setMaxValue(double max) {
    m_maxValue = max;
    update();
}

void Gauge::setUnit(const QString &unit) {
    m_unit = unit;
    update();
}

void Gauge::setColor(const QColor &color) {
    m_color = color;
    update();
}

void Gauge::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    qreal w = width();
    qreal h = height();
    qreal cx = w / 2.0;
    qreal cy = h / 2.0 + 8;
    qreal r = qMin(w, h) / 2.0 - 16;

    qreal startAngle = 180 + 36; // 216 degrees
    qreal span = 360 - 72;       // 288 degrees
    qreal endAngle = startAngle + span;

    // 背景弧
    QPen bgPen(QColor(255, 255, 255, 20));
    bgPen.setWidth(10);
    bgPen.setCapStyle(Qt::RoundCap);
    p.setPen(bgPen);
    p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), startAngle * 16, span * 16);

    // 刻度
    p.setPen(Qt::NoPen);
    for (int i = 0; i <= 10; ++i) {
        qreal ratio = i / 10.0;
        qreal angle = (startAngle + span * ratio) * M_PI / 180.0;
        qreal x1 = cx + cos(angle) * (r - 14);
        qreal y1 = cy - sin(angle) * (r - 14);
        qreal x2 = cx + cos(angle) * (r - 22);
        qreal y2 = cy - sin(angle) * (r - 22);
        p.setPen(QPen(QColor(255, 255, 255, 60), 2));
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2));

        // 数字
        qreal tx = cx + cos(angle) * (r - 34);
        qreal ty = cy - sin(angle) * (r - 34);
        p.setPen(QColor(255, 255, 255, 100));
        p.setFont(QFont("Segoe UI", 8));
        QString num = QString::number(qRound(m_maxValue * ratio));
        QRect textRect(tx - 15, ty - 8, 30, 16);
        p.drawText(textRect, Qt::AlignCenter, num);
        p.setPen(Qt::NoPen);
    }

    // 进度弧
    double progress = qBound(0.0, m_value / m_maxValue, 1.0);
    qreal progressSpan = span * progress;
    QPen fgPen(m_color);
    fgPen.setWidth(10);
    fgPen.setCapStyle(Qt::RoundCap);
    p.setPen(fgPen);
    p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), startAngle * 16, progressSpan * 16);

    // 中心数值
    p.setPen(Colors::textPrimary);
    p.setFont(QFont("Segoe UI", 28, QFont::DemiBold));
    QRect valRect(cx - 50, cy - 28, 100, 36);
    p.drawText(valRect, Qt::AlignCenter, QString::number(qRound(m_value)));

    // 单位
    if (!m_unit.isEmpty()) {
        p.setPen(Colors::textSecondary);
        p.setFont(QFont("Segoe UI", 9));
        QRect unitRect(cx - 30, cy + 10, 60, 16);
        p.drawText(unitRect, Qt::AlignCenter, m_unit);
    }
}

void Gauge::resizeEvent(QResizeEvent *) {
    update();
}
