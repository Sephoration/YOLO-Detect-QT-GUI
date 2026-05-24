#ifndef GAUGE_H
#define GAUGE_H

#include <QWidget>

class Gauge : public QWidget {
    Q_OBJECT
    Q_PROPERTY(double value READ value WRITE setValue NOTIFY valueChanged)
    Q_PROPERTY(double maxValue READ maxValue WRITE setMaxValue)
    Q_PROPERTY(QString unit READ unit WRITE setUnit)
    Q_PROPERTY(QColor color READ color WRITE setColor)

public:
    explicit Gauge(QWidget *parent = nullptr);

    double value() const { return m_value; }
    double maxValue() const { return m_maxValue; }
    QString unit() const { return m_unit; }
    QColor color() const { return m_color; }

public slots:
    void setValue(double val);
    void setMaxValue(double max);
    void setUnit(const QString &unit);
    void setColor(const QColor &color);

signals:
    void valueChanged(double value);

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    double m_value = 0;
    double m_maxValue = 100;
    QString m_unit = "";
    QColor m_color = QColor(0, 212, 255);
    double m_animValue = 0;
};

#endif // GAUGE_H
