#ifndef GEARPANEL_H
#define GEARPANEL_H

#include <QWidget>

class QLabel;

class GearPanel : public QWidget {
    Q_OBJECT
public:
    explicit GearPanel(QWidget *parent = nullptr);
    void setGear(const QString &gear);
    void setMode(const QString &mode);

private:
    QLabel *m_labels[4] = {};
    QLabel *m_modeLabel = nullptr;
    QStringList m_gears = {"P", "R", "N", "D"};
};

#endif // GEARPANEL_H
