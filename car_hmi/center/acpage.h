#ifndef ACPAGE_H
#define ACPAGE_H

#include <QWidget>

class QLabel;

class ACPage : public QWidget {
    Q_OBJECT
public:
    explicit ACPage(QWidget *parent = nullptr);

private slots:
    void setMode(int mode);
    void adjustTemp(int delta);

private:
    QLabel *m_tempLabels[2] = {};
    int m_temps[2] = {22, 22};
    int m_currentMode = 0;
};

#endif // ACPAGE_H
