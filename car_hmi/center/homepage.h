#ifndef HOMEPAGE_H
#define HOMEPAGE_H

#include <QWidget>

class HomePage : public QWidget {
    Q_OBJECT
public:
    explicit HomePage(QWidget *parent = nullptr);

signals:
    void requestSwitchPage(const QString &page);
};

#endif // HOMEPAGE_H
