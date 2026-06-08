#ifndef SOLDIER_H
#define SOLDIER_H

#include <QObject>

class Soldier : public QObject
{
    Q_OBJECT
public:
    explicit Soldier(QObject *parent = nullptr);

public slots:
    void fight();
    void fight(QString s);
    void escape();
};

#endif // SOLDIER_H
