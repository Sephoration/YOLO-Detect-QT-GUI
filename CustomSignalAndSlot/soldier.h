#ifndef SOLDIER_H
#define SOLDIER_H

#include <QObject>

class Soldier : public QObject
{
    Q_OBJECT
public:
    explicit Soldier(QObject *parent = nullptr);

signals:

// 定义槽函数
public slots:
    void fight() ;
    void fight(QString s) ;

};

#endif // SOLDIER_H
