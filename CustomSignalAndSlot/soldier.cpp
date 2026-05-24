#include "soldier.h"

// 打印
#include <QDebug>

Soldier::Soldier(QObject *parent)
    : QObject{parent}
{}


// 实现槽函数
void Soldier::fight()
{
    qDebug( ) << "fight" ;
}



void Soldier::fight(QString s)
{
    qDebug().noquote() << "fight for"  << s ;
}
