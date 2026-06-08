#include "soldier.h"

#include <QDebug>

Soldier::Soldier(QObject *parent)
    : QObject{parent}
{}

void Soldier::fight()
{
    qDebug() << "fight";
}

void Soldier::fight(QString s)
{
    qDebug().noquote() << "fight for" << s;
}

void Soldier::escape()
{
    qDebug() << "i'm afraid of death, escape...";
}
