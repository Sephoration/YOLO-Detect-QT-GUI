#ifndef COMMANDER_H
#define COMMANDER_H

//头文件
#include <QObject>

// 你的代码 → Qt 的 MOC 工具 → 转换成标准 C++ → 正常编译器编译
// Q_OBJECT会先被转换
class Commander : public QObject
{
    Q_OBJECT
public:
    explicit Commander(QObject *parent = nullptr);

signals:
    void go();
    void go(QString s);
    void move();
};

#endif // COMMANDER_H
