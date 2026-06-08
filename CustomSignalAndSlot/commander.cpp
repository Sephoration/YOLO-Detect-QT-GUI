#include "commander.h"

//构造函数
Commander::Commander(QObject *parent)

    //成员初始化列表,调用父类 QObject 的构造函数，把 parent 传给它
    : QObject{parent}
{}
