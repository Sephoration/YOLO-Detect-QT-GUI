#ifndef COMMANDER_H
#define COMMANDER_H

#include <QObject>

class Commander : public QObject
{
    Q_OBJECT //宏
public:
    explicit Commander(QObject *parent = nullptr);

signals:
    // 信号只需要声明不需要实现
    // 信号的返回值为void
    void go () ;
    void go (QString s) ;
};

#endif // COMMANDER_H
