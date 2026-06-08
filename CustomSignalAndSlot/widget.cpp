#include "widget.h"
#include "ui_widget.h"

#include <QDebug>
#include <QPushButton>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // 1、信号槽重载时：SIGNAL/SLOT（Qt4）
    // Qt4的写法中，同时包含函数名和函数参数，因此写法比较简单
#if 0
    commander = new Commander(this);
    soldier = new Soldier(this);

    connect(commander, SIGNAL(go()), soldier, SLOT(fight()));
    connect(commander, SIGNAL(go(QString)), soldier, SLOT(fight(QString)));

    emit commander->go();
    emit commander->go("freedom");
#endif

    // 2、信号槽重载时：函数指针（Qt5）
    // Qt5的写法中，只指明了函数名，没有函数参数，因此需要自定义函数指针
#if 0
    commander = new Commander(this);
    soldier = new Soldier(this);

    // 没有同名的信号和槽时，可以直接这样写。因为不存在二义性
    // connect(commander, &Commander::go, soldier, &Soldier::fight);

    // 有同名的信号和槽时，需要向下面这样定义函数指针。因为存在二义性
    void (Commander::*pGo)() = &Commander::go;
    void (Soldier::*pFight)() = &Soldier::fight;
    connect(commander, pGo, soldier, pFight);

    void (Commander::*pGoForFreedom)(QString) = &Commander::go;
    void (Soldier::*pFightForFreedom)(QString) = &Soldier::fight;
    connect(commander, pGoForFreedom, soldier, pFightForFreedom);

    emit commander->go();
    emit commander->go("freedom");
#endif

    // 3、信号槽重载时：精简写法（QOverload / qOverload）
    // 使用类模板QOverload或函数模板qOverload，可以简化写法
#if 0
    commander = new Commander(this);
    soldier = new Soldier(this);

    // 使用类模板：QOverload<>
    // connect(commander, QOverload<>::of(&Commander::go), soldier, QOverload<>::of(&Soldier::fight));
    // connect(commander, QOverload<QString>::of(&Commander::go), soldier, QOverload<QString>::of(&Soldier::fight));

    // 或者使用函数模板：qOverload（推荐）
    connect(commander, qOverload<>(&Commander::go), soldier, qOverload<>(&Soldier::fight));
    connect(commander, qOverload<QString>(&Commander::go), soldier, qOverload<QString>(&Soldier::fight));

    emit commander->go();
    emit commander->go("freedom");
#endif

    // 4、一个信号连接多个槽
    // 当commander发出go信号时，soldier执行fight，soldier2执行escape
#if 0
    commander = new Commander(this);
    soldier = new Soldier(this);
    soldier2 = new Soldier(this);

    // 士兵1很勇敢，收到冲锋的信号后，开始战斗
    connect(commander, qOverload<>(&Commander::go), soldier, qOverload<>(&Soldier::fight));

    // 士兵2很怕死，收到冲锋的信号后，开始逃跑
    connect(commander, qOverload<>(&Commander::go), soldier2, qOverload<>(&Soldier::escape));

    emit commander->go();
#endif

    // 5、多个信号连接一个槽
    // 当commander发射go信号和move信号时，都会执行士兵的fight槽函数
#if 0
    commander = new Commander(this);
    soldier = new Soldier(this);

    connect(commander, qOverload<>(&Commander::go), soldier, qOverload<>(&Soldier::fight));
    connect(commander, qOverload<>(&Commander::move), soldier, qOverload<>(&Soldier::fight));

    emit commander->go();
    emit commander->move();
#endif

    // 6、信号连接信号
    // 按钮的点击会发射clicked信号 => commander发射move信号 => soldier执行escape槽函数
#if 0
    commander = new Commander();
    soldier = new Soldier();

    connect(ui->btnSignal2Signal, &QPushButton::clicked, commander, &Commander::move);
    connect(commander, &Commander::move, soldier, &Soldier::escape);
#endif

    // 7、断开连接 - disconnect
    // 当一个对象delete之后，Qt自动取消所有连接到这个对象上面的槽
#if 0
    commander = new Commander();
    soldier = new Soldier();

    connect(commander, qOverload<>(&Commander::go), soldier, qOverload<>(&Soldier::fight));
    connect(commander, qOverload<QString>(&Commander::go), soldier, qOverload<QString>(&Soldier::fight));

    emit commander->go();

    // 断开所有连接到commander信号上的槽函数
    commander->disconnect();
    emit commander->go("freedom");
#endif

    // 8、获取发送信号的对象 - sender()
    // 在槽函数内部，可以直接调用sender()函数，获取发送信号的对象
#if 1
    connect(ui->btnStart, &QPushButton::clicked, this, &Widget::onBtnsClicked);
    connect(ui->btnStop, &QPushButton::clicked, this, &Widget::onBtnsClicked);
#endif
}

void Widget::onBtnsClicked()
{
    // 获取发送信号的对象的指针
    QObject *senderObj = sender();

    // 尝试将其转换为QPushButton类型
    QPushButton *button = qobject_cast<QPushButton *>(senderObj);

    // 如果转换成功，则说明是一个按钮发送了信号
    if (button)
    {
        if (button == ui->btnStart)
        {
            qDebug() << "点击了启动按钮";
        }
        else if (button == ui->btnStop)
        {
            qDebug() << "点击了停止按钮";
        }
    }
}

Widget::~Widget()
{
    delete ui;
}
