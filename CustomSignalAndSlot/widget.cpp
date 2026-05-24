#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);


    // 1.创建两个类的实例
    commander = new Commander(this);
    soldier   = new Soldier(this);

    // 2.建立信号和槽的连接
    connect (commander ,QOverload<>::of(&Commander::go) ,soldier ,QOverload<>::of(&Soldier::fight));
    connect (commander ,QOverload<QString>::of(&Commander::go) ,soldier ,QOverload<QString>::of(&Soldier::fight));

    // 3.发送信号
    commander -> go ();
    commander -> go ("freedom");

}

Widget::~Widget()
{
    delete ui;
}
