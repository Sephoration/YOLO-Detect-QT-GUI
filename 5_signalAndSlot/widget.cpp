#include "widget.h"
#include "ui_widget.h"

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    //1.使用signal/slot的方式连接信号和槽
    connect(ui->pushButton_1,SIGNAL(clicked()),this,SLOT(showMaximized()));

    //2.使用函数地址的方式
    connect(ui->pushButton_1,&QPushButton::clicked,this, &QWidget::showNormal);

}

Widget::~Widget()
{
    delete ui;
}
