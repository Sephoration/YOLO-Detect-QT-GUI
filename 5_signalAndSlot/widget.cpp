#include "widget.h"
#include "ui_widget.h"
#include <QDebug>
#include <QDateTime>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // 1.使用signal/slot的方式连接信号和槽
    connect(ui->pushButton_1,SIGNAL(clicked()),this,SLOT(showMaximized()));

    // 2.使用函数地址的方式
    connect(ui->pushButton_4,&QPushButton::clicked,this, &QWidget::showNormal);

    // 3.关闭窗口，使用设计师界面-信号槽编辑器
    // connect(ui->pushButton_3,&QPushButton::clicked,this, &QWidget::close);

#if 0
    //演示lambda表达式
    [](){
        qDebug() <<"lambda...";
    }
#endif

#if 0
    [](){
    qDebug() <<"lambda...";
    }();

#endif

#if 0
    [](){
        int a = 10;
        qDebug() << a ;
    }();
#endif


    //5. 使用lambda表达式做槽函数
    connect(ui->pushButton_5, &QPushButton::clicked, this, [this] (){
        QString title = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss");
        this->setWindowTitle(title);
    });

}

Widget::~Widget()
{
    delete ui;
}

void Widget::on_pushButton_2_clicked()
{
    this->showMinimized();
}

