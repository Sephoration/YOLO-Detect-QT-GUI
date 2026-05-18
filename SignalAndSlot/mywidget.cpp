#include "mywidget.h"
#include "ui_mywidget.h"

MyWidget::MyWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MyWidget)
{
    ui->setupUi(this);


    connect(ui -> btn_max , &QPushButton::clicked , this , &QWidget::showMaximized ) ;
    connect(ui -> btn_min , &QPushButton::clicked , this , &QWidget::showMinimized ) ;
    connect(ui -> btn_normal , &QPushButton::clicked , this , &QWidget::showNormal ) ;
    connect(ui -> btn_close , &QPushButton::clicked , this , &QWidget::close ) ;



}

MyWidget::~MyWidget()
{
    delete ui;
}
