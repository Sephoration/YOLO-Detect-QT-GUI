#include "colorform.h"
#include "ui_colorform.h"

colorform::colorform(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::colorform)
{
    ui->setupUi(this);
}

colorform::~colorform()
{
    delete ui;
}
