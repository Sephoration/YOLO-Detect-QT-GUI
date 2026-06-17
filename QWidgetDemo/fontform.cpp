#include "fontform.h"
#include "ui_fontform.h"


fontform::fontform(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::fontform)
{
    ui->setupUi(this);

connect(ui->chkBold, &QCheckBox::clicked, this, &FontForm::onChkFontClicked);
connect(ui->chkItalic, &QCheckBox::clicked, this, &FontForm::onChkFontClicked);
connect(ui->chkUnderline, &QCheckBox::clicked, this, &FontForm::onChkFontClicked);

}

void FontForm::onChkFontClicked() {
    emit fontChanged(ui->chkBold->isChecked(), ui->chkItalic->isChecked(), ui->chkUnderline->isChecked());
}


fontform::~fontform()
{
    delete ui;
}
