#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>

#include "commander.h"
#include "soldier.h"


QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = nullptr);
    ~Widget() override;

private:
    Ui::Widget *ui;


    Commander *commander;
    Soldier *soldier;

};
#endif // WIDGET_H
