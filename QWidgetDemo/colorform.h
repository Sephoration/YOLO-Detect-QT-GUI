#ifndef COLORFORM_H
#define COLORFORM_H

#include <QWidget>

namespace Ui {
class colorform;
}

class colorform : public QWidget
{
    Q_OBJECT

public:
    explicit colorform(QWidget *parent = nullptr);
    ~colorform();

private:
    Ui::colorform *ui;
};

#endif // COLORFORM_H
