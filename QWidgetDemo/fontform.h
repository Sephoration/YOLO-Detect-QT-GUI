#ifndef FONTFORM_H
#define FONTFORM_H

#include <QWidget>

namespace Ui {
class fontform;
}

class fontform : public QWidget
{
    Q_OBJECT

public:
    explicit fontform(QWidget *parent = nullptr);
    ~fontform();

signals:
    void fontChanged(bool bold, bool italic, bool underline);

private:
    void onChkFontClicked();
    Ui::fontform *ui;
};

#endif // FONTFORM_H
