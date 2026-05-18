/********************************************************************************
** Form generated from reading UI file 'mywidget.ui'
**
** Created by: Qt User Interface Compiler version 6.11.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MYWIDGET_H
#define UI_MYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MyWidget
{
public:
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *btn_max;
    QPushButton *btn_min;
    QPushButton *btn_normal;
    QPushButton *btn_close;
    QSpacerItem *horizontalSpacer_2;

    void setupUi(QWidget *MyWidget)
    {
        if (MyWidget->objectName().isEmpty())
            MyWidget->setObjectName("MyWidget");
        MyWidget->resize(511, 353);
        horizontalLayout = new QHBoxLayout(MyWidget);
        horizontalLayout->setObjectName("horizontalLayout");
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        btn_max = new QPushButton(MyWidget);
        btn_max->setObjectName("btn_max");
        QFont font;
        font.setPointSize(14);
        btn_max->setFont(font);

        horizontalLayout->addWidget(btn_max);

        btn_min = new QPushButton(MyWidget);
        btn_min->setObjectName("btn_min");
        btn_min->setFont(font);

        horizontalLayout->addWidget(btn_min);

        btn_normal = new QPushButton(MyWidget);
        btn_normal->setObjectName("btn_normal");
        btn_normal->setFont(font);

        horizontalLayout->addWidget(btn_normal);

        btn_close = new QPushButton(MyWidget);
        btn_close->setObjectName("btn_close");
        btn_close->setFont(font);

        horizontalLayout->addWidget(btn_close);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        retranslateUi(MyWidget);

        QMetaObject::connectSlotsByName(MyWidget);
    } // setupUi

    void retranslateUi(QWidget *MyWidget)
    {
        MyWidget->setWindowTitle(QCoreApplication::translate("MyWidget", "\346\240\207\345\207\206\344\277\241\345\217\267\346\247\275", nullptr));
        btn_max->setText(QCoreApplication::translate("MyWidget", "\346\234\200\345\244\247\345\214\226\347\216\260\345\256\236", nullptr));
        btn_min->setText(QCoreApplication::translate("MyWidget", "\346\234\200\345\260\217\345\214\226\346\230\276\347\244\272", nullptr));
        btn_normal->setText(QCoreApplication::translate("MyWidget", "\346\255\243\345\270\270\346\230\276\347\244\272", nullptr));
        btn_close->setText(QCoreApplication::translate("MyWidget", "\345\205\263\351\227\255\347\252\227\345\217\243", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MyWidget: public Ui_MyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MYWIDGET_H
