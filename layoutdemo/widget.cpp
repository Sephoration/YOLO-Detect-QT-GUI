#include "widget.h"
#include "ui_widget.h"

#include <QTimer>
#include <QResizeEvent>
#include <QTextCursor>

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);

    // 更新定时器：拖拽期间每 100ms 刷新一次
    m_updateTimer = new QTimer(this);
    m_updateTimer->setInterval(100);

    // 停止定时器：最后一次 resizeEvent 后 500ms 停止刷新
    m_stopTimer = new QTimer(this);
    m_stopTimer->setSingleShot(true);
    m_stopTimer->setInterval(500);

    connect(m_updateTimer, &QTimer::timeout, this, &Widget::updateLayoutInfo);
    connect(m_stopTimer, &QTimer::timeout, this, &Widget::onStopTimerTimeout);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);

    // 立即显示一次
    updateLayoutInfo();

    // 拖拽期间持续刷新
    if (!m_updateTimer->isActive()) {
        m_updateTimer->start();
    }

    // 每次 resize 都重置停止定时器 → 拖拽结束后 500ms 才停止刷新
    m_stopTimer->start();
}

void Widget::onStopTimerTimeout()
{
    m_updateTimer->stop();
}

void Widget::updateLayoutInfo()
{
    // 索引
    m_index++;
    QString strIndex = QString("[%1]-------------").arg(m_index);

    // 按钮的宽度
    QString width = QString("按钮宽度：\t%1,%2,%3,%4")
                        .arg(ui->Btn_add->width())
                        .arg(ui->Btn_del->width())
                        .arg(ui->Btn_modify->width())
                        .arg(ui->Btn_query->width());

    // 边距
    QMargins margins = ui->horizontaMidget->layout()->contentsMargins();
    QString strMargins = QString("边距（左上右下）：\t%1,%2,%3,%4")
                             .arg(margins.left())
                             .arg(margins.top())
                             .arg(margins.right())
                             .arg(margins.bottom());

    // 间距
    int spacing = ui->horizontaMidget->layout()->spacing();
    QString strSpacing = QString("间距：\t\t%1").arg(spacing);

    // 当 textEdit 行数超过 600 行时清空，并重置索引
    int lineCut = ui->textEdit->document()->lineCount();
    if (lineCut > 100 * 6) {
        ui->textEdit->clear();
        m_index = 0;
    }

    ui->textEdit->append(strIndex);
    ui->textEdit->append(width);
    ui->textEdit->append(strMargins);
    ui->textEdit->append(strSpacing);
    ui->textEdit->append("----------------");
    ui->textEdit->append("");

    // 移动光标到最后一行
    ui->textEdit->moveCursor(QTextCursor::End);
}
