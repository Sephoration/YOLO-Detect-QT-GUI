#include "LyricView.h"
#include <QPainter>
#include <QPaintEvent>
#include <QFont>

LyricView::LyricView(QWidget *parent)
    : QWidget(parent) {
    setMinimumWidth(240);
    m_animation = new QPropertyAnimation(this, QByteArray(), this);
    m_animation->setDuration(300);
    m_animation->setEasingCurve(QEasingCurve::OutCubic);
}

void LyricView::loadFromFile(const QString &audioFilePath) {
    if (!m_parser.loadFromLrcFile(audioFilePath)) {
        m_parser.clear();
    }
    m_currentIndex = -1;
    m_offsetY = height() / 2;
    update();
}

void LyricView::setPosition(qint64 positionMs) {
    int idx = m_parser.currentIndex(positionMs);
    if (idx != m_currentIndex) {
        m_currentIndex = idx;
        updateOffset();
        update();
    }
}

void LyricView::clear() {
    m_parser.clear();
    m_currentIndex = -1;
    m_offsetY = height() / 2;
    update();
}

void LyricView::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event)
    QPainter p(this);
    p.setRenderHint(QPainter::TextAntialiasing);

    const auto &lines = m_parser.lines();
    int centerY = height() / 2;

    for (int i = 0; i < lines.size(); ++i) {
        int y = centerY - m_offsetY + i * LINE_HEIGHT;

        // Skip lines outside visible area
        if (y < -LINE_HEIGHT || y > height() + LINE_HEIGHT)
            continue;

        bool isCurrent = (i == m_currentIndex);
        bool isNear = qAbs(i - m_currentIndex) <= 2;

        if (isCurrent) {
            p.setPen(QColor(0, 212, 255));
            QFont f = p.font();
            f.setPointSize(12);
            f.setBold(true);
            p.setFont(f);
        } else if (isNear) {
            p.setPen(QColor(200, 200, 210));
            QFont f = p.font();
            f.setPointSize(11);
            p.setFont(f);
        } else {
            p.setPen(QColor(120, 120, 130));
            QFont f = p.font();
            f.setPointSize(10);
            p.setFont(f);
        }

        QRect textRect(20, y - LINE_HEIGHT / 2, width() - 40, LINE_HEIGHT);
        p.drawText(textRect, Qt::AlignCenter, lines[i].text);
    }
}

void LyricView::resizeEvent(QResizeEvent *event) {
    QWidget::resizeEvent(event);
    updateOffset();
}

void LyricView::updateOffset() {
    if (m_currentIndex < 0) {
        m_offsetY = 0;
        return;
    }
    m_offsetY = m_currentIndex * LINE_HEIGHT;
}
