#include "AlbumCover.h"
#include <QPainter>
#include <QPainterPath>
#include <QPaintEvent>
#include <QImageReader>

AlbumCover::AlbumCover(QWidget *parent)
    : QWidget(parent) {
    setMinimumSize(200, 200);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void AlbumCover::setCover(const QByteArray &data) {
    QImage img;
    if (!data.isEmpty() && img.loadFromData(data)) {
        m_pixmap = QPixmap::fromImage(img);
    } else {
        m_pixmap = QPixmap();
    }
    update();
}

void AlbumCover::setCover(const QPixmap &pixmap) {
    m_pixmap = pixmap;
    update();
}

void AlbumCover::clear() {
    m_pixmap = QPixmap();
    update();
}

void AlbumCover::paintEvent(QPaintEvent *event) {
    Q_UNUSED(event)
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    QRect rect = this->rect().adjusted(8, 8, -8, -8);
    int side = qMin(rect.width(), rect.height());
    QRect target((width() - side) / 2, (height() - side) / 2, side, side);

    int radius = 12;
    QPainterPath path;
    path.addRoundedRect(target, radius, radius);
    p.setClipPath(path);

    if (!m_pixmap.isNull()) {
        QPixmap scaled = m_pixmap.scaled(target.size(), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
        p.drawPixmap(target, scaled);
    } else {
        p.fillRect(target, QColor(40, 40, 50));
        p.setPen(QColor(120, 120, 130));
        p.drawText(target, Qt::AlignCenter, "🎵\n无封面");
    }

    p.setClipping(false);
    QPen pen(QColor(255, 255, 255, 30));
    pen.setWidth(1);
    p.setPen(pen);
    p.drawRoundedRect(target, radius, radius);
}
