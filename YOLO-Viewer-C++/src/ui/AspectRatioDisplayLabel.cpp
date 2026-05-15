#include "AspectRatioDisplayLabel.h"
#include <QtGlobal>
#include <QLayout>

AspectRatioDisplayLabel::AspectRatioDisplayLabel(QWidget *parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(MIN_WIDTH, MIN_HEIGHT);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_pixmapCache.setMaxCost(MAX_CACHE_SIZE);
}

void AspectRatioDisplayLabel::setDisplayPixmap(const QPixmap &pixmap, int frameId)
{
    if (pixmap.isNull()) {
        QLabel::setPixmap(QPixmap());
        return;
    }

    if (frameId >= 0) {
        QPixmap *cached = m_pixmapCache.object(frameId);
        if (cached) {
            QLabel::setPixmap(*cached);
            return;
        }
        QPixmap scaled = scalePixmapToFit(pixmap);
        m_pixmapCache.insert(frameId, new QPixmap(scaled));
        QLabel::setPixmap(scaled);
    } else {
        QLabel::setPixmap(scalePixmapToFit(pixmap));
    }
}

void AspectRatioDisplayLabel::clearCache()
{
    m_pixmapCache.clear();
}

void AspectRatioDisplayLabel::resizeEvent(QResizeEvent *event)
{
    QLabel::resizeEvent(event);

    QWidget *pw = parentWidget();
    if (!pw) return;

    QRect rect = pw->contentsRect();
    int pwW = rect.width();
    int pwH = rect.height();

    int idealW = pwW;
    int idealH = int(idealW * 9.0 / 16.0);
    if (idealH > pwH) {
        idealH = pwH;
        idealW = int(idealH * 16.0 / 9.0);
    }
    setFixedSize(idealW, idealH);
}

QPixmap AspectRatioDisplayLabel::scalePixmapToFit(const QPixmap &pixmap) const
{
    return pixmap.scaled(size(), Qt::KeepAspectRatio, Qt::FastTransformation);
}
