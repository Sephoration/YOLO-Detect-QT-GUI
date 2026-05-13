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

    QWidget *parentWidget = this->parentWidget();
    int parentWidth = width();
    int parentHeight = height();

    if (parentWidget) {
        QRect rect = parentWidget->contentsRect();
        parentWidth = rect.width();
        parentHeight = rect.height();

        QWidget *ctrl = parentWidget->findChild<QWidget *>(QStringLiteral("VideoControlWidget"));
        if (ctrl && ctrl->isVisible()) {
            int ctrlH = ctrl->sizeHint().height();
            QLayout *pl = parentWidget->layout();
            int spacing = pl ? pl->spacing() : 0;
            parentHeight = qMax(0, parentHeight - (ctrlH + spacing));
        }
    }

    int idealWidth = parentWidth;
    int idealHeight = int(idealWidth * 9.0 / 16.0);
    if (idealHeight > parentHeight) {
        idealHeight = parentHeight;
        idealWidth = int(idealHeight * 16.0 / 9.0);
    }
    setFixedSize(idealWidth, idealHeight);
}

QPixmap AspectRatioDisplayLabel::scalePixmapToFit(const QPixmap &pixmap) const
{
    return pixmap.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
}
