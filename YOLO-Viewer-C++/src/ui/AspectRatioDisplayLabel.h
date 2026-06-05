#pragma once

#include <QLabel>
#include <QPixmap>
#include <QCache>
#include <QSize>

class AspectRatioDisplayLabel : public QLabel
{
    Q_OBJECT
public:
    explicit AspectRatioDisplayLabel(QWidget *parent = nullptr);

    void setDisplayPixmap(const QPixmap &pixmap, int frameId = -1);
    void clearCache();

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    QPixmap scalePixmapToFit(const QPixmap &pixmap) const;

    QCache<int, QPixmap> m_pixmapCache;
    static constexpr int MAX_CACHE_SIZE = 100;
    static constexpr int MIN_WIDTH = 320;
    static constexpr int MIN_HEIGHT = 180;
};
