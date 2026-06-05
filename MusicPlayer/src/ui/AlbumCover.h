#pragma once
#include <QWidget>
#include <QPixmap>

class AlbumCover : public QWidget {
    Q_OBJECT
public:
    explicit AlbumCover(QWidget *parent = nullptr);

    void setCover(const QByteArray &data);
    void setCover(const QPixmap &pixmap);
    void clear();

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QPixmap m_pixmap;
    QPixmap m_defaultPixmap;
    static QPixmap roundPixmap(const QPixmap &src, int radius);
};
