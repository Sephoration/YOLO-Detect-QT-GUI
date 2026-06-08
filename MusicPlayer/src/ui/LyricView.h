#pragma once
#include <QWidget>
#include <QPropertyAnimation>
#include "core/LyricParser.h"

class LyricView : public QWidget {
    Q_OBJECT
public:
    explicit LyricView(QWidget *parent = nullptr);

    void loadFromFile(const QString &audioFilePath);
    void setPosition(qint64 positionMs);
    void clear();

protected:
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    LyricParser m_parser;
    int m_currentIndex = -1;
    int m_offsetY = 0;
    static constexpr int LINE_HEIGHT = 36;
    QPropertyAnimation *m_animation;

    void updateOffset();
};
