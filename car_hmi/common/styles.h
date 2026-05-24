#ifndef STYLES_H
#define STYLES_H

#include <QString>

class Styles {
public:
    static QString globalStyle();
    static QString dockButtonStyle(bool active);
    static QString widgetCardStyle();
    static QString actionButtonStyle();
    static QString toggleStyle(bool active);
    static QString sliderStyle();
    static QString playlistItemStyle(bool active);
    static QString settingItemStyle();
    static QString modeButtonStyle(bool active);
    static QString tempButtonStyle();
};

#endif // STYLES_H
