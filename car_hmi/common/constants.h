#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <QColor>

namespace Colors {
    inline constexpr QColor bgDark(10, 10, 15);
    inline constexpr QColor bgPanel(18, 18, 26);
    inline constexpr QColor bgCard(26, 26, 36);
    inline constexpr QColor accent(0, 212, 255);
    inline constexpr QColor accentWarn(255, 149, 0);
    inline constexpr QColor accentDanger(255, 51, 102);
    inline constexpr QColor accentSuccess(0, 230, 118);
    inline constexpr QColor textPrimary(255, 255, 255);
    inline constexpr QColor textSecondary(139, 139, 158);
    inline constexpr QColor border(42, 42, 58);
    inline constexpr QColor glass(255, 255, 255, 13);
}

namespace Dimens {
    inline constexpr int cockpitWidth = 1920;
    inline constexpr int cockpitHeight = 720;
    inline constexpr int dashboardWidth = 720;
    inline constexpr int dockHeight = 80;
}

#endif // CONSTANTS_H
