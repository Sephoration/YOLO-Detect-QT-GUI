#include "styles.h"

QString Styles::globalStyle() {
    return R"(
        QWidget {
            color: #ffffff;
            font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
        }
        QMainWindow {
            background: #050508;
        }
        QLabel {
            color: #ffffff;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #00a8e8);
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            margin: -4px 0;
            background: #ffffff;
            border-radius: 7px;
        }
        QScrollBar:vertical {
            width: 6px;
            background: transparent;
        }
        QScrollBar::handle:vertical {
            background: #2a2a3a;
            border-radius: 3px;
        }
        QLineEdit {
            background: rgba(0,0,0,0.7);
            border: 1px solid #2a2a3a;
            border-radius: 12px;
            padding: 10px 16px;
            color: #ffffff;
            font-size: 15px;
        }
        QLineEdit:focus {
            border-color: #00d4ff;
        }
    )";
}

QString Styles::dockButtonStyle(bool active) {
    if (active) {
        return R"(
            QPushButton {
                background: rgba(0, 212, 255, 0.1);
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 12px;
                color: #00d4ff;
                padding: 8px 16px;
                font-size: 11px;
            }
        )";
    }
    return R"(
        QPushButton {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 12px;
            color: #8b8b9e;
            padding: 8px 16px;
            font-size: 11px;
        }
        QPushButton:hover {
            background: rgba(255,255,255,0.05);
            color: #ffffff;
        }
    )";
}

QString Styles::widgetCardStyle() {
    return R"(
        QWidget {
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 16px;
        }
    )";
}

QString Styles::actionButtonStyle() {
    return R"(
        QPushButton {
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 16px;
            color: #ffffff;
            padding: 16px;
        }
        QPushButton:hover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.05);
        }
    )";
}

QString Styles::toggleStyle(bool active) {
    if (active) {
        return R"(
            QPushButton {
                background: #00d4ff;
                border: none;
                border-radius: 14px;
                min-width: 52px;
                max-width: 52px;
                min-height: 28px;
                max-height: 28px;
            }
        )";
    }
    return R"(
        QPushButton {
            background: rgba(255,255,255,0.08);
            border: 1px solid #2a2a3a;
            border-radius: 14px;
            min-width: 52px;
            max-width: 52px;
            min-height: 28px;
            max-height: 28px;
        }
    )";
}

QString Styles::sliderStyle() {
    return R"(
        QSlider::groove:horizontal {
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00d4ff, stop:1 #00a8e8);
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            width: 14px;
            height: 14px;
            margin: -4px 0;
            background: #ffffff;
            border-radius: 7px;
        }
    )";
}

QString Styles::playlistItemStyle(bool active) {
    if (active) {
        return R"(
            QPushButton {
                background: rgba(255,255,255,0.05);
                border-left: 3px solid #00d4ff;
                border-top: none;
                border-right: none;
                border-bottom: none;
                border-radius: 0px;
                color: #ffffff;
                text-align: left;
                padding: 10px 16px;
            }
        )";
    }
    return R"(
        QPushButton {
            background: transparent;
            border: none;
            color: #ffffff;
            text-align: left;
            padding: 10px 16px;
        }
        QPushButton:hover {
            background: rgba(255,255,255,0.05);
        }
    )";
}

QString Styles::settingItemStyle() {
    return R"(
        QWidget {
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 12px;
        }
    )";
}

QString Styles::modeButtonStyle(bool active) {
    if (active) {
        return R"(
            QPushButton {
                background: rgba(0, 212, 255, 0.05);
                border: 1px solid #00d4ff;
                border-radius: 16px;
                color: #00d4ff;
                padding: 8px;
            }
        )";
    }
    return R"(
        QPushButton {
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 16px;
            color: #8b8b9e;
            padding: 8px;
        }
        QPushButton:hover {
            border-color: #00d4ff;
            color: #00d4ff;
        }
    )";
}

QString Styles::tempButtonStyle() {
    return R"(
        QPushButton {
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 24px;
            color: #ffffff;
            font-size: 20px;
            min-width: 48px;
            max-width: 48px;
            min-height: 48px;
            max-height: 48px;
        }
        QPushButton:hover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }
    )";
}
