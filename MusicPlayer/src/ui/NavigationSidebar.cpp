#include "NavigationSidebar.h"
#include <QListWidgetItem>

NavigationSidebar::NavigationSidebar(QWidget *parent)
    : QListWidget(parent) {
    setupUI();
    setFixedWidth(160);
    setFocusPolicy(Qt::NoFocus);

    connect(this, &QListWidget::currentRowChanged, this, [this](int row) {
        emit pageChanged(static_cast<Page>(row));
    });
}

void NavigationSidebar::setupUI() {
    setStyleSheet(
        "QListWidget { background: #16161e; border: none; padding: 8px 4px; }"
        "QListWidget::item { color: #888; padding: 10px 12px; border-radius: 6px; margin: 2px 4px; }"
        "QListWidget::item:selected { background: #252532; color: #00d4ff; }"
        "QListWidget::item:hover { background: #1e1e2a; color: #ccc; }"
    );

    auto *playing = new QListWidgetItem("🎵 正在播放", this);
    auto *library = new QListWidgetItem("📂 音乐库", this);
    auto *favorites = new QListWidgetItem("❤️ 我喜欢", this);

    for (auto *item : {playing, library, favorites}) {
        item->setData(Qt::UserRole, item->text());
    }

    setCurrentRow(0);
}
