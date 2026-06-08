#pragma once
#include <QListWidget>

class NavigationSidebar : public QListWidget {
    Q_OBJECT
public:
    enum Page {
        Playing = 0,
        Library,
        Favorites
    };

    explicit NavigationSidebar(QWidget *parent = nullptr);

signals:
    void pageChanged(Page page);

private:
    void setupUI();
};
