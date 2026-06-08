#pragma once
#include <QTableView>
#include <QLineEdit>
#include <QSortFilterProxyModel>
#include "core/PlaylistModel.h"

class PlaylistView : public QWidget {
    Q_OBJECT
public:
    explicit PlaylistView(PlaylistModel *model, QWidget *parent = nullptr);

    QTableView *tableView() const { return m_tableView; }

signals:
    void activated(int row);
    void removeRequested(int row);

public slots:
    void setCurrentRow(int row);

private:
    QTableView *m_tableView;
    QLineEdit *m_searchEdit;
    PlaylistModel *m_sourceModel;
    QSortFilterProxyModel *m_proxyModel;

    void setupUI();
    void onDoubleClicked(const QModelIndex &index);
    void onContextMenu(const QPoint &pos);
};
