#pragma once
#include <QAbstractTableModel>
#include <QList>
#include "MusicItem.h"

class PlaylistModel : public QAbstractTableModel {
    Q_OBJECT
public:
    enum Column {
        Number = 0,
        Title,
        Artist,
        Album,
        Duration,
        ColumnCount
    };

    explicit PlaylistModel(QObject *parent = nullptr);

    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;

    bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex()) override;
    bool moveRows(const QModelIndex &sourceParent, int sourceRow, int count,
                  const QModelIndex &destinationParent, int destinationChild) override;

    void addItem(const MusicItem &item);
    void addItems(const QList<MusicItem> &items);
    void clear();

    MusicItem itemAt(int row) const;
    int indexOf(const QString &filePath) const;
    QList<MusicItem> allItems() const { return m_items; }

    int nextIndex(int current, bool loop, bool shuffle, int *shuffleNext = nullptr) const;
    int previousIndex(int current, bool shuffle, const QList<int> &history) const;

private:
    QList<MusicItem> m_items;
    mutable QList<int> m_shuffleHistory;
};
