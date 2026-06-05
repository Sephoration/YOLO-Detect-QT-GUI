#include "PlaylistModel.h"
#include <QRandomGenerator>
#include <QMimeData>
#include <QUrl>
#include <QFileInfo>

PlaylistModel::PlaylistModel(QObject *parent)
    : QAbstractTableModel(parent) {}

int PlaylistModel::rowCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : m_items.size();
}

int PlaylistModel::columnCount(const QModelIndex &parent) const {
    return parent.isValid() ? 0 : ColumnCount;
}

QVariant PlaylistModel::data(const QModelIndex &index, int role) const {
    if (!index.isValid() || index.row() >= m_items.size())
        return QVariant();

    const MusicItem &item = m_items[index.row()];

    if (role == Qt::DisplayRole) {
        switch (index.column()) {
        case Number: return index.row() + 1;
        case Title: return item.title.isEmpty() ? QFileInfo(item.filePath).fileName() : item.title;
        case Artist: return item.artist;
        case Album: return item.album;
        case Duration: return item.durationStr;
        default: return QVariant();
        }
    }

    if (role == Qt::TextAlignmentRole) {
        if (index.column() == Number || index.column() == Duration)
            return int(Qt::AlignCenter);
        return int(Qt::AlignLeft | Qt::AlignVCenter);
    }

    if (role == Qt::ToolTipRole)
        return item.filePath;

    if (role == Qt::UserRole)
        return QVariant::fromValue(item);

    return QVariant();
}

QVariant PlaylistModel::headerData(int section, Qt::Orientation orientation, int role) const {
    if (orientation != Qt::Horizontal || role != Qt::DisplayRole)
        return QVariant();

    switch (section) {
    case Number: return "#";
    case Title: return "标题";
    case Artist: return "艺术家";
    case Album: return "专辑";
    case Duration: return "时长";
    default: return QVariant();
    }
}

Qt::ItemFlags PlaylistModel::flags(const QModelIndex &index) const {
    Qt::ItemFlags f = QAbstractTableModel::flags(index);
    if (index.isValid()) {
        f |= Qt::ItemIsDragEnabled;
    } else {
        f |= Qt::ItemIsDropEnabled;
    }
    return f;
}

bool PlaylistModel::insertRows(int row, int count, const QModelIndex &parent) {
    if (parent.isValid()) return false;
    beginInsertRows(parent, row, row + count - 1);
    m_items.insert(row, count, MusicItem());
    endInsertRows();
    return true;
}

bool PlaylistModel::removeRows(int row, int count, const QModelIndex &parent) {
    if (parent.isValid() || row < 0 || row + count > m_items.size())
        return false;
    beginRemoveRows(parent, row, row + count - 1);
    m_items.remove(row, count);
    endRemoveRows();
    return true;
}

bool PlaylistModel::moveRows(const QModelIndex &sourceParent, int sourceRow, int count,
                             const QModelIndex &destinationParent, int destinationChild) {
    if (sourceParent.isValid() || destinationParent.isValid())
        return false;
    if (sourceRow < 0 || sourceRow + count > m_items.size() || destinationChild < 0 || destinationChild > m_items.size())
        return false;

    beginMoveRows(sourceParent, sourceRow, sourceRow + count - 1, destinationParent, destinationChild);
    QList<MusicItem> tmp;
    for (int i = 0; i < count; ++i)
        tmp.append(m_items.takeAt(sourceRow));

    int dest = destinationChild;
    if (destinationChild > sourceRow)
        dest -= count;
    for (int i = 0; i < count; ++i)
        m_items.insert(dest + i, tmp[i]);
    endMoveRows();
    return true;
}

void PlaylistModel::addItem(const MusicItem &item) {
    int row = m_items.size();
    beginInsertRows(QModelIndex(), row, row);
    m_items.append(item);
    endInsertRows();
}

void PlaylistModel::addItems(const QList<MusicItem> &items) {
    if (items.isEmpty()) return;
    int row = m_items.size();
    beginInsertRows(QModelIndex(), row, row + items.size() - 1);
    m_items.append(items);
    endInsertRows();
}

void PlaylistModel::clear() {
    if (m_items.isEmpty()) return;
    beginResetModel();
    m_items.clear();
    endResetModel();
}

MusicItem PlaylistModel::itemAt(int row) const {
    if (row < 0 || row >= m_items.size()) return MusicItem();
    return m_items[row];
}

int PlaylistModel::indexOf(const QString &filePath) const {
    for (int i = 0; i < m_items.size(); ++i) {
        if (m_items[i].filePath == filePath)
            return i;
    }
    return -1;
}

int PlaylistModel::nextIndex(int current, bool loop, bool shuffle, int *shuffleNext) const {
    if (m_items.isEmpty()) return -1;
    if (m_items.size() == 1) return 0;

    if (shuffle) {
        if (shuffleNext && *shuffleNext >= 0 && *shuffleNext < m_items.size())
            return *shuffleNext;
        int next;
        do {
            next = QRandomGenerator::global()->bounded(m_items.size());
        } while (next == current && m_items.size() > 1);
        return next;
    }

    int next = current + 1;
    if (next >= m_items.size())
        return loop ? 0 : -1;
    return next;
}

int PlaylistModel::previousIndex(int current, bool shuffle, const QList<int> &history) const {
    if (m_items.isEmpty()) return -1;
    if (m_items.size() == 1) return 0;

    if (shuffle && !history.isEmpty()) {
        for (int i = history.size() - 1; i >= 0; --i) {
            if (history[i] != current)
                return history[i];
        }
    }

    int prev = current - 1;
    if (prev < 0)
        prev = m_items.size() - 1;
    return prev;
}
