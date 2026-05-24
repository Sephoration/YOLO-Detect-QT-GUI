#include "PlaylistView.h"
#include <QVBoxLayout>
#include <QTableView>
#include <QLineEdit>
#include <QSortFilterProxyModel>
#include <QHeaderView>
#include <QMenu>
#include <QKeyEvent>

PlaylistView::PlaylistView(PlaylistModel *model, QWidget *parent)
    : QWidget(parent), m_sourceModel(model) {
    setupUI();
}

void PlaylistView::setupUI() {
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    m_searchEdit = new QLineEdit(this);
    m_searchEdit->setPlaceholderText("🔍 搜索歌曲、艺术家、专辑...");
    m_searchEdit->setStyleSheet(
        "QLineEdit { background: #1e1e2a; color: #ccc; border: 1px solid #2a2a3a; "
        "border-radius: 6px; padding: 6px 10px; }"
        "QLineEdit:focus { border-color: #00d4ff; }"
    );

    m_tableView = new QTableView(this);
    m_tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_tableView->setSelectionMode(QAbstractItemView::ExtendedSelection);
    m_tableView->setAlternatingRowColors(false);
    m_tableView->setShowGrid(false);
    m_tableView->verticalHeader()->setVisible(false);
    m_tableView->horizontalHeader()->setStretchLastSection(true);
    m_tableView->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    m_tableView->horizontalHeader()->setDefaultAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    m_tableView->setContextMenuPolicy(Qt::CustomContextMenu);
    m_tableView->setDragEnabled(true);
    m_tableView->setAcceptDrops(true);
    m_tableView->setDropIndicatorShown(true);
    m_tableView->setDragDropMode(QAbstractItemView::InternalMove);
    m_tableView->setStyleSheet(
        "QTableView { background: #16161e; color: #ccc; border: none; outline: none; }"
        "QTableView::item { padding: 6px 4px; border-bottom: 1px solid #1e1e2a; }"
        "QTableView::item:selected { background: #252540; color: #fff; }"
        "QTableView::item:hover { background: #1e1e30; }"
        "QHeaderView::section { background: #16161e; color: #888; padding: 8px 4px; "
        "border: none; border-bottom: 2px solid #2a2a3a; }"
        "QScrollBar:vertical { background: #16161e; width: 8px; }"
        "QScrollBar::handle:vertical { background: #333; border-radius: 4px; }"
    );

    m_proxyModel = new QSortFilterProxyModel(this);
    m_proxyModel->setSourceModel(m_sourceModel);
    m_proxyModel->setFilterCaseSensitivity(Qt::CaseInsensitive);
    m_proxyModel->setFilterRole(Qt::DisplayRole);

    m_tableView->setModel(m_proxyModel);
    m_tableView->horizontalHeader()->setSectionResizeMode(PlaylistModel::Number, QHeaderView::Fixed);
    m_tableView->setColumnWidth(PlaylistModel::Number, 40);
    m_tableView->setColumnWidth(PlaylistModel::Title, 200);
    m_tableView->setColumnWidth(PlaylistModel::Artist, 120);
    m_tableView->setColumnWidth(PlaylistModel::Album, 120);
    m_tableView->setColumnWidth(PlaylistModel::Duration, 60);

    layout->addWidget(m_searchEdit);
    layout->addWidget(m_tableView);

    connect(m_searchEdit, &QLineEdit::textChanged, this, [this](const QString &text) {
        m_proxyModel->setFilterRegularExpression(text);
    });

    connect(m_tableView, &QTableView::doubleClicked, this, &PlaylistView::onDoubleClicked);
    connect(m_tableView, &QTableView::customContextMenuRequested, this, &PlaylistView::onContextMenu);
}

void PlaylistView::setCurrentRow(int row) {
    QModelIndex sourceIndex = m_sourceModel->index(row, 0);
    QModelIndex proxyIndex = m_proxyModel->mapFromSource(sourceIndex);
    m_tableView->setCurrentIndex(proxyIndex);
    m_tableView->scrollTo(proxyIndex, QAbstractItemView::PositionAtCenter);
}

void PlaylistView::onDoubleClicked(const QModelIndex &index) {
    if (!index.isValid()) return;
    QModelIndex sourceIndex = m_proxyModel->mapToSource(index);
    emit activated(sourceIndex.row());
}

void PlaylistView::onContextMenu(const QPoint &pos) {
    QModelIndex idx = m_tableView->indexAt(pos);
    if (!idx.isValid()) return;

    QMenu menu(this);
    menu.setStyleSheet("QMenu { background: #1e1e2a; color: #ccc; border: 1px solid #2a2a3a; }"
                       "QMenu::item:selected { background: #252540; }");
    QAction *del = menu.addAction("删除");
    if (menu.exec(m_tableView->viewport()->mapToGlobal(pos)) == del) {
        QModelIndex sourceIdx = m_proxyModel->mapToSource(idx);
        emit removeRequested(sourceIdx.row());
    }
}
