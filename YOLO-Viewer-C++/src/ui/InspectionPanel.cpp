#include "InspectionPanel.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QHeaderView>
#include <QLineEdit>
#include <QScrollArea>
#include <QFileInfo>
#include <QDateTime>

InspectionPanel::InspectionPanel(QWidget *parent)
    : QWidget(parent)
{
    initUi();
    setupStyle();
}

void InspectionPanel::initUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(4,4,4,4);
    mainLayout->setSpacing(6);

    // ── 模型信息 ──
    auto *modelBox = new QGroupBox(tr("模型信息"), this);
    auto *ml       = new QGridLayout(modelBox);
    ml->addWidget(new QLabel(tr("模型:"), this), 0,0);
    m_modelNameLabel = new QLabel(tr("未加载"), this);
    ml->addWidget(m_modelNameLabel, 0,1);
    ml->addWidget(new QLabel(tr("任务:"), this), 1,0);
    m_taskTypeLabel = new QLabel(tr("-"), this);
    ml->addWidget(m_taskTypeLabel, 1,1);
    ml->addWidget(new QLabel(tr("输入尺寸:"), this), 2,0);
    m_inputSizeLabel = new QLabel(tr("-"), this);
    ml->addWidget(m_inputSizeLabel, 2,1);
    ml->addWidget(new QLabel(tr("类别数:"), this), 3,0);
    m_classCountLabel = new QLabel(tr("-"), this);
    ml->addWidget(m_classCountLabel, 3,1);
    mainLayout->addWidget(modelBox);

    // ── 目标列表 ──
    auto *objBox = new QGroupBox(tr("检测目标"), this);
    auto *ol     = new QVBoxLayout(objBox);

    auto *filterRow = new QHBoxLayout;
    m_filterEdit = new QLineEdit(this);
    m_filterEdit->setPlaceholderText(tr("筛选目标..."));
    connect(m_filterEdit, &QLineEdit::textChanged, this, &InspectionPanel::filterChanged);
    filterRow->addWidget(m_filterEdit, 1);
    m_infoLabel = new QLabel(tr("共 0 个目标"), this);
    filterRow->addWidget(m_infoLabel);
    ol->addLayout(filterRow);

    m_objTable = new QTableWidget(0, 5, this);
    m_objTable->setHorizontalHeaderLabels({tr("标签"), tr("置信度"), tr("X"), tr("Y"), tr("跟踪ID")});
    m_objTable->horizontalHeader()->setStretchLastSection(true);
    m_objTable->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_objTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_objTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_objTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_objTable->verticalHeader()->hide();
    m_objTable->setAlternatingRowColors(true);
    connect(m_objTable, &QTableWidget::cellClicked, this, &InspectionPanel::objectClicked);
    ol->addWidget(m_objTable);
    mainLayout->addWidget(objBox, 1);

    // ── 日志 ──
    auto *logBox = new QGroupBox(tr("系统日志"), this);
    auto *ll     = new QVBoxLayout(logBox);
    m_logView = new QPlainTextEdit(this);
    m_logView->setReadOnly(true);
    m_logView->setMaximumBlockCount(500);
    ll->addWidget(m_logView);
    mainLayout->addWidget(logBox, 1);
}

void InspectionPanel::setupStyle()
{
    setStyleSheet(QStringLiteral(R"(
        QGroupBox { font-weight: normal; border: 1px solid #ccc; border-radius: 4px; margin-top: 6px; padding-top: 10px; font-size: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 4px; }
        QTableWidget { font-size: 10px; }
        QTableWidget::item { padding: 2px; }
        QPlainTextEdit { font-size: 10px; background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; }
        QLineEdit { font-size: 10px; padding: 2px; }
    )"));
}

void InspectionPanel::updateResults(const InferenceResult &result)
{
    m_objTable->setRowCount(0);
    int count = result.detections.size();
    m_infoLabel->setText(tr("共 %1 个目标").arg(count));

    for (int i = 0; i < count; ++i) {
        const auto &d = result.detections[i];
        int r = m_objTable->rowCount();
        m_objTable->insertRow(r);
        m_objTable->setItem(r, 0, new QTableWidgetItem(d.label));
        m_objTable->setItem(r, 1, new QTableWidgetItem(QString::number(d.confidence, 'f', 3)));
        m_objTable->setItem(r, 2, new QTableWidgetItem(QString::number(d.bbox.x(), 'f', 1)));
        m_objTable->setItem(r, 3, new QTableWidgetItem(QString::number(d.bbox.y(), 'f', 1)));
        m_objTable->setItem(r, 4, new QTableWidgetItem(d.trackId >= 0 ? QString::number(d.trackId) : "-"));
    }
}

void InspectionPanel::updateModelInfo(const ModelInfo &info)
{
    m_modelNameLabel->setText(QFileInfo(info.modelPath).fileName());
    m_taskTypeLabel->setText(info.taskType);
    m_inputSizeLabel->setText(info.inputSize);
    m_classCountLabel->setText(QString::number(info.classCount));
}

void InspectionPanel::appendLog(const QString &msg)
{
    m_logView->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss.zzz ") + msg);
}

void InspectionPanel::clearLog()
{
    m_logView->clear();
}

void InspectionPanel::clearResults()
{
    m_objTable->setRowCount(0);
    m_infoLabel->setText(tr("共 0 个目标"));
}
