#pragma once

#include <QWidget>
#include <QTableWidget>
#include <QLabel>
#include <QGroupBox>
#include <QPlainTextEdit>
#include "models/DetectionResult.h"

/**
 * @brief 检测结果检查面板
 *
 * 实时显示检测目标列表、模型信息和系统日志。
 */
class InspectionPanel : public QWidget
{
    Q_OBJECT
public:
    explicit InspectionPanel(QWidget *parent = nullptr);

    void updateResults(const InferenceResult &result);
    void updateModelInfo(const ModelInfo &info);
    void appendLog(const QString &msg);
    void clearLog();
    void clearResults();

signals:
    void objectClicked(int row);          // 选中某行检测目标
    void filterChanged(const QString &text);

private:
    void initUi();
    void setupStyle();

    QTableWidget *m_objTable = nullptr;
    QLabel       *m_modelNameLabel  = nullptr;
    QLabel       *m_taskTypeLabel   = nullptr;
    QLabel       *m_inputSizeLabel  = nullptr;
    QLabel       *m_classCountLabel = nullptr;
    QLabel       *m_infoLabel       = nullptr;   // "共 N 个目标"
    QPlainTextEdit *m_logView       = nullptr;
    QLineEdit    *m_filterEdit      = nullptr;
};
