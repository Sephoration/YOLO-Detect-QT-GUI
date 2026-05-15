#include "RightControlPanel.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFileInfo>

RightControlPanel::RightControlPanel(QWidget *parent)
    : QWidget(parent)
{
    setMinimumWidth(MIN_W);
    initUi();
    setupStyle();
}

void RightControlPanel::initUi()
{
    auto *scroll = new QScrollArea(this);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto *content = new QWidget;
    auto *layout  = new QVBoxLayout(content);
    layout->setContentsMargins(8,8,8,8);
    layout->setSpacing(8);

    // ── 源操作 ──
    auto *srcGroup = new QGroupBox(tr("操作"), content);
    auto *srcL     = new QVBoxLayout(srcGroup);

    auto *row1 = new QHBoxLayout;
    m_modelBtn = new QPushButton(tr("📦 模型"), content);
    connect(m_modelBtn, &QPushButton::clicked, this, &RightControlPanel::loadModelClicked);
    m_imageBtn = new QPushButton(tr("🖼 图片"), content);
    connect(m_imageBtn, &QPushButton::clicked, this, &RightControlPanel::loadImageClicked);
    row1->addWidget(m_modelBtn);
    row1->addWidget(m_imageBtn);
    srcL->addLayout(row1);

    auto *row2 = new QHBoxLayout;
    m_videoBtn = new QPushButton(tr("🎬 视频"), content);
    connect(m_videoBtn, &QPushButton::clicked, this, &RightControlPanel::loadVideoClicked);
    m_cameraBtn = new QPushButton(tr("📷 摄像头"), content);
    connect(m_cameraBtn, &QPushButton::clicked, this, &RightControlPanel::loadCameraClicked);
    row2->addWidget(m_videoBtn);
    row2->addWidget(m_cameraBtn);
    srcL->addLayout(row2);

    layout->addWidget(srcGroup);

    // ── 模型信息 ──
    auto *infoGroup = new QGroupBox(tr("模型信息"), content);
    auto *infoL     = new QVBoxLayout(infoGroup);
    m_modelLabel  = new QLabel(tr("模型: 未加载"), content);
    m_taskLabel   = new QLabel(tr("任务: -"), content);
    m_sizeLabel   = new QLabel(tr("尺寸: -"), content);
    m_classesLabel= new QLabel(tr("类别数: -"), content);
    infoL->addWidget(m_modelLabel);
    infoL->addWidget(m_taskLabel);
    infoL->addWidget(m_sizeLabel);
    infoL->addWidget(m_classesLabel);
    layout->addWidget(infoGroup);

    // ── 模式选择 ──
    auto *modeRow = new QHBoxLayout;
    modeRow->addWidget(new QLabel(tr("检测模式:"), content));
    m_modeCombo = new QComboBox(content);
    m_modeCombo->addItems({tr("目标检测"), tr("图像分类"), tr("关键点检测"), tr("分割检测")});
    m_modeCombo->setCurrentIndex(0);
    connect(m_modeCombo, &QComboBox::currentTextChanged, this, [this](const QString &) {
        // 映射到内部名
        static const QStringList map = {"detection","classification","pose","segmentation"};
        emit modelModeChanged(map.value(m_modeCombo->currentIndex(), "detection"));
    });
    modeRow->addWidget(m_modeCombo, 1);
    layout->addLayout(modeRow);

    // ── 推理参数 ──
    auto *paramGroup = new QGroupBox(tr("推理参数"), content);
    auto *paramL     = new QVBoxLayout(paramGroup);
    paramL->setSpacing(4);

    auto *iw = makeSlider(tr("IOU:"), 0.0, 1.0, 0.45, 100, &RightControlPanel::iouChanged);
    m_iouSlider   = iw->findChild<QSlider*>();
    m_iouValLabel = iw->findChildren<QLabel*>().last();
    paramL->addWidget(iw);

    auto *cw = makeSlider(tr("置信度:"), 0.0, 1.0, 0.5, 100, &RightControlPanel::confidenceChanged);
    m_confSlider   = cw->findChild<QSlider*>();
    m_confValLabel = cw->findChildren<QLabel*>().last();
    paramL->addWidget(cw);

    auto *dw = makeIntSlider(tr("延迟(ms):"), 0, 200, 10, &RightControlPanel::delayChanged);
    m_delaySlider   = dw->findChild<QSlider*>();
    m_delayValLabel = dw->findChildren<QLabel*>().last();
    paramL->addWidget(dw);

    auto *lw = makeIntSlider(tr("线宽:"), 1, 10, 2, &RightControlPanel::lineWidthChanged);
    m_lineWidthSlider   = lw->findChild<QSlider*>();
    m_lineWidthValLabel = lw->findChildren<QLabel*>().last();
    paramL->addWidget(lw);

    layout->addWidget(paramGroup);

    // ── 实时统计 ──
    auto *statGroup = new QGroupBox(tr("实时统计"), content);
    auto *statL     = new QVBoxLayout(statGroup);
    m_detCntLabel = new QLabel(tr("检测数: 0"), content);
    m_confLabel   = new QLabel(tr("置信度: 0.00"), content);
    m_timeLabel   = new QLabel(tr("推理时间: 0 ms"), content);
    m_fpsLabel    = new QLabel(tr("FPS: 0.0"), content);
    statL->addWidget(m_detCntLabel);
    statL->addWidget(m_confLabel);
    statL->addWidget(m_timeLabel);
    statL->addWidget(m_fpsLabel);
    layout->addWidget(statGroup);

    // ── 控制按钮 ──
    auto *btnRow = new QHBoxLayout;
    m_startBtn = new QPushButton(tr("开始推理"), content);
    m_startBtn->setMinimumHeight(34);
    connect(m_startBtn, &QPushButton::clicked, this, &RightControlPanel::startInference);
    m_stopBtn = new QPushButton(tr("停止"), content);
    m_stopBtn->setMinimumHeight(34);
    m_stopBtn->setEnabled(false);
    connect(m_stopBtn, &QPushButton::clicked, this, &RightControlPanel::stopInference);
    btnRow->addWidget(m_startBtn);
    btnRow->addWidget(m_stopBtn);
    layout->addLayout(btnRow);

    m_saveBtn = new QPushButton(tr("保存截图"), content);
    m_saveBtn->setMinimumHeight(30);
    connect(m_saveBtn, &QPushButton::clicked, this, &RightControlPanel::saveScreenshot);
    layout->addWidget(m_saveBtn);

    layout->addStretch(1);
    scroll->setWidget(content);

    auto *mainL = new QVBoxLayout(this);
    mainL->setContentsMargins(0,0,0,0);
    mainL->addWidget(scroll);
}

void RightControlPanel::setupStyle()
{
    setStyleSheet(QStringLiteral(R"(
        QGroupBox { font-weight: normal; border: 1px solid #ccc; border-radius: 4px; margin-top: 6px; padding-top: 10px; font-size: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 4px; }
        QPushButton { background-color: #3a3a3a; color: #eee; border: 1px solid #555; border-radius: 3px; padding: 4px 6px; font-size: 11px; }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:disabled { background-color: #2a2a2a; color: #666; }
        QLabel { color: #ccc; font-size: 10px; }
        QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }
        QSlider::handle:horizontal { background: #4a9eff; width: 10px; height: 10px; margin: -3px 0; border-radius: 5px; }
        QComboBox { background: #3a3a3a; color: #eee; border: 1px solid #555; border-radius: 3px; padding: 2px 4px; font-size: 10px; }
    )"));
}

// ── Slider helpers ──
QWidget* RightControlPanel::makeSlider(const QString &labelText, double minVal, double maxVal,
                                        double defaultVal, int scale,
                                        void (RightControlPanel::*sig)(double))
{
    auto *w = new QWidget;
    auto *l = new QHBoxLayout(w);
    l->setContentsMargins(0,0,0,0);

    auto *lb = new QLabel(labelText, w);
    lb->setMinimumWidth(48);
    l->addWidget(lb);

    auto *sl = new QSlider(Qt::Horizontal, w);
    sl->setRange(int(minVal*scale), int(maxVal*scale));
    sl->setValue(int(defaultVal*scale));

    auto *vl = new QLabel(QString::number(defaultVal,'f',2), w);
    vl->setFixedWidth(40);
    vl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

    connect(sl, &QSlider::valueChanged, this, [this,vl,sig,scale](int v) {
        double val = double(v)/scale;
        vl->setText(QString::number(val,'f',2));
        (this->*sig)(val);
    });

    l->addWidget(sl, 1);
    l->addWidget(vl);
    return w;
}

QWidget* RightControlPanel::makeIntSlider(const QString &labelText, int minVal, int maxVal,
                                           int defaultVal,
                                           void (RightControlPanel::*sig)(int))
{
    auto *w = new QWidget;
    auto *l = new QHBoxLayout(w);
    l->setContentsMargins(0,0,0,0);

    auto *lb = new QLabel(labelText, w);
    lb->setMinimumWidth(48);
    l->addWidget(lb);

    auto *sl = new QSlider(Qt::Horizontal, w);
    sl->setRange(minVal, maxVal);
    sl->setValue(defaultVal);

    auto *vl = new QLabel(QString::number(defaultVal), w);
    vl->setFixedWidth(32);
    vl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);

    connect(sl, &QSlider::valueChanged, this, [this,vl,sig](int v) {
        vl->setText(QString::number(v));
        (this->*sig)(v);
    });

    l->addWidget(sl, 1);
    l->addWidget(vl);
    return w;
}

// ── Public API ──
void RightControlPanel::updateModelInfo(const QString &modelName, const QString &taskType,
                                         const QString &inputSize, const QString &classCount)
{
    m_modelLabel->setText(tr("模型: %1").arg(modelName.isEmpty() ? tr("未加载") : modelName));
    m_taskLabel->setText(tr("任务: %1").arg(taskType.isEmpty() ? "-" : taskType));
    m_sizeLabel->setText(tr("尺寸: %1").arg(inputSize.isEmpty() ? "-" : inputSize));
    m_classesLabel->setText(tr("类别数: %1").arg(classCount.isEmpty() ? "-" : classCount));
}

void RightControlPanel::updateStatistics(int detectionCount, double confidence,
                                          double inferenceTime, double fps)
{
    m_detCntLabel->setText(tr("检测数: %1").arg(detectionCount));
    m_confLabel->setText(tr("置信度: %1").arg(confidence, 0, 'f', 3));
    m_timeLabel->setText(tr("推理时间: %1 ms").arg(inferenceTime, 0, 'f', 2));
    m_fpsLabel->setText(tr("FPS: %1").arg(fps, 0, 'f', 1));
}

void RightControlPanel::setControlState(bool isRunning)
{
    m_startBtn->setEnabled(!isRunning);
    m_stopBtn->setEnabled(isRunning);
}

void RightControlPanel::setMode(const QString &mode)
{
    static const QMap<QString, int> modeIndex = {
        {"detection", 0},
        {"classification", 1},
        {"pose", 2},
        {"segmentation", 3},
    };
    int idx = modeIndex.value(mode, 0);
    m_modeCombo->blockSignals(true);
    m_modeCombo->setCurrentIndex(idx);
    m_modeCombo->blockSignals(false);
}

QMap<QString, QVariant> RightControlPanel::getParameters() const
{
    return {
        {"iou_threshold",        m_iouSlider ? double(m_iouSlider->value()) / 100.0 : 0.45},
        {"confidence_threshold", m_confSlider ? double(m_confSlider->value()) / 100.0 : 0.5},
        {"delay_ms",             m_delaySlider ? m_delaySlider->value() : 10},
        {"line_width",           m_lineWidthSlider ? m_lineWidthSlider->value() : 2},
    };
}

void RightControlPanel::setParameters(double iou, double conf, int delay, int lineWidth)
{
    if (m_iouSlider)      m_iouSlider->setValue(int(iou * 100));
    if (m_confSlider)     m_confSlider->setValue(int(conf * 100));
    if (m_delaySlider)    m_delaySlider->setValue(delay);
    if (m_lineWidthSlider) m_lineWidthSlider->setValue(lineWidth);
}
