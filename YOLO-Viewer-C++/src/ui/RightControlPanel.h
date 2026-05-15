#pragma once

#include <QWidget>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QGroupBox>
#include <QCheckBox>
#include <QMap>
#include <QVariant>

class RightControlPanel : public QWidget
{
    Q_OBJECT
public:
    explicit RightControlPanel(QWidget *parent = nullptr);

    // ── 更新 ──
    void updateModelInfo(const QString &modelName, const QString &taskType,
                         const QString &inputSize, const QString &classCount);
    void updateStatistics(int detectionCount, double confidence,
                          double inferenceTime, double fps);

    // ── 状态 ──
    void setControlState(bool isRunning);
    void setMode(const QString &mode);
    QMap<QString, QVariant> getParameters() const;
    void setParameters(double iou, double conf, int delay, int lineWidth);

signals:
    void iouChanged(double);
    void confidenceChanged(double);
    void delayChanged(int);
    void lineWidthChanged(int);
    void modelModeChanged(const QString &mode);
    void saveScreenshot();
    void startInference();
    void stopInference();
    void loadModelClicked();
    void loadImageClicked();
    void loadVideoClicked();
    void loadCameraClicked();

private:
    void initUi();
    void setupStyle();

    QWidget* makeSlider(const QString &label, double minVal, double maxVal,
                        double defaultVal, int scale,
                        void (RightControlPanel::*sig)(double));
    QWidget* makeIntSlider(const QString &label, int minVal, int maxVal,
                           int defaultVal,
                           void (RightControlPanel::*sig)(int));

    // ── Sliders ──
    QSlider *m_iouSlider        = nullptr;
    QLabel  *m_iouValLabel      = nullptr;
    QSlider *m_confSlider       = nullptr;
    QLabel  *m_confValLabel     = nullptr;
    QSlider *m_delaySlider      = nullptr;
    QLabel  *m_delayValLabel    = nullptr;
    QSlider *m_lineWidthSlider  = nullptr;
    QLabel  *m_lineWidthValLabel= nullptr;

    // ── 模型信息 ──
    QLabel *m_modelLabel   = nullptr;
    QLabel *m_taskLabel    = nullptr;
    QLabel *m_sizeLabel    = nullptr;
    QLabel *m_classesLabel = nullptr;

    // ── 统计 ──
    QLabel *m_detCntLabel = nullptr;
    QLabel *m_confLabel   = nullptr;
    QLabel *m_timeLabel   = nullptr;
    QLabel *m_fpsLabel    = nullptr;

    // ── 模式选择 ──
    QComboBox *m_modeCombo = nullptr;

    // ── 按钮 ──
    QPushButton *m_startBtn  = nullptr;
    QPushButton *m_stopBtn   = nullptr;
    QPushButton *m_saveBtn   = nullptr;
    QPushButton *m_modelBtn  = nullptr;
    QPushButton *m_imageBtn  = nullptr;
    QPushButton *m_videoBtn  = nullptr;
    QPushButton *m_cameraBtn = nullptr;

    static constexpr int MIN_W = 240;
};
