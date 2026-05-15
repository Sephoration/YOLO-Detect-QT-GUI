#pragma once

#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include "AspectRatioDisplayLabel.h"

/**
 * @brief 左侧显示面板
 *
 * 包含：
 * - 图像/视频显示区（保持比例）
 * - 文件名覆盖
 * - 播放控制条（进度条 + 播放/暂停 + 检测开关）
 */
class LeftDisplayPanel : public QWidget
{
    Q_OBJECT
public:
    explicit LeftDisplayPanel(QWidget *parent = nullptr);

    void setDisplayImage(const QPixmap &pixmap, int frameId = -1);
    void clearDisplay();
    void updateInfo(const QString &fileName, const QString &mode);
    void setPlayState(bool playing);
    void setProgress(int frame, int total);
    void setDuration(double seconds);
    void setDetectButtonState(bool enabled);

    AspectRatioDisplayLabel* displayLabel() const { return m_displayLabel; }

signals:
    void playPauseClicked();
    void stopClicked();
    void seekChanged(double ratio);
    void detectToggled(bool on);
    void screenshotClicked();

protected:
    void resizeEvent(QResizeEvent *event) override;

private:
    void initUi();
    void setupStyle();

    AspectRatioDisplayLabel *m_displayLabel   = nullptr;
    QLabel   *m_filenameLabel    = nullptr;
    QLabel   *m_overlayLabel     = nullptr;
    QSlider  *m_progressSlider   = nullptr;
    QLabel   *m_timeLabel        = nullptr;
    QPushButton *m_playBtn       = nullptr;
    QPushButton *m_detectBtn     = nullptr;
    QPushButton *m_screenshotBtn = nullptr;

    bool m_isPlaying  = false;
    bool m_detectOn   = true;
    double m_durationSec = 0.0;

    static constexpr int OVERLAY_HEIGHT = 32;
    static constexpr int MIN_PANEL_W    = 640;
    static constexpr int MIN_PANEL_H    = 480;
};
