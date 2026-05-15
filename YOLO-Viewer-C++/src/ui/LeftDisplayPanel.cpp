#include "LeftDisplayPanel.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDateTime>

LeftDisplayPanel::LeftDisplayPanel(QWidget *parent)
    : QWidget(parent)
{
    setMinimumSize(MIN_PANEL_W, MIN_PANEL_H);
    initUi();
    setupStyle();
}

void LeftDisplayPanel::initUi()
{
    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0,0,0,0);
    mainLayout->setSpacing(0);

    // ── 显示区域 ──
    m_displayLabel = new AspectRatioDisplayLabel(this);
    m_displayLabel->setObjectName(QStringLiteral("DisplayLabel"));
    m_displayLabel->setText(tr("拖放图片/视频，或点击工具栏按钮"));
    mainLayout->addWidget(m_displayLabel, 1);

    // ── 底部控制条 ──
    auto *controlBar = new QWidget(this);
    controlBar->setObjectName(QStringLiteral("ControlBar"));
    controlBar->setFixedHeight(OVERLAY_HEIGHT);

    auto *cl = new QHBoxLayout(controlBar);
    cl->setContentsMargins(6,2,6,2);
    cl->setSpacing(4);

    m_playBtn = new QPushButton(tr("▶"), controlBar);
    m_playBtn->setFixedSize(28,28);
    m_playBtn->setObjectName(QStringLiteral("PlayBtn"));
    connect(m_playBtn, &QPushButton::clicked, this, &LeftDisplayPanel::playPauseClicked);

    m_progressSlider = new QSlider(Qt::Horizontal, controlBar);
    m_progressSlider->setRange(0, 1000);
    m_progressSlider->setValue(0);
    m_progressSlider->setObjectName(QStringLiteral("ProgressSlider"));
    connect(m_progressSlider, &QSlider::sliderMoved, this, [this](int val) {
        emit seekChanged(double(val) / 1000.0);
    });

    m_timeLabel = new QLabel(tr("00:00 / 00:00"), controlBar);
    m_timeLabel->setObjectName(QStringLiteral("TimeLabel"));

    m_detectBtn = new QPushButton(tr("检测:开"), controlBar);
    m_detectBtn->setCheckable(true);
    m_detectBtn->setChecked(true);
    m_detectBtn->setObjectName(QStringLiteral("DetectBtn"));
    connect(m_detectBtn, &QPushButton::toggled, this, [this](bool on) {
        m_detectBtn->setText(on ? tr("检测:开") : tr("检测:关"));
        emit detectToggled(on);
    });

    m_screenshotBtn = new QPushButton(tr("截图"), controlBar);
    m_screenshotBtn->setObjectName(QStringLiteral("ScreenshotBtn"));
    connect(m_screenshotBtn, &QPushButton::clicked, this, &LeftDisplayPanel::screenshotClicked);

    cl->addWidget(m_playBtn);
    cl->addWidget(m_progressSlider, 1);
    cl->addWidget(m_timeLabel);
    cl->addWidget(m_detectBtn);
    cl->addWidget(m_screenshotBtn);

    mainLayout->addWidget(controlBar);

    // ── 文件名称覆盖（在 displayLabel 之上） ──
    m_filenameLabel = new QLabel(m_displayLabel);
    m_filenameLabel->setObjectName(QStringLiteral("FilenameLabel"));
    m_filenameLabel->setFixedHeight(OVERLAY_HEIGHT);
    m_filenameLabel->move(10, 10);
    m_filenameLabel->hide();

    // ── 状态叠加文字 ──
    m_overlayLabel = new QLabel(m_displayLabel);
    m_overlayLabel->setObjectName(QStringLiteral("OverlayLabel"));
    m_overlayLabel->setAlignment(Qt::AlignCenter);
    m_overlayLabel->setWordWrap(true);
    m_overlayLabel->setGeometry(0,0, m_displayLabel->width(), m_displayLabel->height());
    m_overlayLabel->lower();
}

void LeftDisplayPanel::setupStyle()
{
    setStyleSheet(QStringLiteral(R"(
        #DisplayLabel {
            background-color: #1a1a1a; border: none; color: #888;
            font-size: 14px;
        }
        #ControlBar {
            background-color: #2d2d2d; border-top: 1px solid #444;
        }
        #PlayBtn, #DetectBtn, #ScreenshotBtn {
            background-color: #3a3a3a; color: #eee; border: 1px solid #555;
            border-radius: 3px; font-size: 11px; padding: 2px 6px;
        }
        #PlayBtn:hover, #DetectBtn:hover, #ScreenshotBtn:hover {
            background-color: #4a4a4a;
        }
        #DetectBtn:checked { background-color: #2a6d2a; border-color: #4a9a4a; }
        #ProgressSlider::groove:horizontal {
            height: 4px; background: #555; border-radius: 2px;
        }
        #ProgressSlider::handle:horizontal {
            background: #4a9eff; width: 10px; height: 10px;
            margin: -3px 0; border-radius: 5px;
        }
        #TimeLabel { color: #aaa; font-size: 10px; min-width: 80px; }
        #FilenameLabel {
            background: rgba(0,0,0,140); color: #eee; padding: 2px 8px;
            border-radius: 3px; font-size: 11px;
        }
        #OverlayLabel { color: #666; font-size: 13px; background: transparent; }
    )"));
}

void LeftDisplayPanel::setDisplayImage(const QPixmap &pixmap, int frameId)
{
    Q_UNUSED(frameId)
    if (!pixmap.isNull()) {
        m_displayLabel->setDisplayPixmap(pixmap);
        m_overlayLabel->hide();
    } else {
        m_displayLabel->clear();
        m_overlayLabel->show();
    }
}

void LeftDisplayPanel::clearDisplay()
{
    m_displayLabel->clear();
    m_displayLabel->setText(tr("等待显示..."));
    m_filenameLabel->hide();
    m_progressSlider->setValue(0);
    m_timeLabel->setText(tr("00:00 / 00:00"));
    m_overlayLabel->show();
}

void LeftDisplayPanel::updateInfo(const QString &fileName, const QString &mode)
{
    if (mode == QLatin1String("camera")) {
        m_filenameLabel->setText(tr("摄像头"));
        m_filenameLabel->show();
    } else if (!fileName.isEmpty()) {
        QString txt = fileName.length() > 35 ? "..." + fileName.right(32) : fileName;
        m_filenameLabel->setText(txt);
        m_filenameLabel->show();
    } else {
        m_filenameLabel->hide();
    }

    bool isVideo = (mode == QLatin1String("video"));
    m_playBtn->setVisible(isVideo);
    m_progressSlider->setVisible(isVideo);
    m_timeLabel->setVisible(isVideo);

    m_isPlaying = false;
    m_playBtn->setText(tr("▶"));
}

void LeftDisplayPanel::setPlayState(bool playing)
{
    m_isPlaying = playing;
    m_playBtn->setText(playing ? tr("⏸") : tr("▶"));
}

void LeftDisplayPanel::setProgress(int frame, int total)
{
    if (total <= 0) return;
    int val = qBound(0, int(double(frame) / total * 1000), 1000);
    m_progressSlider->setValue(val);

    if (m_durationSec > 0) {
        double elapsed = double(frame) / total * m_durationSec;
        int eSec = int(elapsed);
        int tSec = int(m_durationSec);
        m_timeLabel->setText(QStringLiteral("%1:%2 / %3:%4")
                             .arg(eSec/60, 2, 10, QLatin1Char('0'))
                             .arg(eSec%60, 2, 10, QLatin1Char('0'))
                             .arg(tSec/60, 2, 10, QLatin1Char('0'))
                             .arg(tSec%60, 2, 10, QLatin1Char('0')));
    } else {
        m_timeLabel->setText(QStringLiteral("%1/%2").arg(frame).arg(total));
    }
}

void LeftDisplayPanel::setDuration(double seconds)
{
    m_durationSec = seconds;
}

void LeftDisplayPanel::setDetectButtonState(bool enabled)
{
    m_detectBtn->setEnabled(enabled);
}

void LeftDisplayPanel::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    if (m_overlayLabel)
        m_overlayLabel->setGeometry(0, 0, m_displayLabel->width(), m_displayLabel->height());
    if (m_filenameLabel)
        m_filenameLabel->setFixedWidth(qMin(300, width() - 20));
}
