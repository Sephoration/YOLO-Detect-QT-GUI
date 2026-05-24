#ifndef MUSICPAGE_H
#define MUSICPAGE_H

#include <QWidget>

class QSlider;
class QLabel;
class QPushButton;

class MusicPage : public QWidget {
    Q_OBJECT
public:
    explicit MusicPage(QWidget *parent = nullptr);

private slots:
    void togglePlay();
    void onSliderMoved(int val);

private:
    void setupUI();
    QSlider *m_progress = nullptr;
    QPushButton *m_playBtn = nullptr;
    QLabel *m_timeCurrent = nullptr;
    QLabel *m_timeTotal = nullptr;
    bool m_playing = false;
};

#endif // MUSICPAGE_H
