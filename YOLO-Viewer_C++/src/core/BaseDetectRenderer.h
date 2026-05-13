#pragma once

#include "models/DetectionResult.h"
#include <opencv2/opencv.hpp>
#include <QColor>
#include <QMap>

class BaseDetectRenderer
{
public:
    BaseDetectRenderer();

    cv::Mat render(const InferenceResult &result, const cv::Mat &rawImage);

    void setLineWidth(int w) { m_lineWidth = w; }
    int  lineWidth() const   { return m_lineWidth; }

    void setShowLabels(bool on)    { m_showLabels = on; }
    void setShowConfidence(bool on){ m_showConf   = on; }
    void setShowTracking(bool on)  { m_showTrack  = on; }

private:
    cv::Mat renderDetection(const cv::Mat &img, const InferenceResult &r);
    cv::Mat renderClassification(const cv::Mat &img, const InferenceResult &r);
    cv::Mat renderPose(const cv::Mat &img, const InferenceResult &r);
    cv::Mat renderSegmentation(const cv::Mat &img, const InferenceResult &r);
    cv::Mat drawOverlayStats(const cv::Mat &img, const InferenceResult &r);
    void    drawLabel(cv::Mat &img, const QString &text, int x, int y, const cv::Scalar &color);
    void    drawSkeleton(cv::Mat &img, const QVector<KeypointItem> &kps,
                         const QVector<QPair<int,int>> &conn, const cv::Scalar &color, int thick);
    cv::Scalar colorForIndex(int idx);
    cv::Scalar colorForClass(const QString &name, int classId);

    QMap<int, cv::Scalar> m_colorCache;
    int  m_lineWidth   = 2;
    bool m_showLabels  = true;
    bool m_showConf    = true;
    bool m_showTrack   = true;

    static constexpr double FONT_SCALE    = 0.5;
    static constexpr int    FONT_THICKNESS = 1;
    static const cv::Scalar TEXT_COLOR;
};
