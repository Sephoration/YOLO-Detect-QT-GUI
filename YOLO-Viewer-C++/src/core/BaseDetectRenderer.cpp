#include "BaseDetectRenderer.h"
#include "Config.h"
#include <opencv2/imgproc.hpp>

const cv::Scalar BaseDetectRenderer::TEXT_COLOR = {255, 255, 255};

BaseDetectRenderer::BaseDetectRenderer() = default;

// ── Main entry: single clone, then draw in-place ──

cv::Mat BaseDetectRenderer::render(const InferenceResult &result, const cv::Mat &rawImage)
{
    if (!result.success || rawImage.empty())
        return rawImage.clone();

    cv::Mat out = rawImage.clone();   // only clone once
    const auto &dt = result.dataType;

    if (dt == "detection")       renderDetection(out, result);
    else if (dt == "classification") renderClassification(out, result);
    else if (dt == "pose")       renderPose(out, result);
    else if (dt == "segmentation") renderSegmentation(out, result);

    drawOverlayStats(out, result);
    return out;
}

// ── Detection ──

void BaseDetectRenderer::renderDetection(cv::Mat &img, const InferenceResult &r)
{
    int W = img.cols, H = img.rows;
    for (const auto &d : r.detections) {
        int x1 = qBound(0, int(d.bbox.left()),   W-1);
        int y1 = qBound(0, int(d.bbox.top()),    H-1);
        int x2 = qBound(0, int(d.bbox.right()),  W-1);
        int y2 = qBound(0, int(d.bbox.bottom()), H-1);
        if (x2 <= x1 || y2 <= y1) continue;

        auto color = colorForClass(d.label, d.classId);
        cv::rectangle(img, {x1,y1}, {x2,y2}, color, m_lineWidth);

        if (m_showLabels) {
            QString txt;
            if (m_showConf)
                txt = QStringLiteral("%1 %2").arg(d.label).arg(d.confidence, 0, 'f', 2);
            else
                txt = d.label;
            if (m_showTrack && d.trackId >= 0)
                txt += QStringLiteral(" #%1").arg(d.trackId);
            drawLabel(img, txt, x1, y1, color);
        }
    }
}

// ── Classification ──

void BaseDetectRenderer::renderClassification(cv::Mat &img, const InferenceResult &r)
{
    int y = 30, step = 25;
    for (int i = 0; i < qMin(5, r.classifications.size()); ++i) {
        const auto &c = r.classifications[i];
        QString txt = QStringLiteral("%1: %2%").arg(c.className).arg(c.confidence * 100, 0, 'f', 1);
        auto color = (i == 0) ? cv::Scalar{0,255,0} : cv::Scalar{255,255,255};
        cv::putText(img, txt.toUtf8().constData(), {10,y}, cv::FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, color, FONT_THICKNESS, cv::LINE_AA);
        y += step;
    }
}

// ── Pose ──

void BaseDetectRenderer::renderPose(cv::Mat &img, const InferenceResult &r)
{
    int W = img.cols, H = img.rows;

    for (const auto &pose : r.poses) {
        int x1 = qBound(0, int(pose.bbox.left()),   W-1);
        int y1 = qBound(0, int(pose.bbox.top()),    H-1);
        int x2 = qBound(0, int(pose.bbox.right()),  W-1);
        int y2 = qBound(0, int(pose.bbox.bottom()), H-1);
        cv::rectangle(img, {x1,y1}, {x2,y2}, {0,255,255}, m_lineWidth);

        // Draw skeleton connections
        if (!m_skeleton.isEmpty())
            drawSkeleton(img, pose.keypoints, m_skeleton, {0,255,255}, 2);

        // Draw keypoints
        for (const auto &kp : pose.keypoints) {
            if (kp.confidence < 0.1) continue;
            int kx = qBound(0, int(kp.x), W-1);
            int ky = qBound(0, int(kp.y), H-1);
            cv::circle(img, {kx,ky}, 4, colorForIndex(kp.index), -1);
        }
    }
}

// ── Segmentation ──

void BaseDetectRenderer::renderSegmentation(cv::Mat &img, const InferenceResult &r)
{
    int W = img.cols, H = img.rows;
    for (const auto &s : r.segmentations) {
        int x1 = qBound(0, int(s.bbox.left()),   W-1);
        int y1 = qBound(0, int(s.bbox.top()),    H-1);
        int x2 = qBound(0, int(s.bbox.right()),  W-1);
        int y2 = qBound(0, int(s.bbox.bottom()), H-1);
        auto color = colorForClass(QString(), s.classId);
        cv::rectangle(img, {x1,y1}, {x2,y2}, color, m_lineWidth);
    }
}

// ── Stats overlay (draws in-place) ──

void BaseDetectRenderer::drawOverlayStats(cv::Mat &img, const InferenceResult &r)
{
    QStringList lines;
    if (r.detectionCount > 0) lines << QStringLiteral("检测: %1").arg(r.detectionCount);
    if (r.avgConfidence > 0)  lines << QStringLiteral("置信: %1").arg(r.avgConfidence, 0, 'f', 2);
    if (r.keypointCount > 0)  lines << QStringLiteral("关键点: %1").arg(r.keypointCount);
    if (r.inferenceTimeMs >0) lines << QStringLiteral("推理: %1 ms").arg(r.inferenceTimeMs, 0, 'f', 1);
    if (AppConfig::showFps() && r.fps > 0)
        lines << QStringLiteral("FPS: %1").arg(r.fps, 0, 'f', 1);
    if (lines.isEmpty()) return;

    int x = img.cols - 160, y = 30, step = 20;
    for (const auto &l : lines) {
        if (y >= img.rows) break;
        cv::putText(img, l.toUtf8().constData(), {x,y},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar{255,255,255}, 1, cv::LINE_AA);
        y += step;
    }
}

// ── Utilities ──

void BaseDetectRenderer::drawLabel(cv::Mat &img, const QString &text, int x, int y, const cv::Scalar &color)
{
    auto utf8 = text.toUtf8().constData();
    int baseline = 0;
    auto sz = cv::getTextSize(utf8, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS, &baseline);
    int bgY1 = qMax(0, y - sz.height - 5);
    int bgX2 = qMin(x + sz.width + 5, img.cols - 1);
    cv::rectangle(img, {x, bgY1}, {bgX2, y}, color, -1);
    int textY = (y - 3 > 0) ? (y - 3) : (y + sz.height);
    cv::putText(img, utf8, {x+2, textY}, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv::LINE_AA);
}

void BaseDetectRenderer::drawSkeleton(cv::Mat &img, const QVector<KeypointItem> &kps,
                                       const QVector<QPair<int,int>> &conn,
                                       const cv::Scalar &color, int thick)
{
    QMap<int,KeypointItem> map;
    for (const auto &kp : kps) map[kp.index] = kp;
    int W = img.cols, H = img.rows;
    for (const auto &c : conn) {
        if (!map.contains(c.first) || !map.contains(c.second)) continue;
        const auto &a = map[c.first], &b = map[c.second];
        if (a.confidence < 0.1 || b.confidence < 0.1) continue;
        int ax = qBound(0, int(a.x), W-1);
        int ay = qBound(0, int(a.y), H-1);
        int bx = qBound(0, int(b.x), W-1);
        int by = qBound(0, int(b.y), H-1);
        cv::line(img, {ax,ay}, {bx,by}, color, thick);
    }
}

cv::Scalar BaseDetectRenderer::colorForIndex(int idx)
{
    if (m_colorCache.contains(idx)) return m_colorCache[idx];
    int hue = (idx * 30) % 180;
    cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(hue,255,255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto v = bgr.at<cv::Vec3b>(0,0);
    return m_colorCache[idx] = cv::Scalar(v[0],v[1],v[2]);
}

cv::Scalar BaseDetectRenderer::colorForClass(const QString &name, int classId)
{
    Q_UNUSED(name)
    return colorForIndex(classId >= 0 ? classId : 0);
}
