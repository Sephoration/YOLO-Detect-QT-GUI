#pragma once

#include <QImage>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/// Convert cv::Mat (BGR) to QImage (RGB888).
inline QImage cvMatToQImage(const cv::Mat &mat)
{
    if (mat.empty()) return {};

    cv::Mat tmp;
    if (mat.channels() == 1)
        cv::cvtColor(mat, tmp, cv::COLOR_GRAY2BGR);
    else if (mat.channels() == 4)
        cv::cvtColor(mat, tmp, cv::COLOR_BGRA2BGR);
    else
        tmp = mat;

    cv::Mat cont = tmp.isContinuous() ? tmp : tmp.clone();
    return QImage(cont.data, cont.cols, cont.rows,
                  cont.cols * 3, QImage::Format_BGR888).copy();
}

/// Resize frame for inference — YOLO models typically need 640x.
/// Keeps aspect ratio, returns new Mat.
inline cv::Mat resizeFrameForInference(const cv::Mat &src, int maxSize = 640)
{
    if (src.empty()) return src;

    int w = src.cols, h = src.rows;
    double scale = double(maxSize) / qMax(w, h);
    if (scale >= 1.0) return src;

    cv::Mat dst;
    cv::resize(src, dst, cv::Size(int(w * scale), int(h * scale)),
               0, 0, cv::INTER_LINEAR);
    return dst;
}
