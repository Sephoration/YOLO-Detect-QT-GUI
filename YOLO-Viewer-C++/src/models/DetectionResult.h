#pragma once

#include <QString>
#include <QVector>
#include <QRectF>
#include <QMap>
#include <QVariant>
#include <QJsonObject>
#include <QJsonArray>
#include <QImage>
#include <QDateTime>
#include <opencv2/core.hpp>

// ============================================================
//  Detection
// ============================================================
struct DetectionItem
{
    QRectF  bbox;
    QString label;
    double  confidence = 0.0;
    int     classId    = 0;
    int     trackId    = -1;          // tracking ID (if tracker enabled)
};

// ============================================================
//  Classification
// ============================================================
struct ClassificationItem
{
    QString className;
    double  confidence = 0.0;
};

// ============================================================
//  Keypoint / Pose
// ============================================================
struct KeypointItem
{
    double  x = 0.0, y = 0.0;
    double  confidence = 0.0;
    int     index      = 0;
};

struct PoseItem
{
    QRectF  bbox;
    double  confidence = 0.0;
    QVector<KeypointItem> keypoints;
};

// ============================================================
//  Segmentation
// ============================================================
struct SegmentationItem
{
    QRectF  bbox;
    int     classId    = 0;
    double  confidence = 0.0;
    cv::Mat mask;                     // binary mask if available
};

// ============================================================
//  Unified inference result
// ============================================================
struct InferenceResult
{
    bool    success = false;
    QString error;
    QString dataType;   // detection | classification | pose | segmentation | obb

    QVector<DetectionItem>       detections;
    QVector<ClassificationItem>  classifications;
    QVector<PoseItem>            poses;
    QVector<SegmentationItem>    segmentations;

    // Aggregated stats
    int    detectionCount   = 0;
    double avgConfidence    = 0.0;
    double inferenceTimeMs  = 0.0;
    double fps              = 0.0;
    int    keypointCount    = 0;
    int    numPeople        = 0;

    // Model metadata  (pass-through)
    QMap<QString, QVariant> modelInfo;

    // Timestamp
    qint64 timestampMs = 0;

    static InferenceResult fromJson(const QJsonObject &obj);
    QJsonObject toJson() const;
};

// ============================================================
//  Model information
// ============================================================
struct ModelInfo
{
    QString   modelPath;
    QString   taskType;        // detection | classification | pose | segmentation
    QString   inputSize;
    int       classCount    = 0;
    QStringList classNames;
    int       numKeypoints  = 0;
    QVector<QPair<int,int>> skeletonConnections;

    static ModelInfo fromJson(const QJsonObject &obj);
};

// ============================================================
//  Tracking
// ============================================================
struct TrackedObject
{
    int       trackId;
    QString   label;
    QRectF    bbox;
    double    confidence;
    int       classId;
    int       lostFrames = 0;        // consecutive frames where this ID was missed
    qint64    firstSeenMs;
    qint64    lastSeenMs;
};
