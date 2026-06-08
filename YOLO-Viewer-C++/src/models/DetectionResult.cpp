#include "DetectionResult.h"
#include <QJsonArray>

// ============================================================
//  InferenceResult
// ============================================================
InferenceResult InferenceResult::fromJson(const QJsonObject &obj)
{
    InferenceResult r;
    r.success       = obj.value("success").toBool(false);
    r.error         = obj.value("error").toString();
    r.dataType      = obj.value("data_type").toString();
    r.timestampMs   = QDateTime::currentMSecsSinceEpoch();

    // --- stats ---
    QJsonObject s  = obj.value("stats").toObject();
    r.detectionCount = s.value("detection_count").toInt(0);
    r.avgConfidence  = s.value("avg_confidence").toDouble(0.0);
    r.inferenceTimeMs= s.value("inference_time").toDouble(0.0);
    r.fps            = s.value("fps").toDouble(0.0);
    r.keypointCount  = s.value("keypoint_count").toInt(0);
    r.numPeople      = s.value("num_people").toInt(0);

    // --- processed_data ---
    QJsonObject pd   = obj.value("processed_data").toObject();

    if (r.dataType == "detection") {
        QJsonObject d   = pd.value("detection").toObject();
        QJsonArray  box = d.value("boxes").toArray();
        QJsonArray  lab = d.value("labels").toArray();
        QJsonArray  cnf = d.value("confidences").toArray();
        QJsonArray  cid = d.value("class_ids").toArray();
        QJsonArray  tid = d.value("track_ids").toArray();
        for (int i = 0; i < box.size(); ++i) {
            QJsonArray b = box[i].toArray();
            if (b.size() >= 4) {
                DetectionItem item;
                item.bbox      = QRectF(b[0].toDouble(), b[1].toDouble(),
                                        b[2].toDouble() - b[0].toDouble(),
                                        b[3].toDouble() - b[1].toDouble());
                item.label      = lab.at(i).toString(QStringLiteral("obj_%1").arg(i));
                item.confidence = cnf.at(i).toDouble(0.0);
                item.classId    = cid.at(i).toInt(i);
                item.trackId    = tid.at(i).toInt(-1);
                r.detections.append(item);
            }
        }
    }
    else if (r.dataType == "classification") {
        QJsonObject c   = pd.value("classification").toObject();
        QJsonArray  pr  = c.value("top_predictions").toArray();
        for (const auto &v : pr) {
            QJsonArray p = v.toArray();
            if (p.size() >= 2) {
                ClassificationItem ci;
                ci.className  = p[0].toString();
                ci.confidence = p[1].toDouble();
                r.classifications.append(ci);
            }
        }
    }
    else if (r.dataType == "pose") {
        QJsonObject p   = pd.value("pose").toObject();
        QJsonArray  box = p.value("boxes").toArray();
        QJsonArray  kps = p.value("keypoints").toArray();
        QJsonArray  kpc = p.value("keypoints_conf").toArray();
        QJsonArray  cnf = p.value("confidences").toArray();
        for (int i = 0; i < box.size(); ++i) {
            QJsonArray b = box[i].toArray();
            if (b.size() >= 4) {
                PoseItem pi;
                pi.bbox       = QRectF(b[0].toDouble(), b[1].toDouble(),
                                       b[2].toDouble() - b[0].toDouble(),
                                       b[3].toDouble() - b[1].toDouble());
                pi.confidence = cnf.at(i).toDouble(0.0);
                if (i < kps.size()) {
                    QJsonArray kpArr = kps[i].toArray();
                    QJsonArray kpCf  = kpc.at(i).toArray();
                    for (int j = 0; j < kpArr.size(); ++j) {
                        QJsonArray k = kpArr[j].toArray();
                        if (k.size() >= 2) {
                            KeypointItem ki;
                            ki.x          = k[0].toDouble();
                            ki.y          = k[1].toDouble();
                            ki.confidence = kpCf.at(j).toDouble(1.0);
                            ki.index      = j;
                            pi.keypoints.append(ki);
                        }
                    }
                }
                r.poses.append(pi);
            }
        }
    }
    else if (r.dataType == "segmentation") {
        QJsonObject s   = pd.value("segmentation").toObject();
        QJsonArray  box = s.value("boxes").toArray();
        QJsonArray  cid = s.value("class_ids").toArray();
        QJsonArray  cnf = s.value("confidences").toArray();
        for (int i = 0; i < box.size(); ++i) {
            QJsonArray b = box[i].toArray();
            if (b.size() >= 4) {
                SegmentationItem si;
                si.bbox       = QRectF(b[0].toDouble(), b[1].toDouble(),
                                       b[2].toDouble() - b[0].toDouble(),
                                       b[3].toDouble() - b[1].toDouble());
                si.classId    = cid.at(i).toInt(0);
                si.confidence = cnf.at(i).toDouble(0.0);
                r.segmentations.append(si);
            }
        }
    }

    return r;
}

QJsonObject InferenceResult::toJson() const
{
    QJsonObject obj;
    obj["success"]    = success;
    obj["error"]      = error;
    obj["data_type"]  = dataType;

    QJsonObject st;
    st["detection_count"] = detectionCount;
    st["avg_confidence"]  = avgConfidence;
    st["inference_time"]  = inferenceTimeMs;
    st["fps"]             = fps;
    st["keypoint_count"]  = keypointCount;
    st["num_people"]      = numPeople;
    obj["stats"]          = st;

    return obj;
}

// ============================================================
//  ModelInfo
// ============================================================
ModelInfo ModelInfo::fromJson(const QJsonObject &obj)
{
    ModelInfo mi;
    mi.modelPath   = obj.value("model_path").toString();
    mi.taskType    = obj.value("task_type").toString();
    mi.inputSize   = obj.value("input_size").toVariant().toString();
    mi.classCount  = obj.value("class_count").toInt(0);

    QJsonArray na  = obj.value("class_names").toArray();
    for (const auto &v : na) mi.classNames.append(v.toString());

    mi.numKeypoints = obj.value("num_keypoints").toInt(0);
    QJsonArray sk   = obj.value("skeleton").toArray();
    for (const auto &v : sk) {
        QJsonArray a = v.toArray();
        if (a.size() >= 2)
            mi.skeletonConnections.append({a[0].toInt(), a[1].toInt()});
    }
    return mi;
}
