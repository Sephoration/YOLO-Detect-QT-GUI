#pragma once
#include <QMap>
#include <QString>
#include <QList>

struct LyricLine {
    qint64 timeMs = 0;
    QString text;
};

class LyricParser {
public:
    bool loadFromFile(const QString &filePath);
    bool loadFromLrcFile(const QString &audioFilePath);

    QString currentLine(qint64 positionMs) const;
    int currentIndex(qint64 positionMs) const;
    const QList<LyricLine> &lines() const { return m_lines; }
    bool isValid() const { return !m_lines.isEmpty(); }
    void clear() { m_lines.clear(); }

private:
    QList<LyricLine> m_lines;
    static qint64 parseTime(const QString &timeStr);
};
