#include "LyricParser.h"
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QDir>
#include <QFileInfo>

bool LyricParser::loadFromFile(const QString &filePath) {
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return false;

    m_lines.clear();
    QTextStream stream(&file);
    stream.setAutoDetectUnicode(true);

    QRegularExpression re(R"(\[(\d{2}):(\d{2})\.(\d{2,3})\](.*))");

    while (!stream.atEnd()) {
        QString line = stream.readLine().trimmed();
        if (line.isEmpty()) continue;

        QRegularExpressionMatchIterator it = re.globalMatch(line);
        while (it.hasNext()) {
            QRegularExpressionMatch match = it.next();
            qint64 ms = parseTime(match.captured(0).mid(1, match.captured(0).indexOf(']') - 1));
            QString text = match.captured(4).trimmed();
            if (!text.isEmpty()) {
                m_lines.append({ms, text});
            }
        }
    }

    std::sort(m_lines.begin(), m_lines.end(), [](const LyricLine &a, const LyricLine &b) {
        return a.timeMs < b.timeMs;
    });

    return !m_lines.isEmpty();
}

bool LyricParser::loadFromLrcFile(const QString &audioFilePath) {
    QFileInfo info(audioFilePath);
    QString lrcPath = info.path() + QDir::separator() + info.completeBaseName() + ".lrc";
    if (QFile::exists(lrcPath))
        return loadFromFile(lrcPath);

    // Try same folder with any .lrc
    QDir dir(info.path());
    QStringList lrcFiles = dir.entryList(QStringList() << "*.lrc", QDir::Files);
    if (!lrcFiles.isEmpty())
        return loadFromFile(dir.absoluteFilePath(lrcFiles.first()));

    return false;
}

QString LyricParser::currentLine(qint64 positionMs) const {
    int idx = currentIndex(positionMs);
    return idx >= 0 ? m_lines[idx].text : QString();
}

int LyricParser::currentIndex(qint64 positionMs) const {
    if (m_lines.isEmpty()) return -1;

    int left = 0, right = m_lines.size() - 1;
    while (left < right) {
        int mid = (left + right + 1) / 2;
        if (m_lines[mid].timeMs <= positionMs)
            left = mid;
        else
            right = mid - 1;
    }
    return left;
}

qint64 LyricParser::parseTime(const QString &timeStr) {
    QRegularExpression re(R"((\d{2}):(\d{2})\.(\d{2,3}))");
    QRegularExpressionMatch match = re.match(timeStr);
    if (!match.hasMatch()) return 0;

    int min = match.captured(1).toInt();
    int sec = match.captured(2).toInt();
    QString msStr = match.captured(3);
    int ms = msStr.toInt();
    if (msStr.length() == 2) ms *= 10;

    return min * 60000LL + sec * 1000LL + ms;
}
