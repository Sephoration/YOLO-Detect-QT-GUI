#include <QApplication>
#include "ui/MainWindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    a.setApplicationName("MusicPlayer");
    a.setOrganizationName("MusicPlayer");

    MainWindow w;
    w.show();

    return a.exec();
}
