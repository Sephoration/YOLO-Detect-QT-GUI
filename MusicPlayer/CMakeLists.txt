cmake_minimum_required(VERSION 3.20)
project(MusicPlayer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Multimedia Concurrent)

set(SOURCES
    src/main.cpp
    src/core/LyricParser.cpp
    src/core/PlaylistModel.cpp
    src/core/PlayerCore.cpp
    src/utils/SettingsManager.cpp
    src/utils/MusicScanner.cpp
    src/ui/AlbumCover.cpp
    src/ui/LyricView.cpp
    src/ui/VolumeButton.cpp
    src/ui/NavigationSidebar.cpp
    src/ui/PlaylistView.cpp
    src/ui/PlayerControls.cpp
    src/ui/MainWindow.cpp
)

set(HEADERS
    src/core/PlayMode.h
    src/core/MusicItem.h
    src/core/LyricParser.h
    src/core/PlaylistModel.h
    src/core/PlayerCore.h
    src/utils/SettingsManager.h
    src/utils/MusicScanner.h
    src/ui/AlbumCover.h
    src/ui/LyricView.h
    src/ui/VolumeButton.h
    src/ui/NavigationSidebar.h
    src/ui/PlaylistView.h
    src/ui/PlayerControls.h
    src/ui/MainWindow.h
)

add_executable(${PROJECT_NAME} WIN32 MACOSX_BUNDLE ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::Concurrent
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    MACOSX_BUNDLE TRUE
)
