cmake_minimum_required(VERSION 3.16)
project(CarHMI VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

set(SOURCES
    main.cpp
    mainwindow.cpp
    dashboard/dashboardwidget.cpp
    dashboard/gauge.cpp
    dashboard/gearpanel.cpp
    center/centerconsole.cpp
    center/homepage.cpp
    center/musicpage.cpp
    center/videopage.cpp
    center/acpage.cpp
    center/vehiclepage.cpp
    center/navpage.cpp
    center/settingspage.cpp
    common/styles.cpp
)

set(HEADERS
    mainwindow.h
    dashboard/dashboardwidget.h
    dashboard/gauge.h
    dashboard/gearpanel.h
    center/centerconsole.h
    center/homepage.h
    center/musicpage.h
    center/videopage.h
    center/acpage.h
    center/vehiclepage.h
    center/navpage.h
    center/settingspage.h
    common/styles.h
    common/constants.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE TRUE
)
