cmake_minimum_required(VERSION 3.19)
project(SignalAndSlot LANGUAGES CXX)

find_package(Qt6 6.5 REQUIRED COMPONENTS Core Widgets)

qt_standard_project_setup()

qt_add_executable(SignalAndSlot
    WIN32 MACOSX_BUNDLE
    main.cpp
    mywidget.cpp
    mywidget.h
    mywidget.ui
)

target_link_libraries(SignalAndSlot
    PRIVATE
        Qt::Core
        Qt::Widgets
)

install(TARGETS SignalAndSlot
    BUNDLE  DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

qt_generate_deploy_app_script(
    TARGET SignalAndSlot
    OUTPUT_SCRIPT deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)
install(SCRIPT ${deploy_script})
