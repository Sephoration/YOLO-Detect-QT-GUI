cmake_minimum_required(VERSION 3.18)
project(YOLOViewer VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ── 编译器检测 ──
if(MINGW)
    message(STATUS "检测到 MinGW 编译器，使用兼容配置")
    # 允许链接不兼容的库（临时方案）
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
endif()

# ── Qt6 ──
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Network
)

# ── OpenCV (conda 优先) ──
if(NOT OpenCV_FOUND)
    # 方法1: conda 环境 (推荐 — 激活 conda env 后编译，所有依赖来自 conda)
    if(DEFINED ENV{CONDA_PREFIX})
        set(_conda "$ENV{CONDA_PREFIX}")
        set(_conda_opencv "${_conda}/Library/lib/cmake/opencv4")
        if(EXISTS "${_conda_opencv}")
            set(OpenCV_DIR "${_conda_opencv}")
            find_package(OpenCV REQUIRED)
            message(STATUS "使用 conda OpenCV: ${OpenCV_DIR}")
        endif()
    endif()
endif()

if(NOT OpenCV_FOUND)
    # 方法2: 桌面预编译 OpenCV (MinGW)
    set(_desktop_opencv "C:/Users/Eplis/Desktop/opencv-mingw")
    if(EXISTS "${_desktop_opencv}/OpenCVConfig.cmake")
        set(OpenCV_DIR "${_desktop_opencv}")
        find_package(OpenCV REQUIRED)
        message(STATUS "使用桌面 OpenCV: ${OpenCV_DIR}")
    endif()
endif()

if(NOT OpenCV_FOUND)
    # 方法3: pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenCV REQUIRED opencv4)
        message(STATUS "通过 pkg-config 找到 OpenCV")
    endif()
endif()

if(NOT OpenCV_FOUND)
    # 方法3: 标准 CMake find_package
    find_package(OpenCV REQUIRED)
    message(STATUS "通过标准 find_package 找到 OpenCV: ${OpenCV_DIR}")
endif()

# ── 显示 OpenCV 配置信息 ──
message(STATUS "OpenCV 配置:")
message(STATUS "  - 版本: ${OpenCV_VERSION}")
message(STATUS "  - 包含目录: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  - 库文件: ${OpenCV_LIBS}")
if(MINGW)
    message(STATUS "  - 编译器: MinGW (${CMAKE_CXX_COMPILER})")
endif()

# ── Sources ──
set(SOURCES
    src/main.cpp
    src/Config.cpp
    src/models/DetectionResult.cpp
    src/ui/AspectRatioDisplayLabel.cpp
    src/ui/LeftDisplayPanel.cpp
    src/ui/RightControlPanel.cpp
    src/ui/InspectionPanel.cpp
    src/ui/MainWindow.cpp
    src/core/VideoPlayerThread.cpp
    src/core/DetectorWorker.cpp
    src/core/BaseDetectRenderer.cpp
    src/core/YoloBridge.cpp
    src/core/MainController.cpp
)

set(HEADERS
    src/utils/ImageUtils.h
    src/Config.h
    src/models/DetectionResult.h
    src/ui/AspectRatioDisplayLabel.h
    src/ui/LeftDisplayPanel.h
    src/ui/RightControlPanel.h
    src/ui/InspectionPanel.h
    src/ui/MainWindow.h
    src/core/VideoPlayerThread.h
    src/core/DetectorWorker.h
    src/core/BaseDetectRenderer.h
    src/core/YoloBridge.h
    src/core/MainController.h
)

# ── Resources ──
set(RESOURCES
    # resources/resources.qrc  (optional)
)

# ── Executable ──
add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
    ${RESOURCES}
)

# ── 链接库 ──
# Qt 库
target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Network
)

# OpenCV 库
if(TARGET PkgConfig::opencv4)
    target_link_libraries(${PROJECT_NAME} PRIVATE PkgConfig::opencv4)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif()

# MinGW 特殊处理：添加库路径
if(MINGW AND OpenCV_LIB_DIR)
    target_link_directories(${PROJECT_NAME} PRIVATE ${OpenCV_LIB_DIR})
endif()

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# ── Platform settings ──
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        WIN32_EXECUTABLE FALSE
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        _USE_MATH_DEFINES
        NOMINMAX
    )
endif()

# ── 构建提示 ──
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda 环境: $ENV{CONDA_PREFIX}")
else()
    message(NOTICE "提示: 激活 conda 环境后编译可获得更一致的依赖管理")
endif()

# ── Copy Python bridge ──
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/python_bridge/yolo_service.py
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/yolo_service.py
    COMMENT "Copying Python YOLO service..."
)