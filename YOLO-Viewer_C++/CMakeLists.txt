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

# ── OpenCV 配置 ──
# 方法1: 手动指定 MinGW 编译的 OpenCV（优先）
if(MINGW AND EXISTS "C:/opencv-mingw/lib/cmake/opencv4")
    set(OpenCV_DIR "C:/opencv-mingw/lib/cmake/opencv4")
    find_package(OpenCV REQUIRED)
    message(STATUS "使用 MinGW 编译的 OpenCV: ${OpenCV_DIR}")
endif()

# 方法2: 尝试 pkg-config（scoop 的 opencv 可能支持）
if(NOT OpenCV_FOUND AND MINGW)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OpenCV REQUIRED opencv4)
        message(STATUS "通过 pkg-config 找到 OpenCV")
    endif()
endif()

# 方法3: 尝试标准的 find_package（MSVC 或其他）
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET)
    if(OpenCV_FOUND)
        message(STATUS "通过标准 find_package 找到 OpenCV: ${OpenCV_DIR}")
    endif()
endif()

# 方法4: 最后的应急方案 - 手动指定 scoop 的 MSVC OpenCV（不推荐，但可能工作）
if(NOT OpenCV_FOUND AND MINGW)
    message(WARNING "未找到 MinGW 兼容的 OpenCV，尝试使用 MSVC 版本（可能不稳定）")
    set(OpenCV_DIR "C:/Users/Eplis/scoop/apps/opencv/current/x64/vc16/lib/cmake/opencv4")
    set(OpenCV_INCLUDE_DIRS "C:/Users/Eplis/scoop/apps/opencv/current/include")
    set(OpenCV_LIB_DIR "C:/Users/Eplis/scoop/apps/opencv/current/x64/vc16/lib")
    
    # 手动设置库文件
    set(OpenCV_LIBS 
        opencv_world4120
        # 如果需要，可以添加更多依赖
    )
    set(OpenCV_FOUND TRUE)
    message(STATUS "手动配置 MSVC OpenCV: ${OpenCV_DIR}")
endif()

# 如果还是找不到，报错
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV 未找到。\n\n"
        "对于 MinGW 编译器，请使用以下方法之一安装 OpenCV:\n"
        "1. 使用 gocv 脚本编译 MinGW 版本:\n"
        "   git clone https://github.com/hybridgroup/gocv.git\n"
        "   cd gocv && .\\win_build_opencv.cmd\n"
        "   然后设置路径: -DOpenCV_DIR=C:/opencv/build/install/x64/mingw/lib/cmake/opencv4\n\n"
        "2. 使用 vcpkg 安装:\n"
        "   vcpkg install opencv:x64-mingw\n"
        "   vcpkg integrate install\n\n"
        "3. 临时方案: 安装 Visual Studio 并切换到 MSVC 编译器"
    )
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

# ── Copy Python bridge ──
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/python_bridge/yolo_service.py
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/yolo_service.py
    COMMENT "Copying Python YOLO service..."
)