include("C:/Git_Code/something_Qt/SignalAndSlot/build/Desktop_Qt_6_11_1_MinGW_64_bit-Debug/.qt/QtDeploySupport.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/SignalAndSlot-plugins.cmake" OPTIONAL)
set(__QT_DEPLOY_I18N_CATALOGS "qtbase")

qt6_deploy_runtime_dependencies(
    EXECUTABLE "C:/Git_Code/something_Qt/SignalAndSlot/build/Desktop_Qt_6_11_1_MinGW_64_bit-Debug/SignalAndSlot.exe"
    GENERATE_QT_CONF
)
