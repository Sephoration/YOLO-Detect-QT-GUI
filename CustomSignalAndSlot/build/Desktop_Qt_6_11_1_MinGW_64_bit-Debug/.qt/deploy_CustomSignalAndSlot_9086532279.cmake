include("C:/Git_Code/something_Qt/CustomSignalAndSlot/build/Desktop_Qt_6_11_1_MinGW_64_bit-Debug/.qt/QtDeploySupport.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/CustomSignalAndSlot-plugins.cmake" OPTIONAL)
set(__QT_DEPLOY_I18N_CATALOGS "qtbase")

qt6_deploy_runtime_dependencies(
    EXECUTABLE "C:/Git_Code/something_Qt/CustomSignalAndSlot/build/Desktop_Qt_6_11_1_MinGW_64_bit-Debug/CustomSignalAndSlot.exe"
    GENERATE_QT_CONF
)
