cmake_minimum_required (VERSION 3.1)

set (TARGET_NAME "nn")

if (UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(mkl REQUIRED)
ENDIF()

IF (WIN32)
    #set(MKLROOT $ENV{MKLROOT})
    set(MKLROOT "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.5.281\\windows\\mkl")
    set(MKL_INCLUDE_DIRS  ${MKLROOT}\\include)
    set(MKL_LIBRARIES_DIR ${MKLROOT}\\lib\\ia32)
    set(MKL_INTERFACE_LIBRARY ${MKLROOT}\\interfaces)
    set(MKL_LIBRARIES ${MKL_LIBRARIES_DIR}\\mkl_intel_c.lib ${MKL_LIBRARIES_DIR}\\mkl_tbb_thread.lib ${MKL_LIBRARIES_DIR}\\mkl_core.lib ${MKL_LIBRARIES_DIR}\\tbb.lib)
ENDIF()
    
file (GLOB MAIN_SRC
        ./*.cpp
        )

file (GLOB MAIN_HEADERS
        ./*.h
        )

add_executable(${TARGET_NAME} main.cpp ${MAIN_SRC} ${MAIN_HEADERS})
#add_library(${TARGET_NAME} STATIC ${MAIN_SRC} ${MAIN_HEADERS})
include_directories(${TARGET_NAME} ${MKL_INCLUDE_DIRS})
link_directories(${TARGET_NAME} ${MKL_LIBRARIES_DIR})
target_link_libraries(${TARGET_NAME} ${MKL_LIBRARIES})
target_link_libraries(${TARGET_NAME} csv)
