cmake_minimum_required (VERSION 3.1)

set (TARGET_NAME tests)

file (GLOB MAIN_SRC
        *.cpp
        )

file (GLOB MAIN_HEADERS
        *.h
        )
set(MKLROOT "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2019.5.281\\windows\\mkl")
set(MKL_INCLUDE_DIRS  ${MKLROOT}\\include)
set(MKL_LIBRARIES ${MKLROOT}\\lib\\ai32)
set(MKL_INTERFACE_LIBRARY ${MKLROOT}\\interfaces)

add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})
include_directories(${TARGET_NAME} ../nn ${MKL_INCLUDE_DIRS})
link_directories(${TARGET_NAME} ${MKL_LIBRARIES})
target_link_libraries(${TARGET_NAME} nn)
