cmake_minimum_required(VERSION 3.15)
project(training_suite)
set(CMAKE_CXX_STANDARD 17)

add_executable(training_suite training_suite.cpp)

add_executable(training_suite_gui WIN32 training_suite_gui.cpp training_suite_gui.rc)
target_link_libraries(training_suite_gui PRIVATE comdlg32 shell32 shlwapi)
