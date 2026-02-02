cmake_minimum_required(VERSION 3.25)

project(TrueMotionFidelityEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.4
)

FetchContent_MakeAvailable(imgui)

# Graphics hook DLL (injected into game process)
add_library(graphics_hook SHARED
  src/graphics_hook/graphics_hook.cpp
  src/graphics_hook_info.h
)

target_compile_definitions(graphics_hook PRIVATE
  NOMINMAX
  _UNICODE
  UNICODE
)

target_link_libraries(graphics_hook PRIVATE
  d3d11
  dxgi
)

# Set output name based on architecture
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set_target_properties(graphics_hook PROPERTIES OUTPUT_NAME "graphics_hook64_v2")
else()
  set_target_properties(graphics_hook PROPERTIES OUTPUT_NAME "graphics_hook32_v2")
endif()

# Copy hook DLL to main executable directory
add_custom_command(TARGET graphics_hook POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  "$<TARGET_FILE:graphics_hook>"
  "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/"
)

# Main application
add_executable(TrueMotionFidelityEngine WIN32
  src/app.cpp
  src/app.h
  src/capture_frame.h
  src/d3d11_device.cpp
  src/d3d11_device.h
  src/dll_injector.cpp
  src/dll_injector.h
  src/dup_capture.cpp
  src/dup_capture.h
  src/game_capture.cpp
  src/game_capture.h
  src/graphics_hook_info.h
  src/interpolator.cpp
  src/interpolator.h
  src/main.cpp
  src/shader_utils.cpp
  src/shader_utils.h
  src/ui.cpp
  src/ui.h
  src/wgc_capture.cpp
  src/wgc_capture.h
  src/window_list.cpp
  src/window_list.h
  src/resources.rc
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp
)

target_include_directories(TrueMotionFidelityEngine PRIVATE
  src
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)

target_link_libraries(TrueMotionFidelityEngine PRIVATE
  d3d11
  dxgi
  d3dcompiler
  dwmapi
  windowsapp
  winmm
  shcore
  shlwapi
)

set_target_properties(TrueMotionFidelityEngine PROPERTIES OUTPUT_NAME "True Motion Fidelity Engine")

# Make FrameGen depend on graphics_hook so hook is built first
add_dependencies(TrueMotionFidelityEngine graphics_hook)

target_compile_definitions(TrueMotionFidelityEngine PRIVATE
  WINRT_NO_COROUTINES
  NOMINMAX
)

# Find Windows SDK fxc.exe shader compiler
find_program(FXC_EXECUTABLE fxc
  PATHS
    "$ENV{WindowsSdkDir}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
    "$ENV{WindowsSdkDir}/bin/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64"
    "C:/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64"
)

set(SHADER_OUTPUT_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/shaders")

# Create shader output directory and copy source files
add_custom_command(TARGET TrueMotionFidelityEngine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/src/shaders" "${SHADER_OUTPUT_DIR}"
  COMMENT "Copying shader source files"
)

if(FXC_EXECUTABLE)
  message(STATUS "Found fxc: ${FXC_EXECUTABLE} - shaders will be pre-compiled")
  
  # Compile each shader at build time
  # Use /O1 (less aggressive optimization) to avoid timeouts on complex shaders
  set(SHADER_NAMES Blend CopyScale DebugView DownsampleLuma DownsampleLumaR Interpolate MotionEst MotionRefine MotionSmooth MotionTemporal)
  
  foreach(SHADER_NAME ${SHADER_NAMES})
    add_custom_command(TARGET TrueMotionFidelityEngine POST_BUILD
      COMMAND ${FXC_EXECUTABLE} /T cs_5_0 /E CSMain /O1 /nologo /Fo "${SHADER_OUTPUT_DIR}/${SHADER_NAME}.cso" "${CMAKE_SOURCE_DIR}/src/shaders/${SHADER_NAME}.hlsl" || echo "Warning: ${SHADER_NAME} compilation failed, will use runtime compilation"
      COMMENT "Compiling ${SHADER_NAME}.hlsl"
    )
  endforeach()
else()
  message(WARNING "fxc.exe not found - shaders will be compiled at runtime (slower startup)")
endif()
