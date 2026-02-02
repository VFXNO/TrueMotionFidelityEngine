#pragma once

#include <stdint.h>
#include <windows.h>
#include <dxgi.h>

// Shared memory names
#define SHMEM_HOOK_INFO      L"FrameGenHookInfo"
#define SHMEM_TEXTURE        L"FrameGenTexture"

// Event names
#define EVENT_CAPTURE_RESTART L"FrameGenCaptureRestart"
#define EVENT_CAPTURE_STOP    L"FrameGenCaptureStop"
#define EVENT_HOOK_READY      L"FrameGenHookReady"
#define EVENT_HOOK_EXIT       L"FrameGenHookExit"
#define EVENT_HOOK_INIT       L"FrameGenHookInit"

// Mutex names
#define MUTEX_TEXTURE1        L"FrameGenTexture1"
#define MUTEX_TEXTURE2        L"FrameGenTexture2"
#define WINDOW_HOOK_KEEPALIVE L"FrameGenKeepAlive"

// Pipe name for logging
#define PIPE_NAME             "FrameGenPipe"

// Number of texture buffers for double/triple buffering
#define NUM_BUFFERS 2

// Capture type
enum capture_type {
    CAPTURE_TYPE_MEMORY,   // Shared memory (slower but compatible)
    CAPTURE_TYPE_TEXTURE   // Shared texture (faster, GPU-GPU transfer)
};

// Graphics offsets for hooking (filled by offset finder)
struct graphics_offsets {
    uint32_t d3d11_present;
    uint32_t d3d11_resize;
    uint32_t d3d12_present;
    uint32_t d3d12_resize;
    uint32_t d3d12_execute_command_lists;
    uint32_t dxgi_present;
    uint32_t dxgi_resize;
};

// Hook info shared between injector and hook DLL
struct hook_info {
    // Capture dimensions
    uint32_t cx;
    uint32_t cy;
    uint32_t pitch;
    
    // Frame format
    DXGI_FORMAT format;
    
    // Capture type
    enum capture_type type;
    
    // Shared texture handle (for CAPTURE_TYPE_TEXTURE)
    uint64_t tex_handle;
    
    // Current texture index (for double buffering)
    volatile uint32_t cur_tex;
    
    // Frame number
    volatile uint64_t frame_count;
    
    // Timing
    volatile int64_t frame_time;
    
    // Configuration
    uint32_t force_shmem : 1;
    uint32_t capture_overlay : 1;
    uint32_t allow_srgb_alias : 1;
    uint32_t flip : 1;
    
    // Hook version
    uint32_t hook_ver_major;
    uint32_t hook_ver_minor;
    
    // Target window (for multi-window games)
    uint64_t window;
    
    // Map ID for shared memory
    uint32_t map_id;
    
    // Frame interval limit (0 = unlimited)
    uint64_t frame_interval;
    
    // Graphics API offsets
    struct graphics_offsets offsets;
};

// Shared memory texture data
struct shmem_data {
    volatile int last_tex;
    uint32_t tex1_offset;
    uint32_t tex2_offset;
};

// Shared texture data
struct shtex_data {
    uint64_t tex_handle;
};

// Hook version
#define HOOK_VER_MAJOR 1
#define HOOK_VER_MINOR 0

// Mapping flags
#define GC_MAPPING_FLAGS (FILE_MAP_READ | FILE_MAP_WRITE)
