#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <intrin.h>
#include <stdio.h>
#include <algorithm>
#include <wrl/client.h>

#include "../graphics_hook_info.h"

using Microsoft::WRL::ComPtr;

// MinHook for function hooking
// We'll use manual hooking via VTable for simplicity
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Globals
static HMODULE g_hModule = nullptr;
static HANDLE g_hHookReadyEvent = nullptr;
static HANDLE g_hHookStopEvent = nullptr;
static HANDLE g_hHookExitEvent = nullptr;
static HANDLE g_hKeepAliveMutex = nullptr;
static HANDLE g_hTextureMutex[NUM_BUFFERS] = {nullptr};
static HANDLE g_hHookInfoMap = nullptr;
static HANDLE g_hTextureMap = nullptr;

static hook_info* g_hookInfo = nullptr;
static shmem_data* g_shmemData = nullptr;
static shtex_data* g_shtexData = nullptr;

static ComPtr<ID3D11Device> g_device;
static ComPtr<ID3D11DeviceContext> g_context;
static ComPtr<IDXGISwapChain> g_swapChain; // Use ComPtr, but be careful about ownership

static ComPtr<ID3D11Texture2D> g_captureTexture;
static ComPtr<ID3D11Texture2D> g_stagingTextures[NUM_BUFFERS];
static HANDLE g_sharedHandle = nullptr;

static uint32_t g_cx = 0;
static uint32_t g_cy = 0;
static DXGI_FORMAT g_format = DXGI_FORMAT_UNKNOWN;
static bool g_useSharedTexture = false;
static bool g_initialized = false;
static bool g_active = false;

static DWORD g_processId = 0;

// Original function pointers
typedef HRESULT(STDMETHODCALLTYPE* PFN_Present)(IDXGISwapChain*, UINT, UINT);
typedef HRESULT(STDMETHODCALLTYPE* PFN_Present1)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
typedef HRESULT(STDMETHODCALLTYPE* PFN_ResizeBuffers)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);

static PFN_Present g_origPresent = nullptr;
static PFN_Present1 g_origPresent1 = nullptr;
static PFN_ResizeBuffers g_origResizeBuffers = nullptr;

// VTable indices
#define DXGI_SWAPCHAIN_PRESENT_INDEX 8
#define DXGI_SWAPCHAIN_RESIZEBUFFERS_INDEX 13
#define DXGI_SWAPCHAIN1_PRESENT1_INDEX 22

// Forward declarations
static void CaptureFrame();
static bool InitCapture();
static void FreeCapture();
static void HookSwapChain(IDXGISwapChain* swapChain);

// File-based logging for debugging
static void LogToFile(const char* fmt, ...) {
    FILE* f = fopen("C:\\hook_debug.txt", "a");
    if (f) {
        char buf[1024];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        fprintf(f, "[PID %lu] %s", g_processId, buf);
        fclose(f);
    }
}

// Logging
static void Log(const char* fmt, ...) {
    char buf[1024];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    OutputDebugStringA(buf);
    LogToFile("%s", buf);  // Also write to file
}

// Create named objects with process ID suffix
static HANDLE CreateNamedEvent(const wchar_t* baseName) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, g_processId);
    return CreateEventW(nullptr, FALSE, FALSE, name);
}

static HANDLE OpenNamedEvent(const wchar_t* baseName) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, g_processId);
    return OpenEventW(EVENT_ALL_ACCESS, FALSE, name);
}

static HANDLE CreateNamedMutex(const wchar_t* baseName) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, g_processId);
    return CreateMutexW(nullptr, FALSE, name);
}

static HANDLE CreateNamedFileMapping(const wchar_t* baseName, DWORD size) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, g_processId);
    return CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, size, name);
}

// Initialize shared memory
static bool InitSharedMemory() {
    g_hHookInfoMap = CreateNamedFileMapping(SHMEM_HOOK_INFO, sizeof(hook_info));
    if (!g_hHookInfoMap) {
        Log("[Hook] Failed to create hook info mapping\n");
        return false;
    }
    
    g_hookInfo = (hook_info*)MapViewOfFile(g_hHookInfoMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info));
    if (!g_hookInfo) {
        Log("[Hook] Failed to map hook info\n");
        return false;
    }
    
    memset(g_hookInfo, 0, sizeof(hook_info));
    g_hookInfo->hook_ver_major = HOOK_VER_MAJOR;
    g_hookInfo->hook_ver_minor = HOOK_VER_MINOR;
    
    return true;
}

// Initialize capture textures
static bool InitCaptureTextures() {
    if (!g_device || !g_context) return false;
    
    // Try to use shared texture first (faster)
    g_useSharedTexture = !g_hookInfo->force_shmem;
    
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = g_cx;
    desc.Height = g_cy;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = g_format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    if (g_useSharedTexture) {
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        
        // ComPtr automatically releases old resource if any before assignment, but here we prefer GetAddressOf()
        HRESULT hr = g_device->CreateTexture2D(&desc, nullptr, g_captureTexture.ReleaseAndGetAddressOf());
        if (FAILED(hr)) {
            Log("[Hook] Failed to create shared texture, falling back to shmem\n");
            g_useSharedTexture = false;
        } else {
            // Get shared handle
            ComPtr<IDXGIResource> dxgiRes;
            hr = g_captureTexture.As(&dxgiRes); // Helper for QueryInterface
            if (SUCCEEDED(hr)) {
                hr = dxgiRes->GetSharedHandle(&g_sharedHandle);
                
                if (SUCCEEDED(hr)) {
                    Log("[Hook] Created shared texture with handle %p\n", g_sharedHandle);
                    
                    // Setup shtex_data
                    DWORD mapSize = sizeof(shtex_data);
                    g_hTextureMap = CreateNamedFileMapping(SHMEM_TEXTURE, mapSize);
                    if (g_hTextureMap) {
                        g_shtexData = (shtex_data*)MapViewOfFile(g_hTextureMap, FILE_MAP_ALL_ACCESS, 0, 0, mapSize);
                        if (g_shtexData) {
                            g_shtexData->tex_handle = (uint64_t)g_sharedHandle;
                            g_hookInfo->tex_handle = (uint64_t)g_sharedHandle;
                            g_hookInfo->type = CAPTURE_TYPE_TEXTURE;
                            return true;
                        }
                    }
                }
            }
            
            // Failed to get shared handle or map memory
            g_captureTexture.Reset();
            g_useSharedTexture = false;
        }
    }
    
    // Fallback to shared memory (CPU copy)
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        HRESULT hr = g_device->CreateTexture2D(&desc, nullptr, g_stagingTextures[i].ReleaseAndGetAddressOf());
        if (FAILED(hr)) {
            Log("[Hook] Failed to create staging texture %d\n", i);
            return false;
        }
    }
    
    // Calculate pitch
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = g_context->Map(g_stagingTextures[0].Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        g_hookInfo->pitch = mapped.RowPitch;
        g_context->Unmap(g_stagingTextures[0].Get(), 0);
    } else {
        // Estimate pitch
        g_hookInfo->pitch = g_cx * 4;
    }
    
    // Create shared memory for pixel data
    DWORD dataSize = g_hookInfo->pitch * g_cy;
    DWORD mapSize = sizeof(shmem_data) + (dataSize * NUM_BUFFERS);
    
    g_hTextureMap = CreateNamedFileMapping(SHMEM_TEXTURE, mapSize);
    if (!g_hTextureMap) {
        Log("[Hook] Failed to create texture mapping\n");
        return false;
    }
    
    g_shmemData = (shmem_data*)MapViewOfFile(g_hTextureMap, FILE_MAP_ALL_ACCESS, 0, 0, mapSize);
    if (!g_shmemData) {
        Log("[Hook] Failed to map texture data\n");
        return false;
    }
    
    g_shmemData->last_tex = -1;
    g_shmemData->tex1_offset = sizeof(shmem_data);
    g_shmemData->tex2_offset = sizeof(shmem_data) + dataSize;
    
    g_hookInfo->type = CAPTURE_TYPE_MEMORY;
    g_hookInfo->map_id = g_processId;
    
    Log("[Hook] Using shared memory capture\n");
    return true;
}

// Initialize capture
static bool InitCapture() {
    if (!g_swapChain || !g_device) return false;
    
    // Get backbuffer description
    ComPtr<ID3D11Texture2D> backBuffer;
    HRESULT hr = g_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    if (FAILED(hr)) {
        Log("[Hook] Failed to get backbuffer\n");
        return false;
    }
    
    D3D11_TEXTURE2D_DESC desc;
    backBuffer->GetDesc(&desc);
    // No explicit Release needed!
    
    g_cx = desc.Width;
    g_cy = desc.Height;
    g_format = desc.Format;
    
    // Strip SRGB from format
    switch (g_format) {
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            g_format = DXGI_FORMAT_B8G8R8A8_UNORM;
            break;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            g_format = DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
    }
    
    g_hookInfo->cx = g_cx;
    g_hookInfo->cy = g_cy;
    g_hookInfo->format = g_format;
    
    DXGI_SWAP_CHAIN_DESC swapDesc;
    g_swapChain->GetDesc(&swapDesc);
    g_hookInfo->window = (uint64_t)swapDesc.OutputWindow;
    
    if (!InitCaptureTextures()) {
        return false;
    }
    
    g_initialized = true;
    
    // Signal ready
    if (g_hHookReadyEvent) {
        SetEvent(g_hHookReadyEvent);
    }
    
    Log("[Hook] Capture initialized: %ux%u format=%u\n", g_cx, g_cy, g_format);
    return true;
}

// Free capture resources
static void FreeCapture() {
    g_initialized = false;
    
    g_captureTexture.Reset();
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        g_stagingTextures[i].Reset();
    }
    
    if (g_shmemData) {
        UnmapViewOfFile(g_shmemData);
        g_shmemData = nullptr;
    }
    
    if (g_shtexData) {
        UnmapViewOfFile(g_shtexData);
        g_shtexData = nullptr;
    }
    
    if (g_hTextureMap) {
        CloseHandle(g_hTextureMap);
        g_hTextureMap = nullptr;
    }
    
    g_sharedHandle = nullptr;
    g_useSharedTexture = false;
}

// Capture a frame
static void CaptureFrame() {
    if (!g_initialized || !g_swapChain || !g_context) return;
    
    // Check for stop signal
    if (g_hHookStopEvent && WaitForSingleObject(g_hHookStopEvent, 0) == WAIT_OBJECT_0) {
        FreeCapture();
        return;
    }
    
    // Frame interval limiting
    static int64_t lastFrameTime = 0;
    if (g_hookInfo->frame_interval > 0) {
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        
        if (lastFrameTime > 0) {
            int64_t elapsed = now.QuadPart - lastFrameTime;
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            int64_t interval100ns = (elapsed * 10000000) / freq.QuadPart;
            
            if ((uint64_t)interval100ns < g_hookInfo->frame_interval) {
                return; // Skip this frame
            }
        }
        
        lastFrameTime = now.QuadPart;
    }
    
    // Get backbuffer
    ComPtr<ID3D11Texture2D> backBuffer;
    HRESULT hr = g_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    if (FAILED(hr)) return;
    
    if (g_useSharedTexture && g_captureTexture) {
        // Copy to shared texture
        g_context->CopyResource(g_captureTexture.Get(), backBuffer.Get());
        g_hookInfo->frame_count++;
        
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        g_hookInfo->frame_time = time.QuadPart;
    } else if (g_shmemData) {
        // Copy to staging texture
        int curTex = (g_shmemData->last_tex + 1) % NUM_BUFFERS;
        
        // Wait for mutex
        if (g_hTextureMutex[curTex]) {
            if (WaitForSingleObject(g_hTextureMutex[curTex], 0) != WAIT_OBJECT_0) {
                // backBuffer auto-release
                return;
            }
        }
        
        g_context->CopyResource(g_stagingTextures[curTex].Get(), backBuffer.Get());
        
        // Map and copy to shared memory
        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = g_context->Map(g_stagingTextures[curTex].Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (SUCCEEDED(hr)) {
            uint8_t* dest = (uint8_t*)g_shmemData;
            dest += (curTex == 0) ? g_shmemData->tex1_offset : g_shmemData->tex2_offset;
            
            if (mapped.RowPitch == g_hookInfo->pitch) {
                memcpy(dest, mapped.pData, g_hookInfo->pitch * g_cy);
            } else {
                // Row by row copy
                uint8_t* src = (uint8_t*)mapped.pData;
                uint32_t copyPitch = (std::min)(mapped.RowPitch, g_hookInfo->pitch);
                for (uint32_t y = 0; y < g_cy; y++) {
                    memcpy(dest, src, copyPitch);
                    dest += g_hookInfo->pitch;
                    src += mapped.RowPitch;
                }
            }
            
            g_context->Unmap(g_stagingTextures[curTex].Get(), 0);
            g_shmemData->last_tex = curTex;
            g_hookInfo->frame_count++;
            
            LARGE_INTEGER time;
            QueryPerformanceCounter(&time);
            g_hookInfo->frame_time = time.QuadPart;
        }
        
        if (g_hTextureMutex[curTex]) {
            ReleaseMutex(g_hTextureMutex[curTex]);
        }
    }
}
 
// Hooked Present
static HRESULT STDMETHODCALLTYPE HookedPresent(IDXGISwapChain* swapChain, UINT syncInterval, UINT flags) {
    if (!g_initialized && g_active) {
        g_swapChain = swapChain; // Keep ComPtr reference
        swapChain->GetDevice(__uuidof(ID3D11Device), (void**)g_device.ReleaseAndGetAddressOf());
        if (g_device) {
            g_device->GetImmediateContext(g_context.ReleaseAndGetAddressOf());
            // No explicit Release needed, ComPtr holds reference
            InitCapture();
        }
    }
    
    if (g_initialized && swapChain == g_swapChain.Get()) {
        CaptureFrame();
    }
    
    return g_origPresent(swapChain, syncInterval, flags);
}

// Hooked ResizeBuffers
static HRESULT STDMETHODCALLTYPE HookedResizeBuffers(IDXGISwapChain* swapChain, UINT bufferCount,
                                                      UINT width, UINT height, DXGI_FORMAT format, UINT flags) {
    if (swapChain == g_swapChain.Get()) {
        FreeCapture();
    }
    
    HRESULT hr = g_origResizeBuffers(swapChain, bufferCount, width, height, format, flags);
    
    if (SUCCEEDED(hr) && swapChain == g_swapChain.Get() && g_active) {
        InitCapture();
    }
    
    return hr;
}
// VTable hooking
static void* HookVTable(void* obj, int index, void* newFunc) {
    void** vtable = *(void***)obj;
    void* origFunc = vtable[index];
    
    DWORD oldProtect;
    VirtualProtect(&vtable[index], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect);
    vtable[index] = newFunc;
    VirtualProtect(&vtable[index], sizeof(void*), oldProtect, &oldProtect);
    
    return origFunc;
}

// Hook the swap chain
static void HookSwapChain(IDXGISwapChain* swapChain) {
    if (!swapChain) return;
    
    g_swapChain = swapChain;
    
    // Hook Present
    g_origPresent = (PFN_Present)HookVTable(swapChain, DXGI_SWAPCHAIN_PRESENT_INDEX, HookedPresent);
    
    // Hook ResizeBuffers
    g_origResizeBuffers = (PFN_ResizeBuffers)HookVTable(swapChain, DXGI_SWAPCHAIN_RESIZEBUFFERS_INDEX, HookedResizeBuffers);
    
    Log("[Hook] Hooked swap chain vtable\n");
}

// Find and hook D3D11 swap chain using VTable patching
static bool HookD3D11() {
    // Create dummy window
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = g_hModule;
    wc.lpszClassName = L"FrameGenHookDummy";
    RegisterClassExW(&wc);
    
    HWND hwnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPED,
                                 0, 0, 1, 1, nullptr, nullptr, g_hModule, nullptr);
    if (!hwnd) {
        Log("[Hook] Failed to create dummy window\n");
        return false;
    }
    
    // Create D3D11 device and swap chain
    DXGI_SWAP_CHAIN_DESC desc = {};
    desc.BufferCount = 2;
    desc.BufferDesc.Width = 2;
    desc.BufferDesc.Height = 2;
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.BufferDesc.RefreshRate.Numerator = 60;
    desc.BufferDesc.RefreshRate.Denominator = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.OutputWindow = hwnd;
    desc.SampleDesc.Count = 1;
    desc.Windowed = TRUE;
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &desc, &swapChain, &device, nullptr, &context);
    
    if (FAILED(hr)) {
        DestroyWindow(hwnd);
        UnregisterClassW(wc.lpszClassName, g_hModule);
        Log("[Hook] Failed to create D3D11 device: 0x%08X\n", hr);
        return false;
    }
    
    // Get the VTable and hook it - this modifies the shared vtable in dxgi.dll
    // so ALL swap chains will use our hooked functions
    void** vtable = *(void***)swapChain;
    
    // Store original functions
    g_origPresent = (PFN_Present)vtable[DXGI_SWAPCHAIN_PRESENT_INDEX];
    g_origResizeBuffers = (PFN_ResizeBuffers)vtable[DXGI_SWAPCHAIN_RESIZEBUFFERS_INDEX];
    
    Log("[Hook] Original Present=%p, ResizeBuffers=%p\n", g_origPresent, g_origResizeBuffers);
    
    // Patch the vtable - this requires write access
    DWORD oldProtect;
    if (VirtualProtect(&vtable[DXGI_SWAPCHAIN_PRESENT_INDEX], sizeof(void*) * 6, PAGE_EXECUTE_READWRITE, &oldProtect)) {
        vtable[DXGI_SWAPCHAIN_PRESENT_INDEX] = (void*)HookedPresent;
        vtable[DXGI_SWAPCHAIN_RESIZEBUFFERS_INDEX] = (void*)HookedResizeBuffers;
        VirtualProtect(&vtable[DXGI_SWAPCHAIN_PRESENT_INDEX], sizeof(void*) * 6, oldProtect, &oldProtect);
        Log("[Hook] VTable patched successfully\n");
    } else {
        Log("[Hook] Failed to unprotect vtable: %lu\n", GetLastError());
    }
    
    // Also try to get Present1 from IDXGISwapChain1
    IDXGISwapChain1* swapChain1 = nullptr;
    hr = swapChain->QueryInterface(__uuidof(IDXGISwapChain1), (void**)&swapChain1);
    if (SUCCEEDED(hr) && swapChain1) {
        void** vtable1 = *(void***)swapChain1;
        g_origPresent1 = (PFN_Present1)vtable1[DXGI_SWAPCHAIN1_PRESENT1_INDEX];
        
        if (VirtualProtect(&vtable1[DXGI_SWAPCHAIN1_PRESENT1_INDEX], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect)) {
            // We need to add HookedPresent1 function
            // vtable1[DXGI_SWAPCHAIN1_PRESENT1_INDEX] = (void*)HookedPresent1;
            VirtualProtect(&vtable1[DXGI_SWAPCHAIN1_PRESENT1_INDEX], sizeof(void*), oldProtect, &oldProtect);
        }
        swapChain1->Release();
    }
    
    // Clean up dummy objects - but the vtable hook persists!
    swapChain->Release();
    context->Release();
    device->Release();
    DestroyWindow(hwnd);
    UnregisterClassW(wc.lpszClassName, g_hModule);
    
    Log("[Hook] D3D11 hooks installed\n");
    return true;
}

// Hook thread
static DWORD WINAPI HookThread(LPVOID param) {
    Log("[Hook] Hook thread started, PID=%lu\n", g_processId);
    
    // Initialize shared memory
    if (!InitSharedMemory()) {
        Log("[Hook] Failed to init shared memory\n");
        return 1;
    }
    
    // Create events
    g_hHookReadyEvent = CreateNamedEvent(EVENT_HOOK_READY);
    g_hHookStopEvent = CreateNamedEvent(EVENT_CAPTURE_STOP);
    g_hHookExitEvent = CreateNamedEvent(EVENT_HOOK_EXIT);
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        g_hTextureMutex[i] = CreateNamedMutex((i == 0) ? MUTEX_TEXTURE1 : MUTEX_TEXTURE2);
    }
    
    // Wait for d3d11.dll to be loaded
    int waitCount = 0;
    while (!GetModuleHandleA("d3d11.dll") && waitCount < 100) {
        Sleep(100);
        waitCount++;
    }
    
    if (!GetModuleHandleA("d3d11.dll")) {
        Log("[Hook] d3d11.dll not loaded after 10 seconds\n");
        return 1;
    }
    
    Log("[Hook] d3d11.dll found, installing hooks\n");
    
    // Get D3D11 function addresses and hook vtable
    if (!HookD3D11()) {
        Log("[Hook] Failed to hook D3D11\n");
        return 1;
    }
    
    g_active = true;
    
    // Signal init complete
    HANDLE hInitEvent = CreateNamedEvent(EVENT_HOOK_INIT);
    if (hInitEvent) {
        SetEvent(hInitEvent);
        CloseHandle(hInitEvent);
    }
    
    Log("[Hook] Hooks installed, waiting for Present calls...\n");
    
    // Check keepalive - the main app holds this mutex while capturing
    wchar_t keepaliveName[128];
    swprintf(keepaliveName, 128, L"%s%lu", WINDOW_HOOK_KEEPALIVE, g_processId);
    Log("[Hook] Looking for keepalive mutex: %S\n", keepaliveName);
    
    // Main loop - wait for exit
    while (true) {
        if (g_hHookExitEvent && WaitForSingleObject(g_hHookExitEvent, 0) == WAIT_OBJECT_0) {
            Log("[Hook] Exit event signaled\n");
            break;
        }
        
        // Check if main app is still alive by trying to open its mutex
        HANDLE hKeepalive = OpenMutexW(SYNCHRONIZE, FALSE, keepaliveName);
        if (!hKeepalive) {
            DWORD err = GetLastError();
            Log("[Hook] OpenMutexW failed, error=%lu\n", err);
            // Main app has closed
            Log("[Hook] Keepalive mutex gone, exiting\n");
            break;
        }
        CloseHandle(hKeepalive);
        Log("[Hook] Keepalive OK, waiting...\n");
        
        Sleep(500);
    }
    
    Log("[Hook] Exiting hook thread\n");
    
    g_active = false;
    FreeCapture();
    
    // Cleanup
    if (g_hookInfo) {
        UnmapViewOfFile(g_hookInfo);
        g_hookInfo = nullptr;
    }
    if (g_hHookInfoMap) {
        CloseHandle(g_hHookInfoMap);
        g_hHookInfoMap = nullptr;
    }
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (g_hTextureMutex[i]) {
            CloseHandle(g_hTextureMutex[i]);
            g_hTextureMutex[i] = nullptr;
        }
    }
    
    if (g_hHookReadyEvent) CloseHandle(g_hHookReadyEvent);
    if (g_hHookStopEvent) CloseHandle(g_hHookStopEvent);
    if (g_hHookExitEvent) CloseHandle(g_hHookExitEvent);
    if (g_hKeepAliveMutex) CloseHandle(g_hKeepAliveMutex);
    
    return 0;
}

// Exported hook procedure for SetWindowsHookEx injection method
extern "C" __declspec(dllexport) LRESULT CALLBACK DummyHookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    return CallNextHookEx(nullptr, nCode, wParam, lParam);
}

// Early logging before Log() is ready
static void EarlyLog(const char* msg) {
    FILE* f = fopen("C:\\hook_debug.txt", "a");
    if (f) {
        fprintf(f, "[PID %lu] %s\n", GetCurrentProcessId(), msg);
        fclose(f);
    }
}

// Check if this process is our main app (FrameGen) - we don't want to hook ourselves
static bool IsMainApp() {
    wchar_t moduleName[MAX_PATH];
    if (GetModuleFileNameW(nullptr, moduleName, MAX_PATH)) {
        wchar_t* filename = wcsrchr(moduleName, L'\\');
        if (filename) {
            filename++;
            // Check if this is FrameGen.exe
            if (_wcsicmp(filename, L"FrameGen.exe") == 0) {
                return true;
            }
        }
    }
    return false;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID reserved) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            g_hModule = hModule;
            g_processId = GetCurrentProcessId();
            DisableThreadLibraryCalls(hModule);
            
            // Don't do anything if we're loaded into FrameGen itself
            // (this happens when FrameGen loads the DLL to get the hook procedure)
            if (IsMainApp()) {
                EarlyLog("DllMain: Loaded into FrameGen - skipping hook");
                return TRUE;
            }
            
            EarlyLog("DllMain: DLL_PROCESS_ATTACH");
            
            // Increment reference count so DLL stays loaded even after hook is removed
            // This is needed for SetWindowsHookEx injection
            {
                wchar_t modulePath[MAX_PATH];
                if (GetModuleFileNameW(hModule, modulePath, MAX_PATH)) {
                    LoadLibraryW(modulePath);
                    EarlyLog("DllMain: Incremented ref count");
                }
            }
            
            // Start hook thread
            EarlyLog("DllMain: Creating hook thread");
            CreateThread(nullptr, 0, HookThread, nullptr, 0, nullptr);
            break;
            
        case DLL_PROCESS_DETACH:
            EarlyLog("DllMain: DLL_PROCESS_DETACH");
            g_active = false;
            break;
    }
    return TRUE;
}
