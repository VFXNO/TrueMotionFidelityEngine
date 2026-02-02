#include "game_capture.h"
#include <d3d11_1.h>
#include <fstream>
#include <shlwapi.h>

#pragma comment(lib, "shlwapi.lib")

namespace {

std::wstring GetModulePath() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    PathRemoveFileSpecW(path);
    return path;
}

HANDLE OpenNamedEvent(const wchar_t* baseName, DWORD processId) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, processId);
    return OpenEventW(EVENT_ALL_ACCESS, FALSE, name);
}

HANDLE OpenNamedMutex(const wchar_t* baseName, DWORD processId) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, processId);
    return OpenMutexW(MUTEX_ALL_ACCESS, FALSE, name);
}

HANDLE OpenNamedFileMapping(const wchar_t* baseName, DWORD processId) {
    wchar_t name[128];
    swprintf(name, 128, L"%s%lu", baseName, processId);
    return OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
}

}  // namespace

GameCapture::~GameCapture() {
    Shutdown();
}

bool GameCapture::Initialize(ID3D11Device* device) {
    if (!device) {
        m_lastError = "Invalid device";
        return false;
    }
    
    m_device = device;
    device->GetImmediateContext(&m_context);
    QueryPerformanceFrequency(&m_qpcFreq);
    
    // Find hook DLL path
    m_hookDllPath = GetModulePath();
    #ifdef _WIN64
    m_hookDllPath += L"\\graphics_hook64_v2.dll";
    #else
    m_hookDllPath += L"\\graphics_hook32.dll";
    #endif
    
    return true;
}

void GameCapture::Shutdown() {
    StopCapture();
    m_context.Reset();
    m_device.Reset();
}

bool GameCapture::StartCapture(HWND hwnd) {
    if (!m_device || !hwnd) {
        m_lastError = "Invalid device or window";
        return false;
    }
    
    StopCapture();
    
    // Get process ID AND thread ID from the window
    DWORD processId = 0;
    DWORD threadId = GetWindowThreadProcessId(hwnd, &processId);
    if (!processId || !threadId) {
        m_lastError = "Failed to get process/thread ID";
        return false;
    }
    
    // Log the IDs
    {
        FILE* f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] Window HWND=%p, ProcessID=%lu, ThreadID=%lu\n", hwnd, processId, threadId);
            fclose(f);
        }
    }
    
    m_hwnd = hwnd;
    m_processId = processId;
    m_threadId = threadId;  // Store for injection
    
    // Check architecture match
    #ifdef _WIN64
    bool targetIs64 = DllInjector::Is64BitProcess(processId);
    if (!targetIs64) {
        m_lastError = "Cannot inject 64-bit DLL into 32-bit process";
        return false;
    }
    #else
    bool targetIs64 = DllInjector::Is64BitProcess(processId);
    if (targetIs64) {
        m_lastError = "Cannot inject 32-bit DLL into 64-bit process";
        return false;
    }
    #endif
    
    // Create keepalive mutex first
    wchar_t mutexName[128];
    swprintf(mutexName, 128, L"%s%lu", WINDOW_HOOK_KEEPALIVE, processId);
    
    // Debug log
    {
        FILE* f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] Creating keepalive mutex: %S for PID %lu\n", mutexName, processId);
            fclose(f);
        }
    }
    
    m_keepAliveMutex = CreateMutexW(nullptr, TRUE, mutexName);
    if (!m_keepAliveMutex) {
        DWORD err = ::GetLastError();
        FILE* f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] CreateMutexW FAILED, error=%lu\n", err);
            fclose(f);
        }
        m_lastError = "Failed to create keepalive mutex";
        return false;
    }
    
    {
        FILE* f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] Keepalive mutex created: %p\n", m_keepAliveMutex);
            fclose(f);
        }
    }
    
    // Inject hook DLL
    if (!InjectHook(processId)) {
        CloseHandle(m_keepAliveMutex);
        m_keepAliveMutex = nullptr;
        return false;
    }
    
    // Wait for hook to initialize
    if (!WaitForHookReady(5000)) {
        m_lastError = "Hook initialization timeout";
        StopCapture();
        return false;
    }
    
    // Open shared memory
    if (!OpenSharedMemory()) {
        StopCapture();
        return false;
    }
    
    m_isCapturing = true;
    m_isHooked = true;
    
    // Log success
    std::ofstream log("game_capture_log.txt", std::ios::app);
    if (log.is_open()) {
        log << "Game capture started for PID " << processId << "\n";
        log.close();
    }
    
    return true;
}

void GameCapture::StopCapture() {
    // Signal hook to stop
    if (m_hookStopEvent) {
        SetEvent(m_hookStopEvent);
    }
    
    // Signal hook to exit
    if (m_hookExitEvent) {
        SetEvent(m_hookExitEvent);
    }
    
    // Release keepalive mutex (signals hook to exit)
    if (m_keepAliveMutex) {
        ReleaseMutex(m_keepAliveMutex);
        CloseHandle(m_keepAliveMutex);
        m_keepAliveMutex = nullptr;
    }
    
    CloseSharedMemory();
    
    m_captureTexture.Reset();
    m_sharedTexture.Reset();
    
    m_hwnd = nullptr;
    m_processId = 0;
    m_threadId = 0;
    m_isCapturing = false;
    m_isHooked = false;
    m_width = 0;
    m_height = 0;
    m_pitch = 0;
    m_format = DXGI_FORMAT_UNKNOWN;
}

bool GameCapture::InjectHook(DWORD processId) {
    // Check if DLL exists
    if (GetFileAttributesW(m_hookDllPath.c_str()) == INVALID_FILE_ATTRIBUTES) {
        // Convert wstring to string properly
        int size = WideCharToMultiByte(CP_UTF8, 0, m_hookDllPath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string path(size, '\0');
        WideCharToMultiByte(CP_UTF8, 0, m_hookDllPath.c_str(), -1, &path[0], size, nullptr, nullptr);
        m_lastError = "Hook DLL not found: " + path;
        return false;
    }
    
    // Use the window's thread ID directly for better targeting
    if (m_threadId != 0) {
        FILE* f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] Injecting via window thread ID: %lu\n", m_threadId);
            fclose(f);
        }
        auto result = DllInjector::InjectByHook(m_threadId, m_hookDllPath);
        if (result == DllInjector::Result::Success) {
            return true;
        }
        // Fall through to try other methods
        f = fopen("C:\\hook_debug.txt", "a");
        if (f) {
            fprintf(f, "[MainApp] InjectByHook failed, trying Inject()\n");
            fclose(f);
        }
    }
    
    auto result = DllInjector::Inject(processId, m_hookDllPath);
    if (result != DllInjector::Result::Success) {
        m_lastError = std::string("DLL injection failed: ") + DllInjector::GetErrorString(result);
        return false;
    }
    
    return true;
}

bool GameCapture::WaitForHookReady(DWORD timeout) {
    DWORD start = GetTickCount();
    
    while ((GetTickCount() - start) < timeout) {
        // Try to open hook init event
        m_hookInitEvent = OpenNamedEvent(EVENT_HOOK_INIT, m_processId);
        if (m_hookInitEvent) {
            // Wait for it to be signaled
            DWORD result = WaitForSingleObject(m_hookInitEvent, timeout - (GetTickCount() - start));
            CloseHandle(m_hookInitEvent);
            m_hookInitEvent = nullptr;
            
            if (result == WAIT_OBJECT_0) {
                return true;
            }
        }
        
        Sleep(10);
    }
    
    return false;
}

bool GameCapture::OpenSharedMemory() {
    // Open hook info mapping
    m_hookInfoMap = OpenNamedFileMapping(SHMEM_HOOK_INFO, m_processId);
    if (!m_hookInfoMap) {
        m_lastError = "Failed to open hook info mapping";
        return false;
    }
    
    m_hookInfo = (hook_info*)MapViewOfFile(m_hookInfoMap, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(hook_info));
    if (!m_hookInfo) {
        m_lastError = "Failed to map hook info";
        return false;
    }
    
    // Wait for capture to initialize (cx/cy to be set)
    DWORD start = GetTickCount();
    while (m_hookInfo->cx == 0 || m_hookInfo->cy == 0) {
        if ((GetTickCount() - start) > 5000) {
            m_lastError = "Timeout waiting for capture dimensions";
            return false;
        }
        Sleep(10);
    }
    
    m_width = m_hookInfo->cx;
    m_height = m_hookInfo->cy;
    m_pitch = m_hookInfo->pitch;
    m_format = m_hookInfo->format;
    
    // Open events
    m_hookReadyEvent = OpenNamedEvent(EVENT_HOOK_READY, m_processId);
    m_hookStopEvent = OpenNamedEvent(EVENT_CAPTURE_STOP, m_processId);
    m_hookExitEvent = OpenNamedEvent(EVENT_HOOK_EXIT, m_processId);
    
    if (m_hookInfo->type == CAPTURE_TYPE_TEXTURE) {
        // Open shared texture
        m_textureMap = OpenNamedFileMapping(SHMEM_TEXTURE, m_processId);
        if (m_textureMap) {
            m_shtexData = (shtex_data*)MapViewOfFile(m_textureMap, FILE_MAP_READ, 0, 0, sizeof(shtex_data));
            if (m_shtexData) {
                HANDLE sharedHandle = (HANDLE)m_shtexData->tex_handle;
                
                // Open shared texture
                ID3D11Device1* device1 = nullptr;
                HRESULT hr = m_device->QueryInterface(__uuidof(ID3D11Device1), (void**)&device1);
                if (SUCCEEDED(hr)) {
                    hr = device1->OpenSharedResource1(sharedHandle, __uuidof(ID3D11Texture2D),
                                                       (void**)m_sharedTexture.GetAddressOf());
                    device1->Release();
                }
                
                if (FAILED(hr)) {
                    // Try legacy method
                    hr = m_device->OpenSharedResource(sharedHandle, __uuidof(ID3D11Texture2D),
                                                      (void**)m_sharedTexture.GetAddressOf());
                }
                
                if (SUCCEEDED(hr)) {
                    // Success - using shared texture
                    return true;
                }
            }
        }
        
        // Failed to open shared texture - fall through to shmem
    }
    
    // Open shared memory for texture data
    m_textureMap = OpenNamedFileMapping(SHMEM_TEXTURE, m_processId);
    if (!m_textureMap) {
        m_lastError = "Failed to open texture mapping";
        return false;
    }
    
    DWORD dataSize = m_pitch * m_height;
    DWORD mapSize = sizeof(shmem_data) + (dataSize * NUM_BUFFERS);
    
    m_textureData = MapViewOfFile(m_textureMap, FILE_MAP_READ, 0, 0, mapSize);
    if (!m_textureData) {
        m_lastError = "Failed to map texture data";
        return false;
    }
    
    m_shmemData = (shmem_data*)m_textureData;
    
    // Open texture mutexes
    for (int i = 0; i < NUM_BUFFERS; i++) {
        wchar_t name[128];
        swprintf(name, 128, L"%s%lu", (i == 0) ? MUTEX_TEXTURE1 : MUTEX_TEXTURE2, m_processId);
        m_textureMutex[i] = OpenMutexW(MUTEX_ALL_ACCESS, FALSE, name);
    }
    
    return true;
}

void GameCapture::CloseSharedMemory() {
    if (m_hookInfo) {
        UnmapViewOfFile(m_hookInfo);
        m_hookInfo = nullptr;
    }
    if (m_hookInfoMap) {
        CloseHandle(m_hookInfoMap);
        m_hookInfoMap = nullptr;
    }
    
    if (m_textureData) {
        UnmapViewOfFile(m_textureData);
        m_textureData = nullptr;
    }
    if (m_shtexData) {
        UnmapViewOfFile(m_shtexData);
        m_shtexData = nullptr;
    }
    if (m_textureMap) {
        CloseHandle(m_textureMap);
        m_textureMap = nullptr;
    }
    
    for (int i = 0; i < NUM_BUFFERS; i++) {
        if (m_textureMutex[i]) {
            CloseHandle(m_textureMutex[i]);
            m_textureMutex[i] = nullptr;
        }
    }
    
    if (m_hookReadyEvent) {
        CloseHandle(m_hookReadyEvent);
        m_hookReadyEvent = nullptr;
    }
    if (m_hookStopEvent) {
        CloseHandle(m_hookStopEvent);
        m_hookStopEvent = nullptr;
    }
    if (m_hookExitEvent) {
        CloseHandle(m_hookExitEvent);
        m_hookExitEvent = nullptr;
    }
    
    m_shmemData = nullptr;
}

bool GameCapture::AcquireNextFrame(CapturedFrame& frame) {
    if (!m_isCapturing || !m_hookInfo) {
        return false;
    }
    
    // Check if process is still running
    if (!IsWindow(m_hwnd)) {
        StopCapture();
        return false;
    }
    
    // Check for dimension changes
    if (m_hookInfo->cx != (uint32_t)m_width || m_hookInfo->cy != (uint32_t)m_height) {
        m_width = m_hookInfo->cx;
        m_height = m_hookInfo->cy;
        m_pitch = m_hookInfo->pitch;
        m_format = m_hookInfo->format;
        m_captureTexture.Reset();
    }
    
    EnsureCaptureTexture(m_width, m_height);
    if (!m_captureTexture) {
        return false;
    }
    
    // Get frame data
    if (m_sharedTexture) {
        // Shared texture path - direct GPU copy
        static uint64_t lastFrameCount = 0;
        if (m_hookInfo->frame_count == lastFrameCount) {
            return false; // No new frame
        }
        lastFrameCount = m_hookInfo->frame_count;
        
        m_context->CopyResource(m_captureTexture.Get(), m_sharedTexture.Get());
    } else if (m_shmemData) {
        // Shared memory path - CPU to GPU copy
        int curTex = m_shmemData->last_tex;
        if (curTex < 0 || curTex >= NUM_BUFFERS) {
            return false;
        }
        
        static int lastTex = -1;
        if (curTex == lastTex) {
            return false; // No new frame
        }
        
        // Lock mutex
        if (m_textureMutex[curTex]) {
            if (WaitForSingleObject(m_textureMutex[curTex], 0) != WAIT_OBJECT_0) {
                return false;
            }
        }
        
        // Get texture data pointer
        uint8_t* srcData = (uint8_t*)m_shmemData;
        srcData += (curTex == 0) ? m_shmemData->tex1_offset : m_shmemData->tex2_offset;
        
        // Update texture
        m_context->UpdateSubresource(m_captureTexture.Get(), 0, nullptr, 
                                     srcData, m_pitch, 0);
        
        lastTex = curTex;
        
        // Release mutex
        if (m_textureMutex[curTex]) {
            ReleaseMutex(m_textureMutex[curTex]);
        }
    } else {
        return false;
    }
    
    // Fill frame info
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    
    frame.texture = m_captureTexture;
    frame.width = m_width;
    frame.height = m_height;
    frame.qpcTime = qpc.QuadPart;
    
    if (m_qpcFreq.QuadPart > 0) {
        double qpcTo100ns = 1e7 / static_cast<double>(m_qpcFreq.QuadPart);
        frame.systemTime100ns = static_cast<int64_t>(qpc.QuadPart * qpcTo100ns);
    } else {
        frame.systemTime100ns = 0;
    }
    
    return true;
}

uint64_t GameCapture::GetFrameCount() const {
    return m_hookInfo ? m_hookInfo->frame_count : 0;
}

void GameCapture::SetFrameRateLimit(double fps) {
    if (fps > 0) {
        m_frameInterval = static_cast<uint64_t>(10000000.0 / fps);
    } else {
        m_frameInterval = 0;
    }
    
    if (m_hookInfo) {
        m_hookInfo->frame_interval = m_frameInterval;
    }
}

void GameCapture::EnsureCaptureTexture(int width, int height) {
    if (width <= 0 || height <= 0) return;
    
    if (m_captureTexture) {
        D3D11_TEXTURE2D_DESC desc;
        m_captureTexture->GetDesc(&desc);
        if ((int)desc.Width == width && (int)desc.Height == height) {
            return;
        }
    }
    
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = (m_format != DXGI_FORMAT_UNKNOWN) ? m_format : DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    m_captureTexture.Reset();
    m_device->CreateTexture2D(&desc, nullptr, &m_captureTexture);
}
