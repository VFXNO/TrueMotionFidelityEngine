#pragma once

#include "capture_frame.h"
#include "graphics_hook_info.h"
#include "dll_injector.h"

#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>
#include <string>
#include <atomic>

class GameCapture {
public:
    GameCapture() = default;
    ~GameCapture();
    
    bool Initialize(ID3D11Device* device);
    void Shutdown();
    
    // Start capturing a window by injecting hook into its process
    bool StartCapture(HWND hwnd);
    void StopCapture();
    
    // Acquire next captured frame
    bool AcquireNextFrame(CapturedFrame& frame);
    
    bool IsCapturing() const { return m_isCapturing; }
    bool IsHooked() const { return m_isHooked; }
    
    // Get capture statistics
    uint64_t GetFrameCount() const;
    int GetWidth() const { return m_width; }
    int GetHeight() const { return m_height; }
    
    // Configuration
    void SetForceSharedMemory(bool force) { m_forceSharedMemory = force; }
    void SetFrameRateLimit(double fps);
    
    // Get last error message
    const std::string& GetLastError() const { return m_lastError; }

private:
    bool InjectHook(DWORD processId);
    bool OpenSharedMemory();
    void CloseSharedMemory();
    bool WaitForHookReady(DWORD timeout);
    void EnsureCaptureTexture(int width, int height);
    
    Microsoft::WRL::ComPtr<ID3D11Device> m_device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_captureTexture;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_sharedTexture;
    
    // Target info
    HWND m_hwnd = nullptr;
    DWORD m_processId = 0;
    DWORD m_threadId = 0;
    
    // Shared memory handles
    HANDLE m_hookInfoMap = nullptr;
    HANDLE m_textureMap = nullptr;
    HANDLE m_textureMutex[NUM_BUFFERS] = {nullptr};
    HANDLE m_keepAliveMutex = nullptr;
    HANDLE m_hookReadyEvent = nullptr;
    HANDLE m_hookStopEvent = nullptr;
    HANDLE m_hookExitEvent = nullptr;
    HANDLE m_hookInitEvent = nullptr;
    
    // Shared memory pointers
    hook_info* m_hookInfo = nullptr;
    shmem_data* m_shmemData = nullptr;
    shtex_data* m_shtexData = nullptr;
    void* m_textureData = nullptr;
    
    // State
    bool m_isCapturing = false;
    bool m_isHooked = false;
    int m_width = 0;
    int m_height = 0;
    uint32_t m_pitch = 0;
    DXGI_FORMAT m_format = DXGI_FORMAT_UNKNOWN;
    
    // Configuration
    bool m_forceSharedMemory = false;
    uint64_t m_frameInterval = 0;
    
    // Timing
    LARGE_INTEGER m_qpcFreq = {};
    int64_t m_lastCaptureTime = 0;
    
    std::string m_lastError;
    std::wstring m_hookDllPath;
};
