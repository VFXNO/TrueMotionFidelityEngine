#include "app.h"
#include "resource.h"

#include <imgui.h>
#include <imgui_impl_win32.h>

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <windowsx.h>
#include <mmsystem.h>
#include <shlobj.h>
#include <cstdio>

namespace {

std::string WideToUtf8(const std::wstring& wide) {
  if (wide.empty()) {
    return {};
  }

  int size = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, nullptr, nullptr);
  if (size <= 0) {
    return {};
  }

  std::string result(size, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &result[0], size, nullptr, nullptr);
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

// Helper to detect key name string from VK code
static const char* GetKeyName(int vk) {
    static char name[32];
    if (vk == 0) return "None";
    
    // Function keys
    if (vk >= VK_F1 && vk <= VK_F24) {
        sprintf_s(name, "F%d", vk - VK_F1 + 1);
        return name;
    }
    
    // Numbers
    if (vk >= '0' && vk <= '9') {
        sprintf_s(name, "%c", (char)vk);
        return name;
    }
    
    // Letters
    if (vk >= 'A' && vk <= 'Z') {
        sprintf_s(name, "%c", (char)vk);
        return name;
    }

    switch(vk) {
        case VK_ESCAPE: return "Esc";
        case VK_TAB: return "Tab";
        case VK_SPACE: return "Space";
        case VK_RETURN: return "Enter";
        case VK_BACK: return "Backspace";
        case VK_INSERT: return "Insert";
        case VK_DELETE: return "Delete";
        case VK_HOME: return "Home";
        case VK_END: return "End";
        case VK_PRIOR: return "PgUp";
        case VK_NEXT: return "PgDn";
        case VK_LEFT: return "Left";
        case VK_UP: return "Up";
        case VK_RIGHT: return "Right";
        case VK_DOWN: return "Down";
    }

    sprintf_s(name, "VK_%d", vk);
    return name;
}

// Global hotkey editor helper
static void HotkeyEditor(const char* label, int* currentHotkey) {
    ImGui::Text("%s", label);
    ImGui::SameLine();
    char btnLabel[64];
    sprintf_s(btnLabel, "[ %s ]###%s", GetKeyName(*currentHotkey), label);
    
    if (ImGui::Button(btnLabel, ImVec2(100, 0))) {
        ImGui::OpenPopup(label);
    }
    
    if (ImGui::BeginPopup(label)) {
        ImGui::Text("Press any key to bind...");
        ImGui::Text("(Press Esc to clear/cancel)");
        
        // Scan for key press
        for (int i = 8; i <= 255; i++) {
            if (ImGui::IsKeyPressed((ImGuiKey)i) || (GetAsyncKeyState(i) & 0x8000)) {
               // ImGui maps standard keys, and we can also use Windows VKs
               // Skip mouse buttons
               if (i == VK_LBUTTON || i == VK_RBUTTON || i == VK_MBUTTON) continue;
               
               if (i == VK_ESCAPE) {
                   // *currentHotkey = 0; // Or just cancel? Let's just close
                   ImGui::CloseCurrentPopup();
                   break;
               }

               // Valid key
               *currentHotkey = i;
               ImGui::CloseCurrentPopup();
               break;
            }
        }
        ImGui::EndPopup();
    }
}

}  // namespace

App::App() {
  QueryPerformanceFrequency(&m_qpcFreq);
  m_waitTimer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  if (!m_waitTimer) {
     m_waitTimer = CreateWaitableTimerW(nullptr, FALSE, nullptr);
  }
}

bool App::ShouldUseWgcForWindowCapture() const {
  if (m_captureMode != 0) {
    return false;
  }
  
  // User override
  if (m_forceDxgiCapture) {
      return false;
  }

  // Overlay mode normally requires WGC since the overlay covers the game window
  if (m_overlayMode || m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled) {
    return true;
  }
  return m_windowCapturePreferWgc;
}

bool App::Initialize(HINSTANCE hInstance) {
  // Prevent Windows from sleeping or turning off the display while running
  SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);

  // Set high priority to avoid CPU throttling
  // Use NORMAL to prevent system starvation
  SetPriorityClass(GetCurrentProcess(), NORMAL_PRIORITY_CLASS);
  
  // Set high resolution timer
  timeBeginPeriod(1);

  m_hInstance = hInstance;

  if (!CreateAppWindow(hInstance)) {
    m_captureStatus = "Error: Failed to create main window";
    MessageBoxW(nullptr, L"Error: Failed to create main window", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  if (!m_device.Initialize(m_hwnd)) {
    MessageBoxW(nullptr, L"Error: Failed to initialize D3D11 device. Check GPU drivers and that D3D11 is supported.", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  if (!CreateUiWindow(hInstance)) {
    m_captureStatus = "Error: Failed to create UI window";
    MessageBoxW(nullptr, L"Error: Failed to create UI window", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  const auto& monitors = m_device.Monitors();
  if (!monitors.empty()) {
    SelectMonitor(0);
  }

  if (!m_ui.Initialize(m_uiHwnd, m_device.Device(), m_device.Context())) {
    m_captureStatus = "Error: Failed to initialize UI";
    MessageBoxW(nullptr, L"Error: Failed to initialize UI", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  if (!m_interpolator.Initialize(m_device.Device(), m_device.Context())) {
    m_captureStatus = "Error: Failed to initialize interpolator. Check GPU supports Compute Shader 5.0";
    MessageBoxW(nullptr, L"Error: Failed to initialize interpolator. Check GPU supports Compute Shader 5.0\n\nCheck DebugView output or shader file paths.", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  if (!m_capture.Initialize(m_device.Device())) {
    m_captureStatus = "Error: Failed to initialize WGC capture (requires Windows 10/11)";
    MessageBoxW(nullptr, L"Error: Failed to initialize WGC capture (requires Windows 10/11)", L"True Motion Fidelity Engine Error", MB_OK);
    return false;
  }

  if (!m_dupCapture.Initialize(m_device.Device())) {
    m_captureStatus = "Error: Failed to initialize Desktop Duplication";
    MessageBoxW(nullptr, L"Error: Failed to initialize Desktop Duplication", L"FrameGen Error", MB_OK);
    return false;
  }

  if (!m_gameCapture.Initialize(m_device.Device())) {
    // Game capture is optional, just log warning
    std::ofstream log("init_log.txt", std::ios::app);
    if (log.is_open()) {
      log << "Warning: Game capture initialization failed\n";
      log.close();
    }
  }

  RefreshWindowList();
  return true;
}

int App::Run() {
  MSG msg = {};
  while (msg.message != WM_QUIT) {
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }

    Update();
    Render();
    RenderUiWindow();
  }

  m_gameCapture.Shutdown();
  m_dupCapture.Shutdown();
  m_capture.Shutdown();
  m_ui.Shutdown();
  m_device.Shutdown();

  // Restore default power state and timer
  SetThreadExecutionState(ES_CONTINUOUS);
  timeEndPeriod(1);

  return static_cast<int>(msg.wParam);
}

bool App::CreateAppWindow(HINSTANCE hInstance) {
  const wchar_t* className = L"FrameGenWindow";

  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(WNDCLASSEXW);
  wc.lpfnWndProc = App::WndProc;
  wc.hInstance = hInstance;
  wc.lpszClassName = className;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MAIN_ICON));
  wc.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MAIN_ICON));
  wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));

  if (!RegisterClassExW(&wc)) {
    return false;
  }

  // Create as click-through overlay: WS_EX_LAYERED for transparency,
  // WS_EX_TRANSPARENT to pass input through, WS_EX_NOACTIVATE to never steal focus
  m_hwnd = CreateWindowExW(
      WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOPMOST,
      className,
      L"True Motion Fidelity Engine Output",
      WS_POPUP,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      1280,
      720,
      nullptr,
      nullptr,
      hInstance,
      this);

  if (!m_hwnd) {
    return false;
  }

  // Set window to be fully opaque (alpha = 255)
  SetLayeredWindowAttributes(m_hwnd, 0, 255, LWA_ALPHA);
  
  // Start HIDDEN - only show when capturing
  ShowWindow(m_hwnd, SW_HIDE);
  m_outputWindowVisible = false;
  return true;
}

bool App::CreateUiWindow(HINSTANCE hInstance) {
  const wchar_t* className = L"TrueMotionFidelityEngineUIWindow";

  WNDCLASSEXW wc = {};
  wc.cbSize = sizeof(WNDCLASSEXW);
  wc.lpfnWndProc = App::UiWndProc;
  wc.hInstance = hInstance;
  wc.lpszClassName = className;
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MAIN_ICON));
  wc.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MAIN_ICON));
  wc.hbrBackground = reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));

  if (!RegisterClassExW(&wc)) {
    return false;
  }

  m_uiHwnd = CreateWindowExW(
      0,
      className,
      L"True Motion Fidelity Engine Controls",
      WS_OVERLAPPEDWINDOW | WS_VISIBLE,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      m_uiWidth,
      m_uiHeight,
      nullptr,
      nullptr,
      hInstance,
      this);

  if (!m_uiHwnd) {
    return false;
  }

  RECT rect = {};
  GetClientRect(m_uiHwnd, &rect);
  m_uiWidth = rect.right - rect.left;
  m_uiHeight = rect.bottom - rect.top;

  if (!m_device.CreateSwapChainForWindow(
          m_uiHwnd,
          static_cast<UINT>(m_uiWidth),
          static_cast<UINT>(m_uiHeight),
          m_uiSwapChain,
          m_uiRtv)) {
    return false;
  }

  m_uiVisible = true;
  return true;
}

bool App::StartWindowCapture(HWND hwnd) {
  std::ofstream log("capture_log.txt", std::ios::app);
  log << "\n=== StartWindowCapture called ===\n";

  if (!hwnd) {
    log << "Error: hwnd is null\n";
    log.close();
    return false;
  }

  log << "hwnd: " << (void*)hwnd << "\n";

  m_capture.StopCapture();
  m_dupCapture.StopCapture();
  m_windowCaptureUsingWgc = false;
  m_captureWindowBehindOutput = false;
  m_zOrderCaptureWindow = nullptr;

  HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
  log << "Monitor handle: " << (void*)monitor << "\n";

  bool requireWgc = m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled;
  bool useWgc = ShouldUseWgcForWindowCapture() || m_forceWgcCapture;

  log << "requireWgc: " << requireWgc << "\n";
  log << "useWgc: " << useWgc << "\n";

  if (useWgc) {
    log << "Trying WGC capture...\n";
    m_capture.SetLowLatencyMode(m_wgcLowLatencyMode);
    m_capture.SetPreferNewestFrame(m_wgcPreferNewest);
    if (m_capture.StartCapture(hwnd)) {
      m_windowCaptureUsingWgc = true;
      log << "WGC capture started successfully\n";
    } else if (!requireWgc) {
      m_capture.StopCapture();
      log << "WGC failed, fallback disabled\n";
    } else {
      if (m_forceWgcCapture) {
        m_captureStatus = "Error: WGC failed - Try disabling AMD MPO in GPU control panel, disable Variable Refresh Rate, or update drivers";
        log << "WGC failed with forceWgc, returning error\n";
        log.close();
        return false;
      }
      log << "WGC failed, returning false\n";
      log.close();
      return false;
    }
  }

  if (!m_windowCaptureUsingWgc) {
    log << "Trying Desktop Duplication capture...\n";
    Microsoft::WRL::ComPtr<IDXGIOutput> output;
    RECT outputRect = {};
    if (!m_device.OutputForMonitor(monitor, output, outputRect)) {
      log << "Error: OutputForMonitor failed\n";
      log.close();
      return false;
    }
    log << "OutputForMonitor succeeded\n";

    // Apply Low Latency settings to DXGI capture as well
    m_dupCapture.SetPreferNewestFrame(m_wgcPreferNewest);

    if (!m_dupCapture.StartCapture(hwnd, output, outputRect)) {
      log << "Error: dupCapture.StartCapture failed\n";
      log.close();
      return false;
    }
    log << "Desktop Duplication capture started\n";
  }

  m_captureWindow = hwnd;
  m_captureWindowMonitor = monitor;
  if (m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled) {
    m_outputDisplayMode = 0;
  }
  if (m_fullscreenWindowOutput) {
    SelectMonitor(m_selectedMonitor);
  }
  m_outputMouseIgnore = 0;
  log << "StartWindowCapture returning true\n";
  log.close();
  return true;
}

bool App::StartMonitorCapture(HMONITOR monitor) {
  if (!monitor) {
    return false;
  }

  m_dupCapture.StopCapture();
  m_windowCaptureUsingWgc = false;
  m_captureWindowBehindOutput = false;
  m_zOrderCaptureWindow = nullptr;
  
  // Restore output window if it was moved for DXGI Crop mode
  RestoreDxgiCropWindow();
  
  // Only clear capture window if NOT in DXGI Crop mode
  if (m_captureMode != 3) {
    m_captureWindow = nullptr;
  }
  m_captureWindowMonitor = monitor; // Update the monitor being captured
  
  m_outputDisplayMode = 0;
  m_outputMouseIgnore = 0;

  if (m_captureMode == 3) {
    // DXGI Desktop Duplication
    Microsoft::WRL::ComPtr<IDXGIOutput> output;
    RECT outputRect = {};
    if (!m_device.OutputForMonitor(monitor, output, outputRect)) {
      return false;
    }

    m_dupCapture.SetPreferNewestFrame(m_wgcPreferNewest);

    // Pass the window we're cropping to, if available
    HWND hwnd = m_captureWindow;
    if (!hwnd || !IsWindow(hwnd)) {
        hwnd = GetDesktopWindow(); 
    }
    return m_dupCapture.StartCapture(hwnd, output, outputRect);

  } else {
    // Windows Graphics Capture
    m_capture.SetLowLatencyMode(m_wgcLowLatencyMode);
    m_capture.SetPreferNewestFrame(m_wgcPreferNewest);
    return m_capture.StartCaptureMonitor(monitor);
  }
}

void App::ResetCaptureState() {
  m_frameQueue.clear();
  m_queueWrite = 0;
  m_outputStepIndex = 0;
  m_holdEndFrame = false;
  m_pairPrevSlot = -1;
  m_pairCurrSlot = -1;
  m_pairPrevTime100ns = 0;
  m_pairCurrTime100ns = 0;
  m_frameTime100ns.fill(0);
  m_prevFrameTime100ns = 0;
  m_currFrameTime100ns = 0;
  m_timeOffsetValid = false;
  m_timeOffset100ns = 0.0;
  m_avgFrameInterval = 0.0;
  m_wgcFrameArrivalTime = 0.0;
  m_wgcFrameArrivalCount = 0;
  m_wgcArrivalRate = 0.0f;
  m_frameIntervalSum = 0.0;
  m_frameIntervalCount = 0;
  m_minFrameInterval = 9999.0f;
  m_maxFrameInterval = 0.0f;
  m_frameTimestamps.clear();
  m_lastOutputSrv.Reset();
  m_lastOutputWidth = 0;
  m_lastOutputHeight = 0;
  m_outputMouseIgnore = 0;
  m_previewInputActive = false;
  m_captureWindowBehindOutput = false;
  m_zOrderCaptureWindow = nullptr;
  m_interpolator.ResetTemporal();
  
  // Release cursor confinement when stopping capture
  if (m_cursorConfined) {
    ClipCursor(nullptr);
    m_cursorConfined = false;
  }
  m_lastMousePos = {0, 0};
}

void App::UpdateOutputOverlayWindow() {
  if (!m_hwnd || !m_captureWindow) {
    return;
  }

  RECT clientRect = {};
  if (!GetClientRect(m_captureWindow, &clientRect)) {
    return;
  }

  POINT topLeft{clientRect.left, clientRect.top};
  POINT bottomRight{clientRect.right, clientRect.bottom};
  if (!ClientToScreen(m_captureWindow, &topLeft) || !ClientToScreen(m_captureWindow, &bottomRight)) {
    return;
  }

  int width = bottomRight.x - topLeft.x;
  int height = bottomRight.y - topLeft.y;
  if (width <= 0 || height <= 0) {
    return;
  }

  // Optimize: Only call potentially expensive SetWindowPos if position/size actually changed
  RECT newRect = {topLeft.x, topLeft.y, topLeft.x + width, topLeft.y + height};
  if (memcmp(&newRect, &m_lastOverlayRect, sizeof(RECT)) != 0) {
      SetWindowPos(m_hwnd, HWND_TOPMOST, topLeft.x, topLeft.y, width, height,
                   SWP_NOACTIVATE | SWP_SHOWWINDOW);
      m_lastOverlayRect = newRect;
  }
}

void App::UpdateOutputWindowMode() {
  int mode = m_outputDisplayMode;
  
  bool forceOutputWindow = (m_captureMode == 0 &&
                            (m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled));
  if (forceOutputWindow) {
    mode = 0;
  }
  
  if (mode == 2 && (m_captureMode != 0 || !m_captureWindow)) {
    mode = 1;
  }

  if (m_outputDisplayMode != mode) {
    m_outputDisplayMode = mode;
  }

  if (!m_hwnd) {
    return;
  }

  bool wantTopmost = (m_captureMode == 0 && m_captureWindow && m_hideCaptureWindow && mode == 0);
  if (wantTopmost != m_outputTopmost) {
    SetWindowPos(m_hwnd, wantTopmost ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    m_outputTopmost = wantTopmost;
  }

  // Always keep UI topmost when visible so it appears above the overlay
  bool wantUiTopmost = m_uiHwnd && m_uiVisible && m_showUi;
  if (m_uiHwnd && wantUiTopmost != m_uiTopmost) {
    SetWindowPos(m_uiHwnd, wantUiTopmost ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0,
                 SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
    m_uiTopmost = wantUiTopmost;
  }

  if (wantTopmost && m_captureWindow) {
    if (!m_captureWindowBehindOutput || m_zOrderCaptureWindow != m_captureWindow) {
      SetWindowPos(m_captureWindow, HWND_NOTOPMOST, 0, 0, 0, 0,
                   SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);
      m_captureWindowBehindOutput = true;
      m_zOrderCaptureWindow = m_captureWindow;
    }
  } else if (m_captureWindowBehindOutput) {
    m_captureWindowBehindOutput = false;
    m_zOrderCaptureWindow = nullptr;
  }

  if (m_outputWindowMode == mode) {
    if (mode == 2) {
      UpdateOutputOverlayWindow();
    }
    return;
  }

  m_outputWindowMode = mode;

  LONG exStyle = GetWindowLong(m_hwnd, GWL_EXSTYLE);
  if (mode == 2) {
    exStyle |= WS_EX_TRANSPARENT | WS_EX_LAYERED;
  } else {
    exStyle &= ~(WS_EX_TRANSPARENT | WS_EX_LAYERED);
  }
  SetWindowLong(m_hwnd, GWL_EXSTYLE, exStyle);
  
  if (mode == 2) {
    SetLayeredWindowAttributes(m_hwnd, 0, 255, LWA_ALPHA);
  }

  if (mode == 2) {
    SetWindowDisplayAffinity(m_hwnd, WDA_EXCLUDEFROMCAPTURE);
    UpdateOutputOverlayWindow();
  } else {
    SetWindowDisplayAffinity(m_hwnd, WDA_NONE);
  }
}

bool App::IsPreviewPoint(POINT pt) const {
  if (!m_previewHasImage) {
    return false;
  }
  return pt.x >= m_previewRect.left && pt.x < m_previewRect.right &&
         pt.y >= m_previewRect.top && pt.y < m_previewRect.bottom;
}

bool App::MapPreviewPoint(POINT pt, POINT& outPt) const {
  if (!m_previewHasImage || m_frameWidth <= 0 || m_frameHeight <= 0) {
    return false;
  }

  int width = m_previewRect.right - m_previewRect.left;
  int height = m_previewRect.bottom - m_previewRect.top;
  if (width <= 0 || height <= 0) {
    return false;
  }

  if (pt.x < m_previewRect.left) {
    pt.x = m_previewRect.left;
  } else if (pt.x >= m_previewRect.right) {
    pt.x = m_previewRect.right - 1;
  }

  if (pt.y < m_previewRect.top) {
    pt.y = m_previewRect.top;
  } else if (pt.y >= m_previewRect.bottom) {
    pt.y = m_previewRect.bottom - 1;
  }

  float u = static_cast<float>(pt.x - m_previewRect.left) / static_cast<float>(width);
  float v = static_cast<float>(pt.y - m_previewRect.top) / static_cast<float>(height);
  int x = static_cast<int>(u * static_cast<float>(m_frameWidth));
  int y = static_cast<int>(v * static_cast<float>(m_frameHeight));
  if (x < 0) {
    x = 0;
  } else if (x >= m_frameWidth) {
    x = m_frameWidth - 1;
  }
  if (y < 0) {
    y = 0;
  } else if (y >= m_frameHeight) {
    y = m_frameHeight - 1;
  }
  outPt.x = x;
  outPt.y = y;
  return true;
}

bool App::MapOutputPoint(POINT pt, POINT& outPt) const {
  if (!m_hwnd || m_frameWidth <= 0 || m_frameHeight <= 0) {
    return false;
  }

  RECT rect = {};
  if (!GetClientRect(m_hwnd, &rect)) {
    return false;
  }

  int width = rect.right - rect.left;
  int height = rect.bottom - rect.top;
  if (width <= 0 || height <= 0) {
    return false;
  }

  if (pt.x < 0) {
    pt.x = 0;
  } else if (pt.x >= width) {
    pt.x = width - 1;
  }

  if (pt.y < 0) {
    pt.y = 0;
  } else if (pt.y >= height) {
    pt.y = height - 1;
  }

  float u = static_cast<float>(pt.x) / static_cast<float>(width);
  float v = static_cast<float>(pt.y) / static_cast<float>(height);
  int x = static_cast<int>(u * static_cast<float>(m_frameWidth));
  int y = static_cast<int>(v * static_cast<float>(m_frameHeight));
  if (x < 0) {
    x = 0;
  } else if (x >= m_frameWidth) {
    x = m_frameWidth - 1;
  }
  if (y < 0) {
    y = 0;
  } else if (y >= m_frameHeight) {
    y = m_frameHeight - 1;
  }
  outPt.x = x;
  outPt.y = y;
  return true;
}

bool App::HandleOutputMouse(UINT message, WPARAM wParam, LPARAM lParam) {
  // Only handle mouse messages
  if (message != WM_MOUSEMOVE && message != WM_LBUTTONDOWN && message != WM_LBUTTONUP &&
      message != WM_RBUTTONDOWN && message != WM_RBUTTONUP &&
      message != WM_MBUTTONDOWN && message != WM_MBUTTONUP &&
      message != WM_MOUSEWHEEL && message != WM_MOUSEHWHEEL) {
    return false;
  }
  
  if (!m_outputInputEnabled || m_outputDisplayMode != 0 || !m_captureWindow || m_captureMode != 0) {
    if (m_cursorConfined) {
      ClipCursor(nullptr);
      m_cursorConfined = false;
    }
    return false;
  }

  if (message == WM_MOUSEMOVE && m_outputMouseIgnore > 0) {
    m_outputMouseIgnore--;
    return true;
  }
  
  // Handle cursor confinement
  if (m_outputConfineCursor && !m_cursorConfined) {
    RECT clientRect;
    if (GetClientRect(m_hwnd, &clientRect)) {
      POINT topLeft = {clientRect.left, clientRect.top};
      POINT bottomRight = {clientRect.right, clientRect.bottom};
      ClientToScreen(m_hwnd, &topLeft);
      ClientToScreen(m_hwnd, &bottomRight);
      RECT screenRect = {topLeft.x, topLeft.y, bottomRight.x, bottomRight.y};
      ClipCursor(&screenRect);
      m_cursorConfined = true;
    }
  }

  bool isMouseMove = (message == WM_MOUSEMOVE);
  bool isClick = (message == WM_LBUTTONDOWN || message == WM_LBUTTONUP ||
                  message == WM_RBUTTONDOWN || message == WM_RBUTTONUP ||
                  message == WM_MBUTTONDOWN || message == WM_MBUTTONUP);
  bool isWheel = (message == WM_MOUSEWHEEL || message == WM_MOUSEHWHEEL);
  bool buttonDown = (wParam & (MK_LBUTTON | MK_RBUTTON | MK_MBUTTON)) != 0;
  bool forwardMove = isMouseMove && (m_outputForwardMouseMove || buttonDown);
  bool forwardEvent = forwardMove || isClick || isWheel;
  if (!forwardEvent) {
    return false;
  }

  // Get point in output window client coordinates
  POINT outputPoint = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
  if (isWheel) {
    ScreenToClient(m_hwnd, &outputPoint);
  }
  
  // Map to capture window client coordinates
  POINT target = {};
  if (!MapOutputPoint(outputPoint, target)) {
    return false;
  }

  // Convert to screen coordinates for cursor positioning
  POINT targetScreen = target;
  if (!ClientToScreen(m_captureWindow, &targetScreen)) {
    return false;
  }

  // Always move cursor to target position
  SetCursorPos(targetScreen.x, targetScreen.y);
  m_outputMouseIgnore = 2;

  // Build lParam for PostMessage (client coordinates of target window)
  LPARAM targetLParam = MAKELPARAM(target.x, target.y);

  // Focus the capture window for clicks
  if (isClick) {
    DWORD ourThread = GetCurrentThreadId();
    DWORD targetThread = GetWindowThreadProcessId(m_captureWindow, nullptr);
    bool attached = false;
    if (ourThread != targetThread) {
      attached = AttachThreadInput(ourThread, targetThread, TRUE);
    }
    SetForegroundWindow(m_captureWindow);
    SetFocus(m_captureWindow);
    if (attached) {
      AttachThreadInput(ourThread, targetThread, FALSE);
    }
    
    // Use SendInput for the actual click - more reliable than PostMessage
    INPUT input = {};
    input.type = INPUT_MOUSE;
    if (message == WM_LBUTTONDOWN) {
      input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    } else if (message == WM_LBUTTONUP) {
      input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    } else if (message == WM_RBUTTONDOWN) {
      input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
    } else if (message == WM_RBUTTONUP) {
      input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
    } else if (message == WM_MBUTTONDOWN) {
      input.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
    } else if (message == WM_MBUTTONUP) {
      input.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
    }
    SendInput(1, &input, sizeof(INPUT));
    return true;
  }

  if (isWheel) {
    // For wheel messages, wParam contains wheel delta, lParam is screen coords
    LPARAM wheelLParam = MAKELPARAM(targetScreen.x, targetScreen.y);
    PostMessage(m_captureWindow, message, wParam, wheelLParam);
    return true;
  }

  if (forwardMove) {
    // For mouse move, post to capture window
    PostMessage(m_captureWindow, WM_MOUSEMOVE, wParam, targetLParam);
    return true;
  }

  return false;
}

bool App::HandleOutputKey(UINT message, WPARAM wParam, LPARAM lParam) {
  if (!m_outputInputEnabled || m_outputDisplayMode != 0 || !m_captureWindow || m_captureMode != 0) {
    return false;
  }
  
  // ESC key releases cursor confinement
  if (message == WM_KEYDOWN && wParam == VK_ESCAPE && m_cursorConfined) {
    ClipCursor(nullptr);
    m_cursorConfined = false;
    // Don't forward ESC when releasing cursor
    return true;
  }

  if (message == WM_KEYDOWN || message == WM_SYSKEYDOWN ||
      message == WM_KEYUP || message == WM_SYSKEYUP) {
    INPUT input = {};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = static_cast<WORD>(wParam);
    input.ki.wScan = static_cast<WORD>(MapVirtualKeyW(input.ki.wVk, MAPVK_VK_TO_VSC));
    input.ki.dwFlags = KEYEVENTF_SCANCODE;

    switch (wParam) {
      case VK_INSERT:
      case VK_DELETE:
      case VK_HOME:
      case VK_END:
      case VK_PRIOR:
      case VK_NEXT:
      case VK_LEFT:
      case VK_RIGHT:
      case VK_UP:
      case VK_DOWN:
      case VK_RCONTROL:
      case VK_RMENU:
      case VK_LWIN:
      case VK_RWIN:
      case VK_APPS:
        input.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
        break;
      default:
        break;
    }

    if (message == WM_KEYUP || message == WM_SYSKEYUP) {
      input.ki.dwFlags |= KEYEVENTF_KEYUP;
    }

    if (m_outputAutoFocus) {
      SetForegroundWindow(m_captureWindow);
    }
    SendInput(1, &input, sizeof(INPUT));
    return true;
  }

  return false;
}

bool App::HandlePreviewMouse(UINT message, WPARAM wParam, LPARAM lParam) {
  if (m_outputDisplayMode != 1 || !m_captureWindow || m_captureMode != 0) {
    return false;
  }

  if (!m_previewHasImage) {
    return false;
  }

  POINT uiPoint = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
  bool inPreview = IsPreviewPoint(uiPoint);

  if (message == WM_LBUTTONDOWN) {
    if (inPreview) {
      m_previewInputActive = true;
      SetCapture(m_uiHwnd);
    } else {
      m_previewInputActive = false;
      return false;
    }
  }

  if (!m_previewInputActive && !inPreview) {
    return false;
  }

  POINT target = {};
  if (!MapPreviewPoint(uiPoint, target)) {
    return false;
  }

  POINT screenPt = target;
  ClientToScreen(m_captureWindow, &screenPt);

  LONG vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
  LONG vy = GetSystemMetrics(SM_YVIRTUALSCREEN);
  LONG vw = GetSystemMetrics(SM_CXVIRTUALSCREEN);
  LONG vh = GetSystemMetrics(SM_CYVIRTUALSCREEN);
  LONG absX = 0;
  LONG absY = 0;
  if (vw > 1) {
    absX = (screenPt.x - vx) * 65535 / (vw - 1);
  }
  if (vh > 1) {
    absY = (screenPt.y - vy) * 65535 / (vh - 1);
  }

  INPUT inputs[3] = {};
  int count = 0;
  inputs[count].type = INPUT_MOUSE;
  inputs[count].mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
  inputs[count].mi.dx = absX;
  inputs[count].mi.dy = absY;
  count++;

  DWORD buttonFlag = 0;
  if (message == WM_LBUTTONDOWN) {
    buttonFlag = MOUSEEVENTF_LEFTDOWN;
  } else if (message == WM_LBUTTONUP) {
    buttonFlag = MOUSEEVENTF_LEFTUP;
    ReleaseCapture();
  } else if (message == WM_RBUTTONDOWN) {
    buttonFlag = MOUSEEVENTF_RIGHTDOWN;
  } else if (message == WM_RBUTTONUP) {
    buttonFlag = MOUSEEVENTF_RIGHTUP;
  } else if (message == WM_MBUTTONDOWN) {
    buttonFlag = MOUSEEVENTF_MIDDLEDOWN;
  } else if (message == WM_MBUTTONUP) {
    buttonFlag = MOUSEEVENTF_MIDDLEUP;
  }

  if (buttonFlag != 0) {
    inputs[count].type = INPUT_MOUSE;
    inputs[count].mi.dwFlags = buttonFlag;
    count++;
  }

  if (message == WM_MOUSEWHEEL) {
    inputs[count].type = INPUT_MOUSE;
    inputs[count].mi.dwFlags = MOUSEEVENTF_WHEEL;
    inputs[count].mi.mouseData = GET_WHEEL_DELTA_WPARAM(wParam);
    count++;
  } else if (message == WM_MOUSEHWHEEL) {
    inputs[count].type = INPUT_MOUSE;
    inputs[count].mi.dwFlags = MOUSEEVENTF_HWHEEL;
    inputs[count].mi.mouseData = GET_WHEEL_DELTA_WPARAM(wParam);
    count++;
  }

  if (count > 0) {
    SetForegroundWindow(m_captureWindow);
    SendInput(static_cast<UINT>(count), inputs, sizeof(INPUT));
    return true;
  }

  return false;
}

bool App::HandlePreviewKey(UINT message, WPARAM wParam, LPARAM lParam) {
  if (m_outputDisplayMode != 1 || !m_captureWindow || m_captureMode != 0) {
    return false;
  }

  if (!m_previewInputActive) {
    return false;
  }

  if (message == WM_KEYDOWN || message == WM_SYSKEYDOWN ||
      message == WM_KEYUP || message == WM_SYSKEYUP) {
    if (message == WM_KEYDOWN && wParam == VK_ESCAPE) {
      m_previewInputActive = false;
      ReleaseCapture();
      return true;
    }

    INPUT input = {};
    input.type = INPUT_KEYBOARD;
    input.ki.wVk = static_cast<WORD>(wParam);
    input.ki.wScan = static_cast<WORD>(MapVirtualKeyW(input.ki.wVk, MAPVK_VK_TO_VSC));
    input.ki.dwFlags = KEYEVENTF_SCANCODE;

    switch (wParam) {
      case VK_INSERT:
      case VK_DELETE:
      case VK_HOME:
      case VK_END:
      case VK_PRIOR:
      case VK_NEXT:
      case VK_LEFT:
      case VK_RIGHT:
      case VK_UP:
      case VK_DOWN:
      case VK_RCONTROL:
      case VK_RMENU:
      case VK_LWIN:
      case VK_RWIN:
      case VK_APPS:
        input.ki.dwFlags |= KEYEVENTF_EXTENDEDKEY;
        break;
      default:
        break;
    }

    if (message == WM_KEYUP || message == WM_SYSKEYUP) {
      input.ki.dwFlags |= KEYEVENTF_KEYUP;
    }

    SetForegroundWindow(m_captureWindow);
    SendInput(1, &input, sizeof(INPUT));
    return true;
  }

  return false;
}

void App::RestoreDxgiCropWindow() {
  if (!m_dxgiCropModeActive || !m_hwnd) return;
  
  // Restore output window to original position if we moved it
  if (m_dxgiCropOriginalRect.left != 0 || m_dxgiCropOriginalRect.right != 0) {
    SetWindowPos(m_hwnd, HWND_TOPMOST,
                 m_dxgiCropOriginalRect.left,
                 m_dxgiCropOriginalRect.top,
                 m_dxgiCropOriginalRect.right - m_dxgiCropOriginalRect.left,
                 m_dxgiCropOriginalRect.bottom - m_dxgiCropOriginalRect.top,
                 SWP_SHOWWINDOW);
    m_dxgiCropOriginalRect = {};
  }
  
  m_dxgiCropModeActive = false;
}

void App::Update() {
  // Toggle Overlay Hotkey (Default F9)
  bool overlayKeyDown = (GetAsyncKeyState(m_hotkeyToggleOverlay) & 0x8000) != 0;
  if (overlayKeyDown && !m_toggleOverlayKeyState) {
      m_outputWindowVisible = !m_outputWindowVisible;
      ShowWindow(m_hwnd, m_outputWindowVisible ? SW_SHOW : SW_HIDE);
  }
  m_toggleOverlayKeyState = overlayKeyDown;

  // Toggle UI Hotkey (Default F8)
  bool uiKeyDown = (GetAsyncKeyState(m_hotkeyToggleUi) & 0x8000) != 0;
  if (uiKeyDown && !m_toggleUiKeyState) {
      m_uiVisible = !m_uiVisible;
      ShowWindow(m_uiHwnd, m_uiVisible ? SW_SHOW : SW_HIDE);
  }
  m_toggleUiKeyState = uiKeyDown;

   if (m_monitorDirty) {
     SelectMonitor(m_selectedMonitor);
     m_monitorDirty = false;
   }

   // Text Preservation Mode - adjust parameters for reduced text flickering
   if (m_textPreservationMode) {
     m_temporalHistoryWeight = 0.05f;
     m_temporalConfInfluence = 0.3f;
     m_temporalNeighborhoodSize = 1;
     m_motionEdgeScale = 10.0f;
     m_confidencePower = 2.0f;
   }

   // Removed explicit waitable object wait here as it was causing sluggishness.
  // We rely on Present() for pacing (if VSync is on) or run unlocked (if VSync is off).

  UpdateCapture();
}

void App::UpdateCapture() {
  int processed = 0;
  constexpr int kMaxFramesPerUpdate = 180;
  int maxFramesPerUpdate = kMaxFramesPerUpdate;
  bool neverDropMode = m_neverDropFrames;
  int maxQueueSize = neverDropMode ? m_maxQueueSize : 3;
  if (maxQueueSize < 2) {
    maxQueueSize = 2;
  } else if (maxQueueSize > kFrameQueueSize) {
    maxQueueSize = kFrameQueueSize;
  }

  if (m_captureMode == 0 && m_captureWindow) {
    if (!IsWindow(m_captureWindow)) {
      if (m_windowCaptureUsingWgc) {
        m_capture.StopCapture();
        m_windowCaptureUsingWgc = false;
      } else {
        m_dupCapture.StopCapture();
      }
      m_captureWindow = nullptr;
      m_captureWindowMonitor = nullptr;
      m_captureStatus = "Capture stopped (window closed)";
      
      // Restore output window position if it was moved for DXGI Crop mode
      RestoreDxgiCropWindow();
      return;
    }

    bool wantWgc = ShouldUseWgcForWindowCapture();
    if (wantWgc != m_windowCaptureUsingWgc) {
      if (StartWindowCapture(m_captureWindow)) {
        ResetCaptureState();
      }
      return;
    }

    if (!m_windowCaptureUsingWgc && m_dupCapture.IsCapturing()) {
      HMONITOR monitor = MonitorFromWindow(m_captureWindow, MONITOR_DEFAULTTONEAREST);
      if (monitor && monitor != m_captureWindowMonitor) {
        if (StartWindowCapture(m_captureWindow)) {
          ResetCaptureState();
        }
      }
    }
  }

  while (processed < maxFramesPerUpdate) {
    if (neverDropMode && static_cast<int>(m_frameQueue.size()) >= maxQueueSize) {
      break;
    }
    CapturedFrame frame;
    bool gotFrame = false;
    if (m_captureMode == 0) {
      if (m_windowCaptureUsingWgc) {
        gotFrame = m_capture.AcquireNextFrame(frame);
      } else {
        gotFrame = m_dupCapture.AcquireNextFrame(frame);
      }
    } else if (m_captureMode == 2) {
      // Game capture mode (hook-based)
      gotFrame = m_gameCapture.AcquireNextFrame(frame);
    } else if (m_captureMode == 3) {
      // DXGI Crop mode - capture monitor, crop to window
      gotFrame = m_dupCapture.AcquireNextFrame(frame);
      // Cropping will be done later after we verify we got a frame
    } else {
      gotFrame = m_capture.AcquireNextFrame(frame);
    }
    if (!gotFrame) {
      // In high-FPS mode with spin-wait, keep trying to get frames
      // Don't break immediately - spin until we get one or timeout
      if (m_dupCapture.IsSpinWaitMode()) {
        continue;
      }
      break;
    }
    processed++;

    // DXGI Crop mode: crop the captured monitor frame to window client area
    if (m_captureMode == 3 && m_captureWindow && frame.texture) {
      RECT clientRect;
      if (GetClientRect(m_captureWindow, &clientRect)) {
        POINT clientTopLeft = { 0, 0 };
        ClientToScreen(m_captureWindow, &clientTopLeft);
        
        // Get monitor info to calculate offset
        MONITORINFO mi = { sizeof(mi) };
        if (GetMonitorInfo(m_captureWindowMonitor, &mi)) {
          int cropX = clientTopLeft.x - mi.rcMonitor.left;
          int cropY = clientTopLeft.y - mi.rcMonitor.top;
          int cropW = clientRect.right - clientRect.left;
          int cropH = clientRect.bottom - clientRect.top;
          
          // Clamp to monitor bounds
          if (cropX < 0) { cropW += cropX; cropX = 0; }
          if (cropY < 0) { cropH += cropY; cropY = 0; }
          if (cropX > frame.width) cropX = frame.width;        // Safety
          if (cropY > frame.height) cropY = frame.height;      // Safety
          if (cropX + cropW > frame.width) cropW = frame.width - cropX;
          if (cropY + cropH > frame.height) cropH = frame.height - cropY;
          
          // Debug logging (throttled)
          static int s_logCounter = 0;
          if (s_logCounter < 5) {
             std::ofstream log("dxgi_crop_debug.txt", std::ios::app);
             log << "Frame " << s_logCounter << ": CropX=" << cropX << " CropY=" << cropY 
                 << " CropW=" << cropW << " CropH=" << cropH 
                 << " FrameW=" << frame.width << " FrameH=" << frame.height << "\n";
             
             D3D11_TEXTURE2D_DESC srcDesc = {};
             frame.texture->GetDesc(&srcDesc);
             log << "SrcFormat=" << srcDesc.Format << "\n";
             log.close();
             s_logCounter++;
          }

          if (cropW > 0 && cropH > 0) {
            D3D11_TEXTURE2D_DESC srcDesc = {};
            frame.texture->GetDesc(&srcDesc);

            // Check if current texture matches needed size and format
            bool needRecreate = !m_cropTexture;
            if (m_cropTexture) {
                D3D11_TEXTURE2D_DESC currentDesc = {};
                m_cropTexture->GetDesc(&currentDesc);
                if (currentDesc.Width != static_cast<UINT>(cropW) || 
                    currentDesc.Height != static_cast<UINT>(cropH) ||
                    currentDesc.Format != srcDesc.Format) {
                    needRecreate = true;
                }
            }

            if (needRecreate) {
              D3D11_TEXTURE2D_DESC desc = {};
              desc.Width = cropW;
              desc.Height = cropH;
              desc.MipLevels = 1;
              desc.ArraySize = 1;
              desc.Format = srcDesc.Format; // Use Source Format!
              desc.SampleDesc.Count = 1;
              desc.Usage = D3D11_USAGE_DEFAULT;
              desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
              
              HRESULT hr = m_device.Device()->CreateTexture2D(&desc, nullptr, &m_cropTexture);
              if (FAILED(hr)) {
                   m_cropTexture.Reset();
              }
              m_cropWidth = cropW;
              m_cropHeight = cropH;
            }
            
            if (m_cropTexture) {
                // Copy the cropped region
                D3D11_BOX srcBox = {};
                srcBox.left = cropX;
                srcBox.top = cropY;
                srcBox.right = cropX + cropW;
                srcBox.bottom = cropY + cropH;
                srcBox.front = 0;
                srcBox.back = 1;
                
                m_device.Context()->CopySubresourceRegion(
                  m_cropTexture.Get(), 0, 0, 0, 0,
                  frame.texture.Get(), 0, &srcBox);
                
                // Replace frame texture with cropped version
                frame.texture = m_cropTexture;
                frame.width = cropW;
                frame.height = cropH;
            }
          }
        }
      }
    }

    if (frame.width != m_frameWidth || frame.height != m_frameHeight) {
      ResizeForCapture(frame.width, frame.height);
    }

    if (!frame.texture) {
      continue;
    }

    if (!m_frameTextures[0]) {
      continue;
    }

    if (m_currFrameTime100ns != 0) {
      m_prevFrameTime100ns = m_currFrameTime100ns;
    }
    m_currFrameTime100ns = frame.systemTime100ns;

    // Moved Timestamps Logic UP for access in averaging
    m_frameTimestamps.push_back(frame.systemTime100ns);
    if (m_frameTimestamps.size() > 360) {
      m_frameTimestamps.erase(m_frameTimestamps.begin());
    }

    if (m_prevFrameTime100ns != 0 && m_currFrameTime100ns != m_prevFrameTime100ns) {
      // SMOOTHNESS: Use precise Sliding Window Average of last 20 frames
      if (m_frameTimestamps.size() >= 21) {
          constexpr size_t kWindowSize = 20;
          size_t idxStart = m_frameTimestamps.size() - 1 - kWindowSize;
          int64_t diffNs = m_frameTimestamps.back() - m_frameTimestamps[idxStart];
          if (diffNs > 0) {
             m_avgFrameInterval = (static_cast<double>(diffNs) * 1e-7) / static_cast<double>(kWindowSize);
          }
      } else {
         double interval = static_cast<double>(m_currFrameTime100ns - m_prevFrameTime100ns) * 1e-7;
         if (m_avgFrameInterval <= 0.0) m_avgFrameInterval = interval;
         else m_avgFrameInterval = m_avgFrameInterval * 0.9 + interval * 0.1;
      }
    }

    if (m_qpcFreq.QuadPart > 0 && frame.qpcTime != 0) {
      double qpcTo100ns = 1e7 / static_cast<double>(m_qpcFreq.QuadPart);
      // SYNC: Calculate offset between SystemTime (WGC) and QPC (Present)
      double offset = static_cast<double>(frame.systemTime100ns) -
                      static_cast<double>(frame.qpcTime) * qpcTo100ns;
      
      if (!m_timeOffsetValid) {
        m_timeOffset100ns = offset;
        m_timeOffsetValid = true;
      } else {
        // STABILITY: Extremely stiff filter (0.005) or virtually locked.
        // Clock drift is tiny. Frame jitter is high. We MUST ignore frame jitter.
        m_timeOffset100ns = m_timeOffset100ns * 0.995 + offset * 0.005;
      }
    }

    if (!neverDropMode && static_cast<int>(m_frameQueue.size()) >= maxQueueSize) {
      while (static_cast<int>(m_frameQueue.size()) >= maxQueueSize) {
        m_frameQueue.pop_front();
      }
    }

    int slot = m_queueWrite;
    m_queueWrite = (m_queueWrite + 1) % kFrameQueueSize;

    m_device.Context()->CopyResource(m_frameTextures[slot].Get(), frame.texture.Get());
    
    // DE-JITTER: Virtualize the timestamp (Improved V2)
    int64_t smoothedTime = frame.systemTime100ns;
    
    if (m_avgFrameInterval > 0.0 && m_lastSmoothedTime > 0) {
        int64_t expectedInterval = static_cast<int64_t>(m_avgFrameInterval * 1e7);
        int64_t expectedTime = m_lastSmoothedTime + expectedInterval;
        int64_t diff = std::abs(frame.systemTime100ns - expectedTime);
        
        // Tolerance: Adjusted by user (default 20%)
        // This swallows almost all Windows scheduling jitter
        double tolerance = static_cast<double>(expectedInterval) * static_cast<double>(m_jitterSuppression);
        if (m_jitterSuppression > 0.0f && static_cast<double>(diff) < tolerance) {
            smoothedTime = expectedTime;
        } else {
            // If it's a genuine drop/spike, accept it to avoid desync
            // But blend it slightly to soften the blow
            smoothedTime = (frame.systemTime100ns + expectedTime) / 2;
        }
    }
    
    m_lastSmoothedTime = smoothedTime;
    m_frameTime100ns[slot] = smoothedTime;
    m_frameQueue.push_back(slot);

    // Timestamps update moved up

    m_captureFrameCount++;
    double nowSec = frame.systemTime100ns * 1e-9;
    if (m_captureFpsTime > 0.0) {
      double elapsed = nowSec - m_captureFpsTime;
      if (elapsed >= 1.0) {
        m_captureFps = static_cast<float>(m_captureFrameCount / elapsed);
        m_captureFrameCount = 0;
        m_captureFpsTime = nowSec;
      }
    } else {
      m_captureFpsTime = nowSec;
    }

    if (m_currFrameTime100ns > 0 && m_prevFrameTime100ns > 0) {
      double intervalNs = static_cast<double>(m_currFrameTime100ns - m_prevFrameTime100ns);
      if (intervalNs > 0.0) {
        double intervalMs = intervalNs * 1e-7;
        m_frameIntervalSum += intervalMs;
        m_frameIntervalCount++;
        if (intervalMs < m_minFrameInterval) {
          m_minFrameInterval = static_cast<float>(intervalMs);
        }
        if (intervalMs > m_maxFrameInterval) {
          m_maxFrameInterval = static_cast<float>(intervalMs);
        }
      }
    }

    if (neverDropMode && static_cast<int>(m_frameQueue.size()) > maxQueueSize) {
      m_frameQueue.pop_front();
    }
  }
}

void App::Render() {
  auto* context = m_device.Context();
  if (!context || !m_device.SwapChain()) {
    return;
  }
  
  HANDLE waitHandle = m_device.GetSwapChainWaitHandle();
  if (waitHandle) {
      WaitForSingleObjectEx(waitHandle, 1000, true);
  }

  // Only show overlay when: overlay mode ON, capturing a window, AND we have frames
  bool hasCapture = (m_captureWindow != nullptr && m_captureMode == 0 && m_frameWidth > 0 && m_frameHeight > 0);
  bool wantOverlay = (m_overlayMode && hasCapture);
  
  // In overlay mode, position window over capture window
  if (wantOverlay) {
    UpdateOutputOverlayWindow();
    
    // Force WS_EX_TRANSPARENT style to ensure click-through
    LONG_PTR exStyle = GetWindowLongPtr(m_hwnd, GWL_EXSTYLE);
    if (!(exStyle & WS_EX_TRANSPARENT)) {
        SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, exStyle | WS_EX_TRANSPARENT | WS_EX_LAYERED);
    }
  }

  // Ensure window is transparent to mouse clicks in overlay mode by disabling it
  // This forces Windows to route all input to the window underneath (the game)
  if (m_hwnd) {
    bool shouldBeEnabled = !wantOverlay;
    // Only change state if different to prevent unnecessary updates
    if (static_cast<bool>(IsWindowEnabled(m_hwnd)) != shouldBeEnabled) {
      EnableWindow(m_hwnd, shouldBeEnabled ? TRUE : FALSE);
    }
  }

  UpdateOutputWindowMode();
  bool wantOutputWindow = wantOverlay || (!m_overlayMode && (m_outputDisplayMode == 0 || m_outputDisplayMode == 2));
  if (m_hwnd && wantOutputWindow != m_outputWindowVisible) {
    ShowWindow(m_hwnd, wantOutputWindow ? SW_SHOWNOACTIVATE : SW_HIDE);
    m_outputWindowVisible = wantOutputWindow;
  }

  if (m_outputWidth <= 0 || m_outputHeight <= 0) {
    return;
  }

  int multiplier = (m_outputMultiplier < 1) ? 1 : m_outputMultiplier;
  double captureFps = (m_avgFrameInterval > 0.0) ? (1.0 / m_avgFrameInterval) : 0.0;
  bool lowFpsSource = (captureFps > 0.0 && captureFps < 30.0);
  float monitorHz = m_device.RefreshHz(m_selectedMonitor);
  bool useMonitorSync = (m_outputMode == 1);

  if (useMonitorSync && monitorHz > 0.0f) {
    m_targetFps = monitorHz;
  } else if (m_avgFrameInterval > 0.0) {
    m_targetFps = static_cast<float>(static_cast<double>(multiplier) / m_avgFrameInterval);
  } else {
    m_targetFps = 0.0f;
  }
  // Sub-30 FPS sources benefit from stable pacing near display refresh instead of very high multiplier targets.
  if (!useMonitorSync && lowFpsSource && monitorHz > 0.0f && m_targetFps > monitorHz) {
    m_targetFps = monitorHz;
  }

  LARGE_INTEGER now = {};
  QueryPerformanceCounter(&now);
  double freq = (m_qpcFreq.QuadPart > 0) ? static_cast<double>(m_qpcFreq.QuadPart) : 0.0;
  double nowTime100ns = 0.0;
  if (freq > 0.0) {
    double qpcTo100ns = 1e7 / freq;
    if (m_timeOffsetValid) {
      nowTime100ns = static_cast<double>(now.QuadPart) * qpcTo100ns + m_timeOffset100ns;
    } else {
      nowTime100ns = static_cast<double>(now.QuadPart) * qpcTo100ns;
    }
  }
  int64_t intervalQpc = 0;
  
  bool limitOutput = m_limitOutputFps && !useMonitorSync;
  if (limitOutput && m_targetFps > 0.0f && freq > 0.0) {
    // Use precise floating point interval from target FPS (do not snap to integer)
    double targetFps = static_cast<double>(m_targetFps);
    intervalQpc = static_cast<int64_t>(freq / targetFps);
    if (intervalQpc < 1) {
      intervalQpc = 1;
    }
    if (m_nextOutputQpc == 0) {
      m_nextOutputQpc = now.QuadPart;
    } else {
        if (now.QuadPart > m_nextOutputQpc + intervalQpc * 2) {
             m_nextOutputQpc = now.QuadPart;
        }
    }
    
    m_nextOutputQpc += intervalQpc;
    
    int64_t remainingQpc = m_nextOutputQpc - now.QuadPart;
    if (remainingQpc > 0) {
         double seconds = static_cast<double>(remainingQpc) / freq;
         int64_t hundredsNs = static_cast<int64_t>(seconds * 10000000.0);
         
         if (hundredsNs > 5000) { // Sleep if > 0.5ms using precise timer
             LARGE_INTEGER performWait = {};
             performWait.QuadPart = -hundredsNs;
             SetWaitableTimer(m_waitTimer, &performWait, 0, nullptr, nullptr, 0);
             WaitForSingleObject(m_waitTimer, INFINITE);
         } else {
             while(now.QuadPart < m_nextOutputQpc) {
                 YieldProcessor();
                 QueryPerformanceCounter(&now);
             }
         }
         QueryPerformanceCounter(&now);
         
         // Recalculate time after wait
         if (freq > 0.0) {
            double qpcTo100ns = 1e7 / freq;
            if (m_timeOffsetValid) {
                nowTime100ns = static_cast<double>(now.QuadPart) * qpcTo100ns + m_timeOffset100ns;
            } else {
                nowTime100ns = static_cast<double>(now.QuadPart) * qpcTo100ns;
            }
         }
    }
  } else {
    m_nextOutputQpc = 0;
  }

  bool neverDropMode = m_neverDropFrames;
  int maxQueueSize = neverDropMode ? m_maxQueueSize : (lowFpsSource ? 4 : 3);
  if (maxQueueSize < 2) {
    maxQueueSize = 2;
  } else if (maxQueueSize > kFrameQueueSize) {
    maxQueueSize = kFrameQueueSize;
  }
  if (!neverDropMode) {
    while (m_frameQueue.size() > static_cast<size_t>(maxQueueSize)) {
      m_frameQueue.pop_front();
    }
  }

  ID3D11Texture2D* output = nullptr;
  m_lastAlpha = 1.0f;
  m_lastInterpolated = false;
  m_outputDelayMs = 0.0f;
  m_lastUnstable = false;

  if (!m_frameQueue.empty()) {
    if (multiplier != m_lastMultiplier) {
      m_lastMultiplier = multiplier;
      m_outputStepIndex = 0;
      m_holdEndFrame = false;
      m_pairPrevSlot = -1;
      m_pairCurrSlot = -1;
      m_pairPrevTime100ns = 0;
      m_pairCurrTime100ns = 0;
    }

    double delayScale = (m_delayScale < 0.25f) ? 0.25f : m_delayScale;
    double delay = (m_avgFrameInterval > 0.0) ? (m_avgFrameInterval * delayScale) : 0.0;
    if (m_adaptiveDelay && neverDropMode && m_avgFrameInterval > 0.0) {
      int targetDepth = m_targetQueueDepth;
      if (targetDepth < 2) {
        targetDepth = 2;
      } else if (targetDepth > kFrameQueueSize) {
        targetDepth = kFrameQueueSize;
      }
      double depthError = static_cast<double>(targetDepth - static_cast<int>(m_frameQueue.size()));
      double adjust = depthError * (m_avgFrameInterval * 0.35);
      double maxAdjust = m_avgFrameInterval * 3.0;
      if (adjust > maxAdjust) {
        adjust = maxAdjust;
      } else if (adjust < -maxAdjust) {
        adjust = -maxAdjust;
      }
      delay += adjust;
      if (delay < 0.0) {
        delay = 0.0;
      }
    }
    m_outputDelayMs = static_cast<float>(delay * 1000.0);
    
    // PREDICTED TIME: If pacing is active, use the *intended* output time for alpha calculation.
    // This decouples thread scheduling jitter from the visual animation state.
     double displayTime100ns = 0.0;
    if (limitOutput && m_nextOutputQpc > 0 && freq > 0.0) {
        double qpcTo100ns = 1e7 / freq;
        double predictedNext100ns = 0.0;
        if (m_timeOffsetValid) {
            predictedNext100ns = static_cast<double>(m_nextOutputQpc) * qpcTo100ns + m_timeOffset100ns;
        } else {
            predictedNext100ns = static_cast<double>(m_nextOutputQpc) * qpcTo100ns;
        }
        displayTime100ns = predictedNext100ns - delay * 1e7;
    } else {
        displayTime100ns = nowTime100ns - delay * 1e7;
    }

    if (displayTime100ns < 0.0) {
      displayTime100ns = 0.0;
    }

    if (freq > 0.0) {
      while (m_frameQueue.size() >= 2) {
        int p = m_frameQueue[0];
        int c = m_frameQueue[1];
        double pTime = static_cast<double>(m_frameTime100ns[p]);
        double cTime = static_cast<double>(m_frameTime100ns[c]);
        if (cTime <= pTime) {
          m_frameQueue.pop_front();
          continue;
        }
        if (displayTime100ns >= cTime) {
          if (!neverDropMode) {
            m_frameQueue.pop_front();
            continue;
          }
          break;
        }
        
        if (!neverDropMode && !lowFpsSource && m_frameQueue.size() > 2) {
          int nextSlot = m_frameQueue[2];
          double nextTime = static_cast<double>(m_frameTime100ns[nextSlot]);
          if (nextTime > cTime && (nextTime - cTime) < (m_avgFrameInterval * 0.8)) {
            m_frameQueue.pop_front();
            continue;
          }
        }
        break;
      }
    }

    int prevSlot = m_frameQueue.front();
    int currSlot = (m_frameQueue.size() >= 2) ? m_frameQueue[1] : prevSlot;
    bool hasPair = (m_frameQueue.size() >= 2);
    bool hasPrevSrv = (m_frameSrvs[prevSlot] != nullptr);
    bool hasCurrSrv = (m_frameSrvs[currSlot] != nullptr);
    if (hasPair) {
      int64_t prevTime100ns = m_frameTime100ns[prevSlot];
      int64_t currTime100ns = m_frameTime100ns[currSlot];
      bool pairChanged = (prevSlot != m_pairPrevSlot) ||
                         (currSlot != m_pairCurrSlot) ||
                         (prevTime100ns != m_pairPrevTime100ns) ||
                         (currTime100ns != m_pairCurrTime100ns);
      if (pairChanged) {
        m_pairPrevSlot = prevSlot;
        m_pairCurrSlot = currSlot;
        m_pairPrevTime100ns = prevTime100ns;
        m_pairCurrTime100ns = currTime100ns;
        m_outputStepIndex = 0;
        m_interpolator.ResetTemporal();
      }
    } else {
      m_pairPrevSlot = -1;
      m_pairCurrSlot = -1;
      m_pairPrevTime100ns = 0;
      m_pairCurrTime100ns = 0;
    }
    bool canInterpolate = m_interpolationEnabled && hasPair && hasPrevSrv && hasCurrSrv;
    bool needScale = (m_outputWidth != m_frameWidth) || (m_outputHeight != m_frameHeight);

    float alpha = 1.0f;
    double intervalSec = 0.0;
    double useInterval = 0.0;
    int stepCount = (canInterpolate && multiplier > 0) ? multiplier : 1;
    if (stepCount < 1) {
      stepCount = 1;
    }
    if (hasPair) {
      double prevTime = static_cast<double>(m_frameTime100ns[prevSlot]);
      double currTime = static_cast<double>(m_frameTime100ns[currSlot]);
      intervalSec = (currTime - prevTime) * 1e-7;
      
      // IMPOROVED JITTER HANDLING:
      // Instead of hard snapping (which causes stutters when hovering near the threshold),
      // we now use a "Soft Knee" blend.
      // - Inside threshold: Locked to Average (Butter Smooth)
      // - Just outside: Blends smoothly to Real Time (Absorbs drift)
      // - Way outside: Uses Real Time (Responding to lag spikes)
      useInterval = intervalSec;
      
      if (m_avgFrameInterval > 0.0) {
          if (lowFpsSource) {
              // For low-FPS capture, lock to the running average interval to avoid visible cadence jitter.
              useInterval = m_avgFrameInterval;
          } else if (m_forceInterpolation) {
              useInterval = m_avgFrameInterval;
          } else if (intervalSec > 0.0) {
              double diff = std::abs(intervalSec - m_avgFrameInterval);
              double errorRatio = diff / m_avgFrameInterval;
              double limit = static_cast<double>(m_jitterSuppression);
              
              if (limit > 0.001) { // Avoid divide by zero
                  if (errorRatio <= limit) {
                      // Perfect Lock
                      useInterval = m_avgFrameInterval;
                  } else if (errorRatio < (limit * 2.0)) {
                      // Soft Blend Zone: Fade from Average to Real
                      // This eliminates the "Pop" artifact when jitter increases slightly
                      double blendFactor = (errorRatio - limit) / limit;
                      useInterval = m_avgFrameInterval * (1.0 - blendFactor) + intervalSec * blendFactor;
                  }
              }
          }
      }

      if (useInterval > 0.0) {
        if (neverDropMode) {
          if (m_outputStepIndex < 0 || m_outputStepIndex > stepCount) {
            m_outputStepIndex = 0;
          }
          alpha = static_cast<float>(m_outputStepIndex) / static_cast<float>(stepCount);
        } else {
          double t = (displayTime100ns - prevTime) * 1e-7;
          if (t < 0.0) {
            t = 0.0;
          }
          if (canInterpolate) {
            // UNLOCKED SMOOTHNESS:
            // Instead of stepping (0.0, 0.5, 1.0), calculate exact alpha based on time.
            // This eliminates judder when Monitor Hz != Target Hz.
            float rawAlpha = static_cast<float>(t / useInterval);
            if (rawAlpha < 0.0f) rawAlpha = 0.0f;
            if (rawAlpha > 1.0f) rawAlpha = 1.0f;

            // In multiplier mode, quantize alpha to stable sub-steps.
            // This removes phase jitter that is very visible at 2x.
            // Low-FPS sources look smoother with continuous alpha (no multiplier quantization).
            bool lockAlphaToMultiplier = (!useMonitorSync && multiplier > 1 && !lowFpsSource);
            if (lockAlphaToMultiplier) {
              int quantStep = static_cast<int>(rawAlpha * static_cast<float>(multiplier));
              if (quantStep < 0) {
                quantStep = 0;
              } else if (quantStep > multiplier) {
                quantStep = multiplier;
              }
              alpha = static_cast<float>(quantStep) / static_cast<float>(multiplier);
              m_outputStepIndex = quantStep;
            } else {
              alpha = rawAlpha;
              m_outputStepIndex = static_cast<int>(alpha * static_cast<float>(multiplier));
            }
          } else {
            alpha = static_cast<float>(t / useInterval);
            if (alpha < 0.0f) {
              alpha = 0.0f;
            } else if (alpha > 1.0f) {
              alpha = 1.0f;
            }
            m_outputStepIndex = static_cast<int>(alpha * static_cast<float>(multiplier));
          }
        }
      }
    } else {
      m_outputStepIndex = 0;
    }

    // Fallback if needed
    if (useInterval <= 0.0 && intervalSec > 0.0) useInterval = intervalSec; 
    
    if (alpha < 0.0f) {
      alpha = 0.0f;
    } else if (alpha > 1.0f) {
      alpha = 1.0f;
    }

    bool unstable = false;
    if (hasPair && m_qpcFreq.QuadPart > 0 && intervalSec > 0.0) {
      m_lastIntervalMs = static_cast<float>(useInterval * 1000.0);
      if (m_avgFrameInterval > 0.0) {
        m_lastAvgIntervalMs = static_cast<float>(m_avgFrameInterval * 1000.0);
      }
      if (m_lowLatencyMode && m_avgFrameInterval > 0.0) {
        double delta = std::abs(intervalSec - m_avgFrameInterval);
        unstable = delta > (m_avgFrameInterval * 0.5);
      }
    }

    bool allowInterpolation = canInterpolate;

    m_lastUnstable = unstable;
    m_lastAlpha = alpha;
    
    // WORKLOAD SMOOTHING:
    // If we skip interpolation at alpha=0 or alpha=1, the GPU load fluctuates wildly (0% vs 90%).
    // This causes "Spikes" in the frame time graph and thermal throttling issues.
    // We now report "Interpolated" as true even if alpha is 0/1, so we run the shader pipeline.
    // This keeps the GPU clock high and frame times consistent (e.g. always 8ms instead of alternating 1ms/8ms).
    m_lastInterpolated = allowInterpolation; 

    if (allowInterpolation && m_outputStepIndex >= multiplier) {
      m_holdEndFrame = true;
    }

    auto debugMode = static_cast<Interpolator::DebugViewMode>(m_debugView);
    if (debugMode != Interpolator::DebugViewMode::None && hasCurrSrv) {
      ID3D11ShaderResourceView* currSrv = m_frameSrvs[currSlot].Get();
      ID3D11ShaderResourceView* prevSrv = hasPrevSrv ? m_frameSrvs[prevSlot].Get() : currSrv;
      if (!hasPrevSrv && (debugMode == Interpolator::DebugViewMode::MotionFlow ||
                          debugMode == Interpolator::DebugViewMode::ConfidenceHeatmap ||
                          debugMode == Interpolator::DebugViewMode::ResidualError)) {
        debugMode = Interpolator::DebugViewMode::None;
      }
      m_interpolator.SetMotionModel(m_motionModel);
      m_interpolator.SetMotionSmoothing(m_motionEdgeScale, m_confidencePower);
      m_interpolator.SetTemporalStabilization(m_temporalStabilization,
                                              m_temporalHistoryWeight,
                                              m_temporalConfInfluence,
                                              m_temporalNeighborhoodSize);
      m_interpolator.Debug(prevSrv, currSrv, debugMode, m_debugMotionScale, m_debugDiffScale);
      output = m_interpolator.OutputTexture();
    } else if (allowInterpolation) {
        // Keep alpha in sync with output timing.
        // In never-drop mode we keep the explicit step alpha.
        if (!neverDropMode) {
          double prevTime = static_cast<double>(m_frameTime100ns[prevSlot]);
          double t = (displayTime100ns - prevTime) * 1e-7;
          if (t < 0.0) {
            t = 0.0;
          }
          float rawAlpha = 0.0f;
          if (useInterval > 0.0) {
            rawAlpha = static_cast<float>(t / useInterval);
          }
          if (rawAlpha < 0.0f) rawAlpha = 0.0f;
          if (rawAlpha > 1.0f) rawAlpha = 1.0f;

          bool lockAlphaToMultiplier = (!useMonitorSync && multiplier > 1 && !lowFpsSource);
          if (lockAlphaToMultiplier) {
            int quantStep = static_cast<int>(rawAlpha * static_cast<float>(multiplier));
            if (quantStep < 0) {
              quantStep = 0;
            } else if (quantStep > multiplier) {
              quantStep = multiplier;
            }
            alpha = static_cast<float>(quantStep) / static_cast<float>(multiplier);
            m_outputStepIndex = quantStep;
          } else {
            alpha = rawAlpha;
          }
        }
        m_lastAlpha = alpha;
        
        m_interpolator.SetMotionModel(m_motionModel);
        m_interpolator.SetMotionSmoothing(m_motionEdgeScale, m_confidencePower);
        m_interpolator.SetQualityMode(m_interpolationQuality);
        m_interpolator.SetTemporalStabilization(m_temporalStabilization,
                                                m_temporalHistoryWeight,
                                                m_temporalConfInfluence,
                                                m_temporalNeighborhoodSize);
        m_interpolator.SetMotionVectorPrediction(m_useMotionPrediction);
        float textProtect = m_textPreservationMode ? m_textPreservationStrength : 0.0f;
        m_interpolator.SetTextPreservation(textProtect, m_textPreservationEdgeThreshold);
        
        // ALWAYS EXECUTE if allowInterpolation is true.
        // Even if alpha is 0.0 or 1.0, we want the GPU to do the work.
        // This ensures the pipeline (including motion history) stays valid and hot.
        m_interpolator.Execute(m_frameSrvs[prevSlot].Get(), m_frameSrvs[currSlot].Get(), alpha);
        
        output = m_interpolator.OutputTexture();
    } else {
      if (needScale && hasCurrSrv) {
        m_interpolator.Blit(m_frameSrvs[currSlot].Get());
        output = m_interpolator.OutputTexture();
      } else {
        output = m_frameTextures[currSlot].Get();
      }
    }

    if (neverDropMode && hasPair) {
      int advanceSteps = (allowInterpolation && multiplier > 1) ? multiplier : 1;
      if (advanceSteps < 1) {
        advanceSteps = 1;
      }
      if (!allowInterpolation) {
        m_frameQueue.pop_front();
        m_outputStepIndex = 0;
      } else if (m_outputStepIndex >= advanceSteps) {
        m_outputStepIndex = 0;
        m_frameQueue.pop_front();
      } else {
        m_outputStepIndex++;
      }
    }
  }

  ID3D11ShaderResourceView* outputSrv = nullptr;
  int outputWidth = 0;
  int outputHeight = 0;
  if (output) {
    if (output == m_interpolator.OutputTexture()) {
      outputSrv = m_interpolator.OutputSrv();
      outputWidth = m_outputWidth;
      outputHeight = m_outputHeight;
    } else {
      for (int i = 0; i < kFrameQueueSize; ++i) {
        if (output == m_frameTextures[i].Get()) {
          outputSrv = m_frameSrvs[i].Get();
          outputWidth = m_frameWidth;
          outputHeight = m_frameHeight;
          break;
        }
      }
    }
  }

  if (outputSrv && outputWidth > 0 && outputHeight > 0) {
    m_lastOutputSrv = outputSrv;
    m_lastOutputWidth = outputWidth;
    m_lastOutputHeight = outputHeight;
  } else {
    m_lastOutputSrv.Reset();
    m_lastOutputWidth = 0;
    m_lastOutputHeight = 0;
  }

  if (m_outputDisplayMode == 0 || m_outputDisplayMode == 2) {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
    if (SUCCEEDED(m_device.SwapChain()->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
      if (output) {
        context->CopyResource(backBuffer.Get(), output);
      } else if (m_device.RenderTargetView()) {
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        context->ClearRenderTargetView(m_device.RenderTargetView(), clearColor);
      }
    }

    ID3D11RenderTargetView* rtv = m_device.RenderTargetView();
    context->OMSetRenderTargets(1, &rtv, nullptr);

    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(m_outputWidth);
    viewport.Height = static_cast<float>(m_outputHeight);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    context->RSSetViewports(1, &viewport);
  }

  if (GetAsyncKeyState(VK_F1) & 1) {
    m_showUi = !m_showUi;
    if (m_uiHwnd) {
      if (m_showUi) {
        ShowWindow(m_uiHwnd, SW_SHOW);
        // Bring UI to front and make topmost (above overlay)
        SetWindowPos(m_uiHwnd, HWND_TOPMOST, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
        SetForegroundWindow(m_uiHwnd);
        m_uiTopmost = true;
      } else {
        ShowWindow(m_uiHwnd, SW_HIDE);
        m_uiTopmost = false;
      }
      m_uiVisible = m_showUi;
    }
  }

  if (m_outputDisplayMode == 0 || m_outputDisplayMode == 2) {
    bool presentVsync = m_useVsync || useMonitorSync;
    
    // Force sync interval 0 if unlocked app fps is on
    if (m_unlockAppFps) {
        presentVsync = false;
    }
    
    UINT syncInterval = presentVsync ? 1 : 0;
    UINT presentFlags = 0;
    if (!presentVsync && m_device.AllowTearing()) {
      presentFlags |= DXGI_PRESENT_ALLOW_TEARING;
    }
    
    // If we're not syncing, use 'Do Not Wait' to prevent blocking
    if (m_unlockAppFps && m_device.AllowTearing()) {
        presentFlags |= DXGI_PRESENT_DO_NOT_WAIT;
    }

    HRESULT hr = m_device.SwapChain()->Present(syncInterval, presentFlags);
    
    // If DO_NOT_WAIT dropped the frame, it's fine, we'll try next loop
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        // Just continue
        // Sleep slightly to yield if we are hammering the GPU
        if (m_unlockAppFps) {
             Sleep(1);
        }
    } else if (m_unlockAppFps) {
        // Even on success, yield briefly if unstable
        // YieldProcessor(); or Sleep(0) better than hard spin
    }

    if (m_qpcFreq.QuadPart > 0) {
      LARGE_INTEGER now = {};
      QueryPerformanceCounter(&now);
      if (m_lastPresentQpc != 0) {
        double interval = static_cast<double>(now.QuadPart - m_lastPresentQpc) /
                          static_cast<double>(m_qpcFreq.QuadPart);
        if (interval > 0.0) {
          if (m_presentAvgInterval <= 0.0) {
            m_presentAvgInterval = interval;
          } else {
            m_presentAvgInterval = m_presentAvgInterval * 0.9 + interval * 0.1;
          }
          m_presentFps = static_cast<float>(1.0 / m_presentAvgInterval);
        }
      }
      m_lastPresentQpc = now.QuadPart;
    }
  } else {
    m_presentFps = 0.0f;
    m_presentAvgInterval = 0.0;
    m_lastPresentQpc = 0;
  }

  if (intervalQpc > 0 && limitOutput) {
     // Intentionally empty, m_nextOutputQpc already updated
  }
}

void App::RenderUiWindow() {
  if (!m_uiSwapChain || !m_uiRtv || !m_uiVisible || !m_showUi) {
    return;
  }

  if (!IsWindowVisible(m_uiHwnd)) {
    return;
  }

  auto* context = m_device.Context();
  if (!context) {
    return;
  }

  ID3D11RenderTargetView* rtv = m_uiRtv.Get();
  context->OMSetRenderTargets(1, &rtv, nullptr);

  D3D11_VIEWPORT viewport = {};
  viewport.Width = static_cast<float>(m_uiWidth);
  viewport.Height = static_cast<float>(m_uiHeight);
  viewport.MinDepth = 0.0f;
  viewport.MaxDepth = 1.0f;
  context->RSSetViewports(1, &viewport);

  float clearColor[4] = {0.06f, 0.06f, 0.06f, 1.0f};
  context->ClearRenderTargetView(rtv, clearColor);

  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(static_cast<float>(m_uiWidth), static_cast<float>(m_uiHeight));

  m_ui.BeginFrame();
  RenderUi();
  m_ui.Render();

  UINT uiSyncInterval = 0;
  UINT uiFlags = DXGI_PRESENT_DO_NOT_WAIT;
  if (m_device.AllowTearing()) {
    uiFlags |= DXGI_PRESENT_ALLOW_TEARING;
  }
  HRESULT uiPresent = m_uiSwapChain->Present(uiSyncInterval, uiFlags);
  if (uiPresent == DXGI_STATUS_OCCLUDED || uiPresent == DXGI_ERROR_WAS_STILL_DRAWING) {
    return;
  }
}

void App::UpdateUiSwapChain(UINT width, UINT height) {
  if (!m_uiSwapChain || width == 0 || height == 0) {
    return;
  }

  m_uiRtv.Reset();
  UINT flags = m_device.AllowTearing() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
  if (SUCCEEDED(m_uiSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, flags))) {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
    if (SUCCEEDED(m_uiSwapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)))) {
      m_device.Device()->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_uiRtv);
    }
    m_uiWidth = static_cast<int>(width);
    m_uiHeight = static_cast<int>(height);
  }
}

void App::RenderUi() {
  ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f), ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.90f);
  ImGui::Begin("FrameGen Control Panel", nullptr,
               ImGuiWindowFlags_AlwaysAutoResize |
               ImGuiWindowFlags_NoCollapse |
               ImGuiWindowFlags_NoSavedSettings | 
               ImGuiWindowFlags_NoTitleBar); // Custom Title Bar

  // =========================================================================
  // APP LOGO & HEADER
  // =========================================================================
  {
     ImDrawList* drawList = ImGui::GetWindowDrawList();
     ImVec2 p = ImGui::GetCursorScreenPos();
     float width = ImGui::GetContentRegionAvail().x;
     
     // Background Gradient for Header
     drawList->AddRectFilledMultiColor(
         p, 
         ImVec2(p.x + width, p.y + 40), 
         IM_COL32(0, 100, 200, 255), // Top Left (Blue)
         IM_COL32(0, 0, 0, 0),       // Top Right (Transparent)
         IM_COL32(0, 0, 0, 0),       // Bot Right
         IM_COL32(0, 50, 100, 150)   // Bot Left
     );
     
     ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 255, 255));
     ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 5);
     ImGui::Text("  FRAME GENERATION");
     ImGui::SameLine();
     ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.5f, 1.0f), "UNLOCKED //");
     ImGui::PopStyleColor();
     
     ImGui::SetCursorPosY(ImGui::GetCursorPosY() - 5);
     ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "    True Motion Fidelity Engine");
     
     ImGui::Dummy(ImVec2(0, 15)); // Spacer
     ImGui::Separator();
     ImGui::Spacing();
  }

  // =========================================================================
  // MODERN STARTUP ALERT
  // =========================================================================
  if (m_showStartupAlert) {
      ImGui::OpenPopup("BETA NOTICE");
  }

  // Center the popup
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

  if (ImGui::BeginPopupModal("BETA NOTICE", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration)) {
      
      // Responsive width: 90% of viewport or max 450px
      float viewportWidth = ImGui::GetMainViewport()->Size.x;
      float wrapWidth = (viewportWidth > 500.0f) ? 450.0f : (viewportWidth * 0.9f);

      ImGui::PushTextWrapPos(wrapWidth);
      
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 200, 0, 255));
      ImGui::Text("WARNING: EXPERIMENTAL FEATURES ACTIVE");
      ImGui::PopStyleColor();
      
      ImGui::Separator();
      ImGui::Spacing();
      
      ImGui::Text("DXGI Capture (Desktop Duplication) is currently in BETA.");
      ImGui::Spacing();
      ImGui::Text("If you experience flickering, black screens, or crashes, please switch to 'WGC Capture' or 'Monitor Capture' immediately.");
      ImGui::Spacing();
      ImGui::Text("Report all issues on the GitHub repository to help us improve.");

      ImGui::PopTextWrapPos();
      
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      // Right-aligned OK button
      // Use wrapWidth for alignment logic
      float avail = ImGui::GetContentRegionAvail().x;
      ImGui::SetCursorPosX(avail - 120);
      if (ImGui::Button("UNDERSTOOD", ImVec2(120, 35))) {
          m_showStartupAlert = false;
          ImGui::CloseCurrentPopup();
      }
      
      ImGui::EndPopup();
  }

  const auto& monitors = m_device.Monitors();
  if (!monitors.empty()) {
    if (m_selectedMonitor < 0) {
      m_selectedMonitor = 0;
    } else if (m_selectedMonitor >= static_cast<int>(monitors.size())) {
      m_selectedMonitor = static_cast<int>(monitors.size()) - 1;
    }
    std::string currentName = WideToUtf8(monitors[m_selectedMonitor].name);
    if (ImGui::BeginCombo("Monitor", currentName.c_str())) {
      for (int i = 0; i < static_cast<int>(monitors.size()); ++i) {
        bool selected = (i == m_selectedMonitor);
        std::string label = WideToUtf8(monitors[i].name);
        if (ImGui::Selectable(label.c_str(), selected)) {
          m_selectedMonitor = i;
          m_monitorDirty = true;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Select which monitor to display the output on.\nAlso used as capture target in Monitor mode.");
  }

  if (ImGui::Button("Refresh Windows")) {
    RefreshWindowList();
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rescan for available windows to capture.\\nUse after opening/closing applications.");

  const char* captureModes[] = {"Window", "Monitor", "Game (Hook)", "DXGI Crop"};
  ImGui::Combo("Capture Mode", &m_captureMode, captureModes, IM_ARRAYSIZE(captureModes));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Window: WGC capture (may be limited to 60fps by DWM)\\nMonitor: Capture entire monitor at full refresh rate\\nGame (Hook): DLL injection (blocked by anti-cheat)\\nDXGI Crop: Capture monitor at full refresh rate, crop to window (best for >60fps)");

  if (m_captureMode == 0 && !m_windows.empty()) {
    std::string windowPreview;
    const char* preview = "Select window";
    if (m_selectedWindow >= 0 && m_selectedWindow < static_cast<int>(m_windows.size())) {
      windowPreview = WideToUtf8(m_windows[m_selectedWindow].title);
      preview = windowPreview.c_str();
    }
    if (ImGui::BeginCombo("Capture Window", preview)) {
      for (int i = 0; i < static_cast<int>(m_windows.size()); ++i) {
        bool selected = (i == m_selectedWindow);
        std::string label = WideToUtf8(m_windows[i].title);
        if (ImGui::Selectable(label.c_str(), selected)) {
          m_selectedWindow = i;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    if (ImGui::Button("Start Capture") && m_selectedWindow >= 0 &&
        m_selectedWindow < static_cast<int>(m_windows.size())) {
      if (StartWindowCapture(m_windows[m_selectedWindow].hwnd)) {
        m_captureStatus = "Capture started";
        ResetCaptureState();
      } else {
        m_captureStatus = "Capture failed - try switching capture mode or check window is visible";
      }
    }
  }

  if (m_captureMode == 1) {
    if (ImGui::Button("Start Capture")) {
      HMONITOR monitor = m_device.MonitorHandle(m_selectedMonitor);
      if (monitor && StartMonitorCapture(monitor)) {
        m_captureStatus = "Capture started";
        ResetCaptureState();
      } else {
        m_captureStatus = "Capture failed - ensure Windows 10/11 with graphics drivers installed";
      }
    }
  }

  // Game capture mode (hook-based like OBS)
  if (m_captureMode == 2 && !m_windows.empty()) {
    std::string windowPreview;
    const char* preview = "Select window";
    if (m_selectedWindow >= 0 && m_selectedWindow < static_cast<int>(m_windows.size())) {
      windowPreview = WideToUtf8(m_windows[m_selectedWindow].title);
      preview = windowPreview.c_str();
    }
    if (ImGui::BeginCombo("Target Game", preview)) {
      for (int i = 0; i < static_cast<int>(m_windows.size()); ++i) {
        bool selected = (i == m_selectedWindow);
        std::string label = WideToUtf8(m_windows[i].title);
        if (ImGui::Selectable(label.c_str(), selected)) {
          m_selectedWindow = i;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    if (m_gameCapture.IsCapturing()) {
      ImGui::Text("Hook Status: Active (Frames: %llu)", m_gameCapture.GetFrameCount());
      if (ImGui::Button("Stop Game Capture")) {
        m_gameCapture.StopCapture();
        m_captureStatus = "Game capture stopped";
      }
    } else {
      if (ImGui::Button("Start Game Capture") && m_selectedWindow >= 0 &&
          m_selectedWindow < static_cast<int>(m_windows.size())) {
        if (m_gameCapture.StartCapture(m_windows[m_selectedWindow].hwnd)) {
          m_captureStatus = "Game capture started (hook injected)";
          m_captureWindow = m_windows[m_selectedWindow].hwnd;
          ResetCaptureState();
        } else {
          m_captureStatus = "Game capture failed: " + m_gameCapture.GetLastError();
        }
      }
    }
    ImGui::TextWrapped("Note: Hook-based capture requires the game to use DirectX 11. Run as Administrator if injection fails.");
  }

  // DXGI Crop mode - capture monitor, crop to window
  if (m_captureMode == 3 && !m_windows.empty()) {
    std::string windowPreview;
    const char* preview = "Select window";
    if (m_selectedWindow >= 0 && m_selectedWindow < static_cast<int>(m_windows.size())) {
      windowPreview = WideToUtf8(m_windows[m_selectedWindow].title);
      preview = windowPreview.c_str();
    }
    if (ImGui::BeginCombo("Target Window", preview)) {
      for (int i = 0; i < static_cast<int>(m_windows.size()); ++i) {
        bool selected = (i == m_selectedWindow);
        std::string label = WideToUtf8(m_windows[i].title);
        if (ImGui::Selectable(label.c_str(), selected)) {
          m_selectedWindow = i;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    if (ImGui::Button("Start DXGI Crop Capture") && m_selectedWindow >= 0 &&
        m_selectedWindow < static_cast<int>(m_windows.size())) {
      HWND hwnd = m_windows[m_selectedWindow].hwnd;
      HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
      
      std::ofstream log("dxgi_crop_debug.txt");
      log << "Attempting DXGI Crop Capture\n";
      log << "Target Window: " << (void*)hwnd << "\n";
      log << "Target Monitor: " << (void*)monitor << "\n";
      
      if (monitor) {
        m_captureWindow = hwnd;  // Store window for cropping
        m_captureWindowMonitor = monitor;
        
        // IMPORTANT: Move output window to DIFFERENT monitor to avoid feedback loop
        // DXGI captures the ENTIRE monitor, so if our overlay is visible, it will be captured too
        // This creates a feedback loop that freezes at ~60fps
        if (m_hwnd) {
          HMONITOR currentMonitor = MonitorFromWindow(m_hwnd, MONITOR_DEFAULTTONEAREST);
          if (currentMonitor == monitor) {
            // Output is on same monitor as capture target - need to move it
            log << "Output window on same monitor as capture target - moving to different monitor\n";
            
            // Store original position for later restoration
            WINDOWPLACEMENT wp = { sizeof(WINDOWPLACEMENT) };
            if (GetWindowPlacement(m_hwnd, &wp)) {
              m_dxgiCropOriginalRect = wp.rcNormalPosition;
              m_dxgiCropOriginalMonitor = currentMonitor;
            }
            
            // Find a different monitor
            HMONITOR otherMonitor = nullptr;
            EnumDisplayMonitors(nullptr, nullptr, [](HMONITOR hMonitor, HDC, LPRECT, LPARAM lParam) -> BOOL {
              HMONITOR* pOtherMonitor = reinterpret_cast<HMONITOR*>(lParam);
              if (*pOtherMonitor == nullptr) {
                *pOtherMonitor = hMonitor;
              }
              return *pOtherMonitor == nullptr; // Continue until we find a different one
            }, reinterpret_cast<LPARAM>(&otherMonitor));
            
            if (otherMonitor && otherMonitor != monitor) {
              // Move output window to other monitor
              MONITORINFO mi = { sizeof(mi) };
              if (GetMonitorInfo(otherMonitor, &mi)) {
                int monitorW = mi.rcWork.right - mi.rcWork.left;
                int monitorH = mi.rcWork.bottom - mi.rcWork.top;
                
                RECT winRect;
                GetWindowRect(m_hwnd, &winRect);
                int winW = winRect.right - winRect.left;
                int winH = winRect.bottom - winRect.top;
                
                int newX = mi.rcWork.left + (monitorW - winW) / 2;
                int newY = mi.rcWork.top + (monitorH - winH) / 2;
                
                SetWindowPos(m_hwnd, HWND_TOPMOST, newX, newY, winW, winH, SWP_SHOWWINDOW);
                log << "Moved output window to other monitor at (" << newX << "," << newY << ")\n";
                m_dxgiCropModeActive = true;
              }
            } else {
              // No other monitor available - try to hide the overlay temporarily
              log << "No other monitor available - overlay will cause feedback\n";
              // We'll still try, but it may not work well
            }
          } else {
            log << "Output window already on different monitor\n";
            m_dxgiCropModeActive = true;
          }
        }
        
        log << "Starting monitor capture...\n";
        if (StartMonitorCapture(monitor)) {
          m_captureStatus = "DXGI Crop capture started";
          ResetCaptureState();
          log << "StartMonitorCapture returned TRUE\n";
        } else {
          m_captureStatus = "DXGI capture failed";
          m_captureWindow = nullptr;
          m_dxgiCropModeActive = false;
          log << "StartMonitorCapture returned FALSE\n";
        }
      } else {
        log << "MonitorFromWindow failed\n";
        m_captureStatus = "Failed to get monitor for window";
      }
      log.close();
    }
    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Captures monitor at full refresh rate, crops to window");
    ImGui::TextWrapped("Best for >60fps capture when WGC is limited by DWM.");
  }

  if (m_captureMode == 0) {
    ImGui::Separator();
    
    ImGui::Checkbox("Overlay Mode (Play Through)", &m_overlayMode);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show interpolated frames as overlay on top of game.\nYour mouse and keyboard go DIRECTLY to the game!\nThe overlay shows smoother frames while you play normally.\n\nRECOMMENDED for gaming!");
    
    if (m_overlayMode) {
      ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Overlay active - interact with game normally!");
      ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Output window positioned over game, clicks pass through");
      
      ImGui::Separator();
      ImGui::Text("Game Resolution & Overlay Scaling");
      
      // Update values from current window if we haven't touched them
      if (m_captureWindow) {
         RECT rect;
         if (GetClientRect(m_captureWindow, &rect)) {
             int w = rect.right - rect.left;
             int h = rect.bottom - rect.top;
             ImGui::Text("Current Game Size: %dx%d", w, h);
             
             // If m_resizeWidth/Height are uninitialized or 0, set to current
             if (m_resizeWidth == 0 || m_resizeHeight == 0) {
                 m_resizeWidth = w;
                 m_resizeHeight = h;
             }
         }
      }

      ImGui::TextDisabled("Global Hotkeys");
      HotkeyEditor("Toggle Overlay", &m_hotkeyToggleOverlay);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hotkey to show/hide the main output overlay.\nWorks globally (even in-game).");
      HotkeyEditor("Toggle UI Window", &m_hotkeyToggleUi);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Hotkey to show/hide this settings window.\nWorks globally (even in-game).");

      bool sizeChanged = false;
      if (ImGui::InputInt("Width", &m_resizeWidth)) sizeChanged = true;
      if (ImGui::InputInt("Height", &m_resizeHeight)) sizeChanged = true;
      
      // Basic aspect ratio lock
      ImGui::Checkbox("Lock Aspect Ratio (16:9)", &m_resizeLockAspect);
      if (m_resizeLockAspect && sizeChanged) {
          // If width changed, update height
          // This is a simple approximation
          if (ImGui::IsItemActive()) { // Can't easily detect which specific input was edited in imgui immediate mode this way without more complex logic
              // Just enforce 16:9 for now on apply or if user modifies one
          }
      }

      // Show status if any
      if (!m_resizeStatus.empty()) {
          ImGui::TextWrapped("%s", m_resizeStatus.c_str());
      }

      if (ImGui::Button("Apply Resize to Game & Overlay")) {
          m_resizeStatus = "Applying...";
          if (m_captureWindow && IsWindow(m_captureWindow)) {
              if (m_resizeWidth > 0 && m_resizeHeight > 0) {
                  // If window is maximized, restore it first so it can be resized
                  if (IsZoomed(m_captureWindow)) {
                      ShowWindow(m_captureWindow, SW_RESTORE);
                  }

                  // Calculate the total window size required to achieve the desired CLIENT area size
                  RECT rect = { 0, 0, m_resizeWidth, m_resizeHeight };
                  DWORD style = GetWindowLong(m_captureWindow, GWL_STYLE);
                  DWORD exStyle = GetWindowLong(m_captureWindow, GWL_EXSTYLE);
                  bool hasMenu = GetMenu(m_captureWindow) != nullptr;
                  
                  // Adjust rect to include borders/title bar
                  AdjustWindowRectEx(&rect, style, hasMenu, exStyle);
                  int totalW = rect.right - rect.left;
                  int totalH = rect.bottom - rect.top;

                  // Use synchronous call to verify result
                  // Remove SWP_ASYNCWINDOWPOS to catch errors
                  BOOL result = SetWindowPos(m_captureWindow, nullptr, 0, 0, totalW, totalH, 
                              SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOCOPYBITS);
                  
                  if (result) {
                      m_resizeStatus = "Success! Resized window to " + std::to_string(totalW) + "x" + std::to_string(totalH);
                      // Force a UI update immediately so user sees the change
                      UpdateOutputOverlayWindow();
                  } else {
                      DWORD error = GetLastError();
                      m_resizeStatus = "Failed to resize. Error code: " + std::to_string(error);
                      if (error == 5) {
                          m_resizeStatus += " (Access Denied). Try running FrameGen as Admin.";
                      }
                  }
              }
          } else {
              m_resizeStatus = "Error: No capture window selected or window is invalid.";
          }
      }

      ImGui::Separator();
    }
    
    if (!m_overlayMode) {
      if (ImGui::Checkbox("Fullscreen Output (Window)", &m_fullscreenWindowOutput)) {
        if (m_fullscreenWindowOutput) {
          m_monitorDirty = true;
        }
      }
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Display interpolated output in a fullscreen window.\nBest used on a SECOND MONITOR while gaming on primary.\nOr for viewing recordings/videos.");
      
      ImGui::Checkbox("Hide Source Window", &m_hideCaptureWindow);
      if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimize the original window.\nUseful when output is on second monitor.");
    }
    
    ImGui::Checkbox("Prefer WGC for Window", &m_windowCapturePreferWgc);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Use Windows Graphics Capture instead of Desktop Duplication.\nWGC works better with hidden/fullscreen windows.");
    
    ImGui::Checkbox("Force DXGI Capture (Desktop Duplication)", &m_forceDxgiCapture);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("EXPERIMENTAL: Force Desktop Duplication even in Overlay mode.\nUse this if WGC is capped at 60 FPS.\nWARNING: May cause infinite loop feedback if overlay is visible!");
    
    // Alert for DXGI
    ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "ALERT: DXGI is currently unstable.");
    ImGui::SameLine();
    ImGui::TextDisabled("If broken, report on GitHub.");

    if (m_forceDxgiCapture) {
         ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Warning: DXGI captures visible screen content (occlusion prone).");
         ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "It may recursively capture this overlay.");
    }

    ImGui::Checkbox("Force WGC Capture (No Fallback)", &m_forceWgcCapture);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Never fallback to Desktop Duplication if WGC fails.\nUse when Desktop Duplication causes issues.");
    
    ImGui::Checkbox("Unlock App FPS (High CPU)", &m_unlockAppFps);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Removes frame pacing limits.\nAllows the capture loop to run as fast as possible (1000+ FPS).\nEssential for capturing >60FPS on some systems.\nWARNING: Increases CPU usage.");
     
    ImGui::Checkbox("Force Interpolation", &m_forceInterpolation);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bypass framerate safety checks.\nForces interpolation even if Capture FPS matches Monitor FPS (which normally disables it).\nEnable this if game looks like 'base fps'.");

    ImGui::Checkbox("Capture Low Latency Mode", &m_wgcLowLatencyMode);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("WGC: Use smaller frame pool (2 frames).\nDXGI: Use aggressive cleanup.\nDisable if you experience frame drops with high-FPS content.");
     
    ImGui::Checkbox("Capture Prefer Newest Frame", &m_wgcPreferNewest);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Always use the most recent frame, dropping older buffered frames.\nReduces latency but may increase dropped frame count.\nRecommended for gaming.");
    
    if (m_forceWgcCapture) {
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "WARNING: Force WGC will not fallback to Desktop Duplication");
      ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "If WGC fails, check: AMD MPO disabled, VRR off, Performance mode");
    }
    if (m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled) {
      ImGui::Text("Fullscreen/hidden output forces WGC for window capture.");
      ImGui::Text("Keep the source window unminimized.");
    }
  }

  ImGui::Separator();
  const char* outputModes[] = {"Multiplier", "Monitor Sync"};
  ImGui::Combo("Output Mode", &m_outputMode, outputModes, IM_ARRAYSIZE(outputModes));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Multiplier: Output at capture FPS x multiplier (e.g., 60->120)\nMonitor Sync: Match monitor refresh rate for smoothest output");
  
  const char* outputDisplays[] = {"Output Window", "Preview (UI)", "Overlay (click-through)"};
  bool forceOutputWindow = (m_captureMode == 0 &&
                            (m_fullscreenWindowOutput || m_hideCaptureWindow || m_outputInputEnabled));
  if (forceOutputWindow) {
    m_outputDisplayMode = 0;
    ImGui::Text("Output Display: Output Window");
  } else {
    ImGui::Combo("Output Display", &m_outputDisplayMode, outputDisplays, IM_ARRAYSIZE(outputDisplays));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Output Window: Separate fullscreen window\nPreview (UI): Small preview in this settings window\nOverlay: Click-through transparent overlay");
  }
  if (forceOutputWindow) {
    ImGui::Text("Output display forced to Output Window for window capture.");
  }
  if (m_outputDisplayMode == 1) {
    ImGui::Text("Output window hidden to avoid occluding capture.");
  } else if (m_outputDisplayMode == 2) {
    ImGui::Text("Overlay can be captured by duplication; preview is safer.");
    ImGui::Text("If overlay is black, switch to Preview (UI).");
  } else if (m_captureMode == 0 && !m_windowCaptureUsingWgc) {
    ImGui::Text("Tip: Output window can occlude window capture.");
  }
  ImGui::Checkbox("Interpolation", &m_interpolationEnabled);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable motion-compensated frame interpolation.\nGenerates new frames between captured frames for smoother output.\nDisable for passthrough (no frame generation).");
  
  ImGui::Checkbox("Low Latency", &m_lowLatencyMode);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimize input-to-display latency.\nUses smaller frame buffer and faster timing.\nMay cause occasional stutter if capture is inconsistent.");
  
  /* Never Drop Frames removed
  if (m_neverDropFrames) {
    ImGui::SliderInt("Max Queue Size", &m_maxQueueSize, 4, kFrameQueueSize);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum frames to buffer before forcing output.\nHigher = smoother but more latency.\nLower = less latency but may stutter.");
  }
  */
  ImGui::Checkbox("Temporal Stabilization", &m_temporalStabilization);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Blend motion vectors across frames for stability.\nReduces flickering in motion estimation.\nSlightly increases smoothness at cost of responsiveness.");
  
  if (m_temporalStabilization) {
    ImGui::SliderFloat("Temporal History", &m_temporalHistoryWeight, 0.0f, 0.99f, "%.2f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much to blend with previous frame's motion.\n0 = No history (current only)\nHigher = More stable but less responsive to changes.");
    
    ImGui::SliderFloat("Temporal Conf Influence", &m_temporalConfInfluence, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("How much confidence affects temporal blending.\nHigher = Only trust history when confident.\nLower = Always blend with history.");

    ImGui::SliderInt("Neighbor Size", &m_temporalNeighborhoodSize, 1, 5);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Size of the clamping box (Kernel Radius).\n1 = 3x3 (Sharpest)\n2 = 5x5 (Balanced)\n3 = 7x7 (Smoother)\n4 = 9x9 (Very Stable)\n5 = 11x11 (Maximum Stability)");
  }
  
  ImGui::Checkbox("Motion Prediction (Multi-Frame)", &m_useMotionPrediction);
  const char* predictionHelp = "Uses previous frame's motion to help find the next motion.\nImproves quality for fast moving objects.";
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", predictionHelp);

  ImGui::Checkbox("Text Preservation Mode", &m_textPreservationMode);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Optimize settings for text rendering.\nReduces flickering on text by adjusting temporal stabilization and motion smoothing.\nMay reduce smoothness for fast-moving content.");
  ImGui::BeginDisabled(!m_textPreservationMode);
  ImGui::SliderFloat("Text Preserve Strength", &m_textPreservationStrength, 0.0f, 1.0f, "%.2f");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Bias interpolation toward unwarped frames on sharp edges (text/hud).\nHigher reduces shimmer but can look less smooth.");
  ImGui::SliderFloat("Text Edge Threshold", &m_textPreservationEdgeThreshold, 0.0f, 0.2f, "%.3f");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edge sensitivity for text protection.\nLower protects more edges; higher limits protection to only the sharpest text.");
  ImGui::EndDisabled();

  // Smooth Blend removed

  ImGui::Checkbox("Limit Output FPS", &m_limitOutputFps);
  const char* limitFpsHelp = "Limit frame rate using a high-resolution waitable timer.\nDisable Monitor Sync to use this pacing.";
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", limitFpsHelp);
  
  ImGui::Checkbox("VSync (Monitor Sync)", &m_useVsync);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Synchronize output with monitor refresh rate.\nPrevents tearing but may add latency.\nDisable for lowest latency with possible tearing.");
  const char* motionModelLabels[] = {"Adaptive (Recommended)", "Stable (No Flicker)", "Balanced", "Coverage (Fast Motion)"};
  ImGui::Combo("Motion Model", &m_motionModel, motionModelLabels, IM_ARRAYSIZE(motionModelLabels));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Replaces manual Search/Refine Radius.\nAdaptive: scene-aware model for mixed content.\nStable: strongest anti-flicker and deterministic vector selection.\nBalanced: default quality/speed.\nCoverage: wider motion capture for fast movement.");
  
  ImGui::SliderInt("Output Multiplier", &m_outputMultiplier, 1, 20);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Frame rate multiplier.\n1x = No interpolation (passthrough)\n2x = Double frame rate (60->120)\n3x = Triple (60->180)\n4x = Quadruple (60->240)\n5-20x = Extreme multipliers (quality/latency may degrade)");
  
  ImGui::SliderFloat("Delay Scale", &m_delayScale, 0.5f, 1.5f, "%.2f");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scale the timing delay for frame presentation.\n<1.0 = Show frames earlier (lower latency, may stutter)\n>1.0 = Show frames later (smoother, more latency)\n1.0 = Default timing.");
  
  ImGui::SliderFloat("Jitter Suppression", &m_jitterSuppression, 0.0f, 1.0f, "%.2f");
  const char* jitterHelp = "Tolerance for using average frame interval (0.0 = Off, 0.2 = 20%).\nIncreases smoothness but might drift if source fps fluctuates.";
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", jitterHelp);

  ImGui::Checkbox("Adaptive Delay", &m_adaptiveDelay);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Automatically adjust timing based on frame queue depth.\nHelps maintain smooth output when capture rate varies.\nRecommended for variable frame rate sources.");
  
  if (m_adaptiveDelay) {
    ImGui::SliderInt("Target Queue Depth", &m_targetQueueDepth, 2, kFrameQueueSize);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Target number of frames to keep buffered.\nHigher = Smoother but more latency.\nLower = Less latency but may stutter.");
  }
  
  ImGui::SliderFloat("Conf Power", &m_confidencePower, 0.5f, 3.0f, "%.2f");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Confidence curve power for motion reliability.\nHigher = Only trust very confident motion estimates\nLower = Trust motion more liberally\nRecommended: 1.0-1.5");
  
  ImGui::SliderFloat("Motion Edge", &m_motionEdgeScale, 1.0f, 12.0f, "%.2f");
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edge detection sensitivity for motion smoothing.\nHigher = Preserve sharp edges better (may be noisy)\nLower = Smoother motion field (may blur edges)\nRecommended: 6.0");
  
  const char* qualityLabels[] = {"Standard (Fast)", "High (Sharp)"};
  ImGui::Combo("Quality Mode", &m_interpolationQuality, qualityLabels, IM_ARRAYSIZE(qualityLabels));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Standard: Bilinear sampling (Blurrier, less GPU usage)\nHigh: Bicubic + Linear Light (Sharper, brightness correct, more GPU usage)");

  ImGui::Text("Delay: %.2f ms", m_outputDelayMs);
  const char* debugLabels[] = {"None", "Motion Flow", "Confidence Heatmap", "Motion Needles", "Residual Error", "Split Screen", "Occlusion", "AI Ghost Mask", "Structure Gradient"};
  ImGui::Combo("Debug View", &m_debugView, debugLabels, IM_ARRAYSIZE(debugLabels));
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Debug visualization modes:\nNone: Final interpolated result\nFlow: Color-coded motion\nHeatmap: Confidence (Green=Good, Red=Bad)\nNeedles: Motion vectors\nResidual: Warping error check\nSplit: Compare Source vs Warped\nOcclusion: Disoccluded areas\nAI Ghost Mask: Visualization of Disocclusion Logic\nStructure Gradient: Edges used for Motion Search");
  if (m_debugView == static_cast<int>(Interpolator::DebugViewMode::MotionFlow) || 
      m_debugView == static_cast<int>(Interpolator::DebugViewMode::MotionNeedles)) {
    ImGui::SliderFloat("Motion Scale", &m_debugMotionScale, 0.005f, 0.2f, "%.3f");
  }
  if (m_debugView == static_cast<int>(Interpolator::DebugViewMode::ResidualError)) {
    ImGui::SliderFloat("Diff Scale", &m_debugDiffScale, 0.5f, 8.0f, "%.2f");
  }

  float captureFps = (m_avgFrameInterval > 0.0) ? static_cast<float>(1.0 / m_avgFrameInterval) : 0.0f;
  float monitorHz = m_device.RefreshHz(m_selectedMonitor);
  float maxHz = m_device.MaxRefreshHz(m_selectedMonitor);
  float targetFps = m_targetFps;
  ImGui::Text("Capture FPS: %.1f", captureFps);
  ImGui::Text("Actual Capture: %.1f", m_captureFps);
  ImGui::Text("Target FPS: %.1f", targetFps);
  ImGui::Text("Output FPS: %.1f", m_presentFps);
  ImGui::Text("Monitor Hz: %.1f", monitorHz);
  ImGui::Text("Monitor Max Hz: %.1f", maxHz);
  if (m_captureMode == 0) {
    float captureMonitorHz = 0.0f;
    HMONITOR captureMonitor = nullptr;
    if (m_captureWindow) {
      captureMonitor = MonitorFromWindow(m_captureWindow, MONITOR_DEFAULTTONEAREST);
    }
    if (captureMonitor) {
      const auto& monitors = m_device.Monitors();
      for (int i = 0; i < static_cast<int>(monitors.size()); ++i) {
        if (monitors[i].monitor == captureMonitor) {
          captureMonitorHz = m_device.RefreshHz(i);
          break;
        }
      }
    }
    if (captureMonitorHz > 0.0f) {
      ImGui::Text("Capture Monitor Hz: %.1f", captureMonitorHz);
    }
  }
  const char* backendLabel = "WGC";
  if (m_captureMode == 0) {
    backendLabel = m_windowCaptureUsingWgc ? "WGC (Window)" : "Desktop Duplication (Window)";
  }
  ImGui::Text("Capture Backend: %s", backendLabel);
  if (m_captureMode == 0 && !m_windowCaptureUsingWgc) {
    ImGui::Text("Window capture is visible-only (occlusion/minimize affects frames).");
  }
  ImGui::Text("Allow Tearing: %s", m_device.AllowTearing() ? "yes" : "no");
  ImGui::Text("Interp alpha: %.2f", m_lastAlpha);
  ImGui::Text("Interpolated: %s", m_lastInterpolated ? "yes" : "no");
  ImGui::Text("Interval: %.2f ms", m_lastIntervalMs);
  ImGui::Text("Avg Interval: %.2f ms", m_lastAvgIntervalMs);
  ImGui::Text("Unstable: %s", m_lastUnstable ? "yes" : "no");
  ImGui::Text("Frame: %dx%d", m_frameWidth, m_frameHeight);
  ImGui::Text("Output: %dx%d", m_outputWidth, m_outputHeight);
  bool isCapturing = false;
  if (m_captureMode == 0) {
    isCapturing = m_windowCaptureUsingWgc ? m_capture.IsCapturing() : m_dupCapture.IsCapturing();
  } else {
    isCapturing = m_capture.IsCapturing();
  }
  ImGui::Text("Capturing: %s", isCapturing ? "yes" : "no");
  if (!m_captureStatus.empty()) {
    ImGui::Text("Status: %s", m_captureStatus.c_str());
  }

  // Frame Timing Diagnostics removed

  // WGC capture statistics
  if (m_windowCaptureUsingWgc || m_captureMode == 1) {
    auto stats = m_capture.GetStatistics();
    if (stats.capturedFrames > 0) {
      ImGui::Text("WGC Captured: %u frames", stats.capturedFrames);
      if (stats.avgFrameIntervalMs > 0.0) {
        ImGui::Text("WGC Avg Interval: %.2f ms (%.1f FPS)", stats.avgFrameIntervalMs, 1000.0 / stats.avgFrameIntervalMs);
      }
      if (stats.lastFrameAgeMs > 0.0) {
        ImVec4 ageColor = (stats.lastFrameAgeMs < 5.0) ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f) :
                          (stats.lastFrameAgeMs < 16.0) ? ImVec4(1.0f, 1.0f, 0.3f, 1.0f) :
                                                          ImVec4(1.0f, 0.5f, 0.0f, 1.0f);
        ImGui::TextColored(ageColor, "Frame Age: %.1f ms", stats.lastFrameAgeMs);
      }
      if (stats.droppedFrames > 0) {
        float dropRate = 100.0f * stats.droppedFrames / (stats.capturedFrames + stats.droppedFrames);
        ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.0f, 1.0f), "WGC Dropped: %u (%.1f%%)", stats.droppedFrames, dropRate);
      }
    }
  }

  // FPS Cap Detect removed

  ImGui::Separator();
  if (ImGui::Button("Export Diagnostics to File")) {
    ExportDiagnostics();
  }
  ImGui::SameLine();
  ImGui::Text("(Creates FrameGen_Diagnostics_TIMESTAMP.txt in exe dir)");
  if (!m_captureStatus.empty()) {
    ImGui::TextColored(ImVec4(0.3f, 0.8f, 0.3f, 1.0f), "%s", m_captureStatus.c_str());
  }



  ImGui::End();

  if (m_outputDisplayMode == 1) {
    ImGui::SetNextWindowBgAlpha(0.9f);
    ImGui::Begin("Output Preview", nullptr,
                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings);
    if (m_lastOutputSrv && m_lastOutputWidth > 0 && m_lastOutputHeight > 0) {
      ImVec2 display = ImGui::GetIO().DisplaySize;
      float maxWidth = display.x * 0.6f;
      float maxHeight = display.y * 0.6f;
      if (maxWidth < 320.0f) {
        maxWidth = 320.0f;
      }
      if (maxHeight < 180.0f) {
        maxHeight = 180.0f;
      }
      if (maxWidth > 960.0f) {
        maxWidth = 960.0f;
      }
      if (maxHeight > 540.0f) {
        maxHeight = 540.0f;
      }

      float aspect = static_cast<float>(m_lastOutputWidth) /
                     static_cast<float>(m_lastOutputHeight);
      float width = maxWidth;
      float height = width / aspect;
      if (height > maxHeight) {
        height = maxHeight;
        width = height * aspect;
      }

      ImVec2 imageMin = ImGui::GetCursorScreenPos();
      ImGui::Image(reinterpret_cast<ImTextureID>(m_lastOutputSrv.Get()), ImVec2(width, height));
      ImVec2 imageMax = ImVec2(imageMin.x + width, imageMin.y + height);
      POINT clientMin{static_cast<LONG>(imageMin.x), static_cast<LONG>(imageMin.y)};
      POINT clientMax{static_cast<LONG>(imageMax.x), static_cast<LONG>(imageMax.y)};
      ScreenToClient(m_uiHwnd, &clientMin);
      ScreenToClient(m_uiHwnd, &clientMax);
      m_previewRect = {clientMin.x, clientMin.y, clientMax.x, clientMax.y};
      m_previewHasImage = true;
      ImGui::Text("Click preview to control; Esc to release.");
      ImGui::Text("Input: %s", m_previewInputActive ? "active" : "inactive");
    } else {
      ImGui::Text("Waiting for frames...");
      m_previewHasImage = false;
    }
    ImGui::End();
  } else {
    m_previewHasImage = false;
    m_previewInputActive = false;
  }
}

void App::RefreshWindowList() {
  m_windows = EnumerateTopLevelWindows(m_hwnd);
  if (m_selectedWindow >= static_cast<int>(m_windows.size())) {
    m_selectedWindow = -1;
  }
}

void App::SelectMonitor(int index) {
  if (index < 0 || index >= m_device.MonitorCount()) {
    return;
  }

  m_device.SetMonitor(index, m_hwnd);
  RECT rect = m_device.MonitorRect(index);
  m_outputWidth = rect.right - rect.left;
  m_outputHeight = rect.bottom - rect.top;

  if (m_frameWidth > 0 && m_frameHeight > 0) {
    m_interpolator.Resize(m_frameWidth, m_frameHeight, m_outputWidth, m_outputHeight);
  }
}

void App::ResizeForCapture(int width, int height) {
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384) {
    return;
  }
  m_frameWidth = width;
  m_frameHeight = height;

  m_frameQueue.clear();
  m_queueWrite = 0;
  m_outputStepIndex = 0;
  m_holdEndFrame = false;
  m_pairPrevSlot = -1;
  m_pairCurrSlot = -1;
  m_pairPrevTime100ns = 0;
  m_pairCurrTime100ns = 0;
  m_frameTime100ns.fill(0);
  m_prevFrameTime100ns = 0;
  m_currFrameTime100ns = 0;
  m_timeOffsetValid = false;
  m_timeOffset100ns = 0.0;
  m_avgFrameInterval = 0.0;

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(width);
  desc.Height = static_cast<UINT>(height);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

  for (int i = 0; i < kFrameQueueSize; ++i) {
    m_frameTextures[i].Reset();
    m_frameSrvs[i].Reset();
    m_device.Device()->CreateTexture2D(&desc, nullptr, &m_frameTextures[i]);
    if (m_frameTextures[i]) {
      m_device.Device()->CreateShaderResourceView(m_frameTextures[i].Get(), nullptr, &m_frameSrvs[i]);
    }
  }

  if (m_outputWidth > 0 && m_outputHeight > 0) {
    m_interpolator.Resize(m_frameWidth, m_frameHeight, m_outputWidth, m_outputHeight);
  }
  m_interpolator.ResetTemporal();
}

LRESULT CALLBACK App::WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
  App* app = nullptr;
  if (message == WM_NCCREATE) {
    auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
    app = static_cast<App*>(cs->lpCreateParams);
    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
  } else {
    app = reinterpret_cast<App*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
  }

  if (app) {
    return app->HandleMessage(hwnd, message, wParam, lParam);
  }

  return DefWindowProc(hwnd, message, wParam, lParam);
}

LRESULT CALLBACK App::UiWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
  App* app = nullptr;
  if (message == WM_NCCREATE) {
    auto* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
    app = static_cast<App*>(cs->lpCreateParams);
    SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
  } else {
    app = reinterpret_cast<App*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
  }

  if (app) {
    return app->HandleUiMessage(hwnd, message, wParam, lParam);
  }

  return DefWindowProc(hwnd, message, wParam, lParam);
}

LRESULT App::HandleMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
  // OVERLAY MODE: Pass ALL input through to game window
  // This MUST be first, before any other message handling
  if (m_overlayMode && m_captureWindow && m_outputWindowVisible) {
    if (message == WM_NCHITTEST) {
      return HTTRANSPARENT;  // All mouse input passes through
    }
    if (message == WM_MOUSEACTIVATE) {
      return MA_NOACTIVATEANDEAT;  // Don't activate AND eat the message
    }
    if (message == WM_SETCURSOR) {
      return TRUE;  // Don't change cursor
    }
    if (message == WM_ACTIVATE) {
      if (LOWORD(wParam) != WA_INACTIVE) {
        SetForegroundWindow(m_captureWindow);
        return 0;
      }
    }
  }

  switch (message) {
    case WM_NCHITTEST:
      // Also handle overlay display mode
      if (m_outputDisplayMode == 2 && m_captureMode == 0 &&
          !m_outputInputEnabled && !m_fullscreenWindowOutput) {
        return HTTRANSPARENT;
      }
      break;
    case WM_SIZE: {
      if (wParam == SIZE_MINIMIZED) {
        return 0;
      }
      UINT width = LOWORD(lParam);
      UINT height = HIWORD(lParam);
      if (width > 0 && height > 0) {
        m_outputWidth = static_cast<int>(width);
        m_outputHeight = static_cast<int>(height);
        m_device.ResizeSwapChain(width, height);
        if (m_frameWidth > 0 && m_frameHeight > 0) {
          m_interpolator.Resize(m_frameWidth, m_frameHeight, m_outputWidth, m_outputHeight);
        }
      }
      return 0;
    }
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      break;
  }

  return DefWindowProc(hwnd, message, wParam, lParam);
}

void App::ExportDiagnostics() {
  std::ostringstream ss;
  ss << "=== True Motion Fidelity Engine Diagnostics ===" << std::endl;
  ss << "Capture Backend: " << (m_windowCaptureUsingWgc ? "WGC" : "Desktop Duplication") << std::endl;
  ss << "WGC Low Latency Mode: " << (m_wgcLowLatencyMode ? "Enabled" : "Disabled") << std::endl;
  if (m_windowCaptureUsingWgc) {
    auto stats = m_capture.GetStatistics();
    ss << "WGC Captured Frames: " << stats.capturedFrames << std::endl;
    ss << "WGC Dropped Frames: " << stats.droppedFrames << std::endl;
  }
  ss << "Force WGC: " << (m_forceWgcCapture ? "Yes" : "No") << std::endl;
  ss << "Monitor Refresh Rate: " << m_device.RefreshHz(m_selectedMonitor) << " Hz" << std::endl;
  ss << "Monitor Max Hz: " << m_device.MaxRefreshHz(m_selectedMonitor) << " Hz" << std::endl;
  ss << "Capture FPS: " << ((m_avgFrameInterval > 0.0) ? (1.0 / m_avgFrameInterval) : 0.0) << std::endl;
  ss << "Actual Capture Rate: " << m_captureFps << " FPS" << std::endl;
  ss << "Output FPS: " << m_presentFps << " FPS" << std::endl;
  ss << "Frame Interval Avg: " << ((m_frameIntervalCount > 0) ? (m_frameIntervalSum / m_frameIntervalCount) : 0.0) << " ms" << std::endl;
  ss << "Frame Interval Min: " << m_minFrameInterval << " ms" << std::endl;
  ss << "Frame Interval Max: " << m_maxFrameInterval << " ms" << std::endl;
  ss << "Frame Jitter: " << (m_maxFrameInterval - m_minFrameInterval) << " ms" << std::endl;
  ss << "Frame Count: " << m_frameTimestamps.size() << std::endl;

  if (!m_frameTimestamps.empty()) {
    ss << std::endl << "=== Last 60 Frame Intervals (ms) ===" << std::endl;
    size_t start = (m_frameTimestamps.size() > 60) ? (m_frameTimestamps.size() - 60) : 0;
    for (size_t i = start + 1; i < m_frameTimestamps.size(); ++i) {
      double interval = (m_frameTimestamps[i] - m_frameTimestamps[i-1]) * 1e-7;
      ss << interval << std::endl;
    }
  }

  OSVERSIONINFOEXW osvi = {};
  osvi.dwOSVersionInfoSize = sizeof(osvi);
#pragma warning(push)
#pragma warning(disable : 4996)
  GetVersionExW(reinterpret_cast<OSVERSIONINFOW*>(&osvi));
#pragma warning(pop)
  ss << std::endl << "=== System Information ===" << std::endl;
  ss << "Windows Version: " << osvi.dwMajorVersion << "." << osvi.dwMinorVersion << std::endl;
  ss << "Build Number: " << osvi.dwBuildNumber << std::endl;
  ss << "Admin Rights: " << (IsUserAnAdmin() ? "Yes" : "No") << std::endl;

  ss << std::endl << "=== Capture Configuration ===" << std::endl;
  ss << "Interpolation: " << (m_interpolationEnabled ? "Enabled" : "Disabled") << std::endl;
  ss << "Output Multiplier: " << m_outputMultiplier << "x" << std::endl;
  static const char* kMotionModelNames[] = {"Adaptive", "Stable", "Balanced", "Coverage"};
  int motionModel = m_motionModel;
  if (motionModel < 0) motionModel = 0;
  if (motionModel > 3) motionModel = 3;
  ss << "Motion Model: " << kMotionModelNames[motionModel] << std::endl;
  ss << "Never Drop Frames: " << (m_neverDropFrames ? "Yes" : "No") << std::endl;
  ss << "Max Queue Size: " << m_maxQueueSize << std::endl;
  ss << "Temporal Stabilization: " << (m_temporalStabilization ? "Enabled" : "Disabled") << std::endl;

  std::string filename = "TrueMotion_Diagnostics_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".txt";
  std::ofstream file(filename);
  if (file.is_open()) {
    file << ss.str();
    file.close();
    m_captureStatus = "Diagnostics exported to: " + filename;
  }
}

LRESULT App::HandleUiMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
  if (HandlePreviewMouse(message, wParam, lParam)) {
    return 0;
  }
  if (HandlePreviewKey(message, wParam, lParam)) {
    return 0;
  }

  bool imguiHandled = ImGui_ImplWin32_WndProcHandler(hwnd, message, wParam, lParam);

  switch (message) {
    case WM_SIZE: {
      if (wParam == SIZE_MINIMIZED) {
        return 0;
      }
      UINT width = LOWORD(lParam);
      UINT height = HIWORD(lParam);
      if (width > 0 && height > 0) {
        UpdateUiSwapChain(width, height);
      }
      return 0;
    }
    case WM_CLOSE:
      m_showUi = false;
      m_uiVisible = false;
      ShowWindow(hwnd, SW_HIDE);
      return 0;
    case WM_KILLFOCUS:
      m_previewInputActive = false;
      break;
    default:
      break;
  }

  if (imguiHandled) {
    return true;
  }

  return DefWindowProc(hwnd, message, wParam, lParam);
}
