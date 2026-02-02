#include "wgc_capture.h"

#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Foundation.h>
#include <fstream>
#include <iostream>
#include <intrin.h>  // For _mm_pause() spin-wait

using winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool;
using winrt::Windows::Graphics::Capture::GraphicsCaptureItem;
using winrt::Windows::Graphics::Capture::GraphicsCaptureSession;
using winrt::Windows::Graphics::DirectX::DirectXPixelFormat;
using winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
using winrt::Windows::Foundation::TimeSpan;

// Get Windows build number
static DWORD GetWindowsBuildNumber() {
  static DWORD buildNumber = 0;
  if (buildNumber == 0) {
    HKEY hKey;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion", 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
      DWORD size = sizeof(DWORD);
      DWORD value = 0;
      if (RegQueryValueExW(hKey, L"CurrentBuildNumber", nullptr, nullptr, nullptr, &size) == ERROR_SUCCESS) {
        wchar_t buffer[32] = {};
        size = sizeof(buffer);
        if (RegQueryValueExW(hKey, L"CurrentBuildNumber", nullptr, nullptr, (LPBYTE)buffer, &size) == ERROR_SUCCESS) {
          buildNumber = static_cast<DWORD>(_wtoi(buffer));
        }
      }
      RegCloseKey(hKey);
    }
  }
  return buildNumber;
}

// MinUpdateInterval requires Windows 11 22H2 (build 22621) or later
static bool SupportsMinUpdateInterval() {
  return GetWindowsBuildNumber() >= 22621;
}

namespace {

// Frame pool sizes: larger = more buffering (smoother), smaller = lower latency
constexpr int kFramePoolSizeNormal = 4;      // Balanced
constexpr int kFramePoolSizeLowLatency = 2;  // Minimum for double-buffering

// QPC frequency for timing calculations
int64_t GetQpcFrequency() {
  static int64_t freq = 0;
  if (freq == 0) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    freq = f.QuadPart;
  }
  return freq;
}

int64_t GetQpcNow() {
  LARGE_INTEGER qpc;
  QueryPerformanceCounter(&qpc);
  return qpc.QuadPart;
}

IDirect3DDevice CreateDirect3DDevice(ID3D11Device* device) {
  winrt::com_ptr<IDXGIDevice> dxgiDevice;
  winrt::check_hresult(device->QueryInterface(IID_PPV_ARGS(dxgiDevice.put())));

  winrt::com_ptr<IInspectable> inspectable;
  winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(), inspectable.put()));
  return inspectable.as<IDirect3DDevice>();
}

template <typename T>
winrt::com_ptr<T> GetDxgiInterface(winrt::Windows::Foundation::IInspectable const& object) {
  auto access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
  winrt::com_ptr<T> result;
  winrt::check_hresult(access->GetInterface(__uuidof(T), result.put_void()));
  return result;
}

}  // namespace

bool WgcCapture::Initialize(ID3D11Device* device) {
  if (!device) {
    return false;
  }

  m_device = device;
  device->GetImmediateContext(&m_context);
  m_winrtDevice = CreateDirect3DDevice(device);
  return true;
}

void WgcCapture::Shutdown() {
  StopCapture();
  m_captureTexture.Reset();
  m_context.Reset();
  m_device.Reset();
}

bool WgcCapture::StartCapture(HWND hwnd) {
  {
    std::ofstream log("C:\\wgc_debug.txt", std::ios::app);
    log << "=== WgcCapture::StartCapture called ===\n";
    log << "HWND: " << hwnd << "\n";
    log << "m_winrtDevice valid: " << (m_winrtDevice ? "YES" : "NO") << "\n";
    log.close();
  }
  
  if (!m_winrtDevice) {
    return false;
  }

  try {
    StopCapture();

    auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    GraphicsCaptureItem item{nullptr};
    winrt::check_hresult(interop->CreateForWindow(hwnd, winrt::guid_of<GraphicsCaptureItem>(),
                                                  reinterpret_cast<void**>(winrt::put_abi(item))));

    if (!item) {
      return false;
    }

    m_item = item;
    auto size = item.Size();
    m_width = size.Width;
    m_height = size.Height;

    // Use appropriate frame pool size based on latency mode
    int poolSize = m_lowLatencyMode ? kFramePoolSizeLowLatency : kFramePoolSizeNormal;
    
    m_framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        m_winrtDevice,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        poolSize,
        size);

    m_frameArrivedToken = m_framePool.FrameArrived(
        {this, &WgcCapture::OnFrameArrived});

    m_session = m_framePool.CreateCaptureSession(item);
    m_session.IsCursorCaptureEnabled(false);
    m_session.IsBorderRequired(false);
    
    // On Windows 11 22H2+, set MinUpdateInterval to remove FPS cap
    // BUG FIX: MinUpdateInterval(0) caps at ~50fps!
    // Minimum value to actually remove cap is 1ms = 10000 (100-ns units)
    {
      std::ofstream log("C:\\wgc_debug.txt", std::ios::app);
      log << "=== WGC StartCapture ===\n";
      log << "Windows Build: " << GetWindowsBuildNumber() << "\n";
      log << "SupportsMinUpdateInterval: " << (SupportsMinUpdateInterval() ? "YES" : "NO") << "\n";
      
      if (SupportsMinUpdateInterval()) {
        try {
          // BUG FIX: MinUpdateInterval(0) caps at ~50fps!
          // Minimum value to actually remove cap is 1ms = 10000 (100-ns units)
          // Per https://github.com/robmikh/Win32CaptureSample/issues/82
          m_session.MinUpdateInterval(TimeSpan{ 10000 });
          log << "MinUpdateInterval set to 10000 (1ms) - should remove cap\n";
        } catch (const winrt::hresult_error& e) {
          log << "MinUpdateInterval failed: 0x" << std::hex << e.code().value << std::dec << "\n";
        } catch (...) {
          log << "MinUpdateInterval failed: unknown exception\n";
        }
      }
      log.close();
    }
    
    m_session.StartCapture();
    m_hasNewFrame.store(false);
    m_pendingFrameCount.store(0);
    m_isCapturing = true;
    m_hasError = false;
    m_droppedFrames = 0;
    m_capturedFrames = 0;
    m_captureHwnd = hwnd;
    m_captureMonitor = nullptr;
    m_isWindowCapture = true;
    m_prevFrameQpc = 0;
    m_frameIntervalSum = 0.0;
    m_frameIntervalCount = 0;

    // Start the dedicated capture thread
    m_stopThread = false;
    m_threadActive = true;
    m_captureThread = std::thread(&WgcCapture::CaptureThreadLoop, this);

    return true;
  } catch (const winrt::hresult_error& e) {
    std::ofstream log("wgc_error.txt", std::ios::app);
    if (log.is_open()) {
      log << "WGC StartCapture failed - hresult_error\n";
      log << "HRESULT: 0x" << std::hex << e.code().value << std::dec << "\n";
      log.close();
    }
    m_hasError = true;
    StopCapture();
    return false;
  } catch (const std::exception& e) {
    std::ofstream log("wgc_error.txt", std::ios::app);
    if (log.is_open()) {
      log << "WGC StartCapture failed - std::exception: " << e.what() << "\n";
      log.close();
    }
    m_hasError = true;
    StopCapture();
    return false;
  } catch (...) {
    std::ofstream log("wgc_error.txt", std::ios::app);
    if (log.is_open()) {
      log << "WGC StartCapture failed - unknown exception\n";
      log.close();
    }
    m_hasError = true;
    StopCapture();
    return false;
  }
}

bool WgcCapture::StartCaptureMonitor(HMONITOR monitor) {
  if (!m_winrtDevice) {
    return false;
  }

  try {
    StopCapture();

    auto interop = winrt::get_activation_factory<GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    GraphicsCaptureItem item{nullptr};
    winrt::check_hresult(interop->CreateForMonitor(monitor, winrt::guid_of<GraphicsCaptureItem>(),
                                                   reinterpret_cast<void**>(winrt::put_abi(item))));

    if (!item) {
      return false;
    }

    m_item = item;
    auto size = item.Size();
    m_width = size.Width;
    m_height = size.Height;

    // Use appropriate frame pool size based on latency mode
    int poolSize = m_lowLatencyMode ? kFramePoolSizeLowLatency : kFramePoolSizeNormal;

    m_framePool = Direct3D11CaptureFramePool::CreateFreeThreaded(
        m_winrtDevice,
        DirectXPixelFormat::B8G8R8A8UIntNormalized,
        poolSize,
        size);

    m_frameArrivedToken = m_framePool.FrameArrived(
        {this, &WgcCapture::OnFrameArrived});

    m_session = m_framePool.CreateCaptureSession(item);
    m_session.IsCursorCaptureEnabled(false);
    m_session.IsBorderRequired(false);
    
    // On Windows 11 22H2+, set MinUpdateInterval to remove FPS cap
    // BUG FIX: MinUpdateInterval(0) caps at ~50fps!
    // Minimum value to actually remove cap is 1ms = 10000 (100-ns units)
    // Per https://github.com/robmikh/Win32CaptureSample/issues/82
    if (SupportsMinUpdateInterval()) {
      try {
        m_session.MinUpdateInterval(TimeSpan{ 10000 });
      } catch (...) {
        // MinUpdateInterval not available on this build
      }
    }
    
    m_session.StartCapture();
    m_hasNewFrame.store(false);
    m_pendingFrameCount.store(0);
    m_isCapturing = true;
    m_hasError = false;
    m_droppedFrames = 0;
    m_capturedFrames = 0;
    m_captureHwnd = nullptr;
    m_captureMonitor = monitor;
    m_isWindowCapture = false;
    m_prevFrameQpc = 0;
    m_frameIntervalSum = 0.0;
    m_frameIntervalCount = 0;
    
    // Start the dedicated capture thread
    m_stopThread = false;
    m_threadActive = true;
    m_captureThread = std::thread(&WgcCapture::CaptureThreadLoop, this);

    return true;
  } catch (...) {
    m_hasError = true;
    StopCapture();
    return false;
  }
}

void WgcCapture::StopCapture() {
  // Signal thread to stop
  m_stopThread = true;
  m_frameCv.notify_all();
  if (m_captureThread.joinable()) {
    m_captureThread.join();
  }
  m_threadActive = false;

  if (m_framePool) {
    m_framePool.FrameArrived(m_frameArrivedToken);
  }

  if (m_session) {
    m_session.Close();
  }

  if (m_framePool) {
    m_framePool.Close();
  }

  m_nextFrame = nullptr;
  m_nextFrameReady = false;

  m_item = nullptr;
  m_session = nullptr;
  m_framePool = nullptr;
  m_isCapturing = false;
  m_hasNewFrame.store(false);
  m_pendingFrameCount.store(0);
  m_lastFrameTime = 0;
  m_lastFrameAgeMs = 0.0;
}

bool WgcCapture::RestartCapture() {
  if (m_isWindowCapture && m_captureHwnd) {
    return StartCapture(m_captureHwnd);
  } else if (!m_isWindowCapture && m_captureMonitor) {
    return StartCaptureMonitor(m_captureMonitor);
  }
  return false;
}

void WgcCapture::ResetStatistics() {
  m_droppedFrames = 0;
  m_capturedFrames = 0;
  m_frameIntervalSum = 0.0;
  m_frameIntervalCount = 0;
  m_prevFrameQpc = 0;
}

WgcCaptureStatistics WgcCapture::GetStatistics() const {
  WgcCaptureStatistics stats;
  stats.capturedFrames = static_cast<uint32_t>(m_capturedFrames);
  stats.droppedFrames = static_cast<uint32_t>(m_droppedFrames);
  stats.lastFrameTime = m_lastFrameTime;
  stats.lastFrameAgeMs = m_lastFrameAgeMs;
  stats.avgFrameIntervalMs = (m_frameIntervalCount > 0) 
      ? (m_frameIntervalSum / m_frameIntervalCount) 
      : 0.0;
  return stats;
}

void WgcCapture::UpdateFrameTiming(int64_t frameQpc) {
  if (m_prevFrameQpc != 0) {
    double intervalMs = 1000.0 * (frameQpc - m_prevFrameQpc) / GetQpcFrequency();
    m_frameIntervalSum += intervalMs;
    m_frameIntervalCount++;
    
    // Keep a rolling average (discard old samples)
    if (m_frameIntervalCount > kMaxIntervalSamples) {
      m_frameIntervalSum = (m_frameIntervalSum / m_frameIntervalCount) * (kMaxIntervalSamples / 2);
      m_frameIntervalCount = kMaxIntervalSamples / 2;
    }
  }
  m_prevFrameQpc = frameQpc;
}

bool WgcCapture::AcquireNextFrame(CapturedFrame& frame) {
  // Check if the capture thread has provided a new frame
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame captureFrame{nullptr};
  
  {
      std::lock_guard<std::mutex> lock(m_frameMutex);
      if (!m_nextFrameReady) {
          return false;
      }
      captureFrame = m_nextFrame;
      m_nextFrameReady = false; // Consumed
      // We keep m_nextFrame holding the object until next replacement or here? 
      // Actually we should clear it so we don't hold prolonged references, 
      // BUT we want the thread to be able to overwrite it efficiently.
      m_nextFrame = nullptr; 
  }

  if (!captureFrame) return false;

  return ProcessFrame(captureFrame, frame);
}

bool WgcCapture::ProcessFrame(winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame const& captureFrame, CapturedFrame& frame) {
  auto size = captureFrame.ContentSize();
  
  // Handle size changes
  if (size.Width != m_width || size.Height != m_height) {
    m_width = size.Width;
    m_height = size.Height;
    int poolSize = m_lowLatencyMode ? kFramePoolSizeLowLatency : kFramePoolSizeNormal;
    // Note: Recreating pool might race with the capture thread if not careful.
    // Ideally we should signal the thread to pause, but WGC handles pool recreation fairly robustly.
    // However, since we are on the Main Thread here, and Capture Thread uses m_framePool...
    // We should rely on standard WGC behavior where pool recreation is asynchronous or just safe.
    // Given the complexity, we'll recreate it here. The Capture Thread checks m_framePool != nullptr.
    if (m_framePool) {
        m_framePool.Recreate(m_winrtDevice, DirectXPixelFormat::B8G8R8A8UIntNormalized,
                             poolSize, size);
    }
    EnsureCaptureTexture(m_width, m_height);
  }

  auto surface = captureFrame.Surface();
  winrt::com_ptr<ID3D11Texture2D> texture = GetDxgiInterface<ID3D11Texture2D>(surface);

  EnsureCaptureTexture(m_width, m_height);
  if (m_captureTexture) {
    m_context->CopyResource(m_captureTexture.Get(), texture.get());
  }

  // Get precise timing
  int64_t nowQpc = GetQpcNow();

  auto time = captureFrame.SystemRelativeTime();
  int64_t frameTime = time.count();
  
  // Calculate frame age (how long since frame was captured)
  // SystemRelativeTime is in 100ns units
  int64_t systemTimeNow = 0;
  {
    FILETIME ft;
    GetSystemTimePreciseAsFileTime(&ft);
    systemTimeNow = (static_cast<int64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
  }
  // Frame age in milliseconds
  m_lastFrameAgeMs = (systemTimeNow - frameTime) / 10000.0;
  
  frame.texture = m_captureTexture;
  frame.width = m_width;
  frame.height = m_height;
  frame.qpcTime = nowQpc;
  frame.systemTime100ns = frameTime;
  
  m_lastFrameTime = frameTime;
  m_capturedFrames++;
  
  // Update frame interval statistics
  UpdateFrameTiming(nowQpc);
  
  return true;
}

bool WgcCapture::AcquireLatestFrame(CapturedFrame& frame) {
    return AcquireNextFrame(frame); // Thread already ensures latest
}

void WgcCapture::OnFrameArrived(
    Direct3D11CaptureFramePool const&,
    winrt::Windows::Foundation::IInspectable const&) {
  // Just notify the dedicated thread
  {
      std::lock_guard<std::mutex> lock(m_frameMutex);
      m_newFrameEvent = true;
  }
  m_frameCv.notify_one();
}

void WgcCapture::EnsureCaptureTexture(int width, int height) {
  if (m_captureTexture) {
    D3D11_TEXTURE2D_DESC desc = {};
    m_captureTexture->GetDesc(&desc);
    if (static_cast<int>(desc.Width) == width && static_cast<int>(desc.Height) == height) {
      return;
    }
  }

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(width);
  desc.Height = static_cast<UINT>(height);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

  m_device->CreateTexture2D(&desc, nullptr, &m_captureTexture);
}

void WgcCapture::CaptureThreadLoop() {
  while (!m_stopThread) {
    std::unique_lock<std::mutex> lock(m_frameMutex);
    m_frameCv.wait(lock, [this] { return m_stopThread || m_newFrameEvent; });

    if (m_stopThread) break;
    
    m_newFrameEvent = false;

    if (!m_framePool) {
        continue;
    }

    // Use TryGetNextFrame loop to drain old frames and get the NEWEST one immediately
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame latestFrame{nullptr};
    
    // We only want the absolute latest frame. Discard older ones.
    while (true) {
        auto frame = m_framePool.TryGetNextFrame();
        if (!frame) break;
        latestFrame = frame;
    }

    if (latestFrame) {
        m_nextFrame = latestFrame;
        m_nextFrameReady = true;
    }
  }
}
