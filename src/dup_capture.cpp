#include "dup_capture.h"
#include <intrin.h>
#include <dwmapi.h>
#include <algorithm>
#include <fstream>
#include <iostream>

#pragma comment(lib, "dwmapi.lib")

#include <ShellScalingApi.h>

namespace {
int64_t GetQpcNow() {
  LARGE_INTEGER qpc;
  QueryPerformanceCounter(&qpc);
  return qpc.QuadPart;
}
}

bool DupCapture::Initialize(ID3D11Device* device) {
  if (!device) {
    return false;
  }

  m_device = device;
  device->GetImmediateContext(&m_context);
  QueryPerformanceFrequency(&m_qpcFreq);
  return true;
}

void DupCapture::Shutdown() {
  StopCapture();
  m_context.Reset();
  m_device.Reset();
}

bool DupCapture::StartCapture(HWND hwnd,
                              Microsoft::WRL::ComPtr<IDXGIOutput> output,
                              const RECT& outputRect) {
  if (!m_device || !hwnd || !output) {
    return false;
  }

  StopCapture();

  m_hwnd = hwnd;
  m_output = output;
  
  // Get DPI for the monitor containing the window
  HMONITOR hMon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
  GetDpiForMonitor(hMon, MDT_EFFECTIVE_DPI, &m_dpiX, &m_dpiY);

  // Force update from the output directly to ensure coordinates are correct
  // (Ignore the passed outputRect which might be stale or incorrect from the caller)
  if (!UpdateOutputRect()) {
      StopCapture();
      return false;
  }

  // Double check dimensions
  m_outputWidth = m_outputRect.right - m_outputRect.left;
  m_outputHeight = m_outputRect.bottom - m_outputRect.top;
  
  if (m_outputWidth <= 0 || m_outputHeight <= 0) {
     StopCapture();
     return false; 
  }

  Microsoft::WRL::ComPtr<IDXGIOutput1> output1;
  if (FAILED(m_output.As(&output1))) {
    StopCapture();
    return false;
  }

  // Try IDXGIOutput5::DuplicateOutput1 first (better format support, Windows 10+)
  HRESULT hr = E_FAIL;
  Microsoft::WRL::ComPtr<IDXGIOutput5> output5;
  if (SUCCEEDED(m_output.As(&output5))) {
    DXGI_FORMAT supportedFormats[] = {
      DXGI_FORMAT_B8G8R8A8_UNORM,
      DXGI_FORMAT_R8G8B8A8_UNORM,
      DXGI_FORMAT_R10G10B10A2_UNORM,
      DXGI_FORMAT_R16G16B16A16_FLOAT
    };
    hr = output5->DuplicateOutput1(m_device.Get(), 0, 
                                   _countof(supportedFormats), supportedFormats, 
                                   &m_duplication);
    if (SUCCEEDED(hr)) {
      m_useOutput5 = true;
    }
  }

  // Fallback to IDXGIOutput1::DuplicateOutput
  if (!m_duplication) {
    hr = output1->DuplicateOutput(m_device.Get(), &m_duplication);
    m_useOutput5 = false;
  }

  if (FAILED(hr)) {
    StopCapture();
    return false;
  }

  m_isCapturing = true;
  return true;
}

void DupCapture::StopCapture() {
  m_duplication.Reset();
  m_output.Reset();
  m_captureTexture.Reset();
  m_stagingTexture.Reset();
  m_hwnd = nullptr;
  m_isCapturing = false;
  m_outputWidth = 0;
  m_outputHeight = 0;
  m_outputRect = {};
  m_useOutput5 = false;
}

bool DupCapture::AcquireNextFrame(CapturedFrame& frame) {
  if (!m_duplication || !m_context || !m_hwnd) {
    return false;
  }

  // Allow GetDesktopWindow() for capture (used in DXGI Crop Mode)
  bool isDesktop = (m_hwnd == GetDesktopWindow());
  if (!isDesktop && !IsWindow(m_hwnd)) {
    return false;
  }

  DXGI_OUTDUPL_FRAME_INFO frameInfo = {};
  Microsoft::WRL::ComPtr<IDXGIResource> resource;
  HRESULT hr = DXGI_ERROR_WAIT_TIMEOUT;

  // SPIN WAIT MODE - spin until we get a frame (for high-FPS capture)
  if (m_spinWaitMode) {
      int64_t startQpc = GetQpcNow();
      int64_t maxTicks = (m_qpcFreq.QuadPart * m_spinWaitMs) / 1000;
      int attempts = 0;
      
      while (true) {
          hr = m_duplication->AcquireNextFrame(0, &frameInfo, &resource);
          if (hr != DXGI_ERROR_WAIT_TIMEOUT) break;

          if ((GetQpcNow() - startQpc) > maxTicks) {
            // Keep trying even after timeout in spin-wait mode for high-FPS
            // But yield occasionally to prevent CPU saturation
            if (++attempts % 10 == 0) {
              Sleep(0);
            }
            continue;
          }
          break;
      }
  } else if (m_pollingMode) {
      // Polling mode with timeout=0 - return immediately if no frame
      hr = m_duplication->AcquireNextFrame(0, &frameInfo, &resource);
  } else {
      // Normal blocking mode - wait up to 16ms (roughly 60fps)
      UINT timeout = 16; 
      hr = m_duplication->AcquireNextFrame(timeout, &frameInfo, &resource);
  }

  if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
    return false;
  }
  if (hr == DXGI_ERROR_ACCESS_LOST) {
    StopCapture();
    return false;
  }
  if (FAILED(hr)) {
    return false;
  }

  bool success = false;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> frameTexture;
  if (SUCCEEDED(resource.As(&frameTexture))) {
    D3D11_TEXTURE2D_DESC srcDesc = {};
    frameTexture->GetDesc(&srcDesc);

    // Validate source texture
    if (srcDesc.Width == 0 || srcDesc.Height == 0) {
      m_duplication->ReleaseFrame();
      return false;
    }

    // Accept more formats when using Output5
    bool validFormat = (srcDesc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                        srcDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM);
    if (m_useOutput5) {
      validFormat = validFormat || 
                    srcDesc.Format == DXGI_FORMAT_R10G10B10A2_UNORM ||
                    srcDesc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
    
    if (!validFormat) {
      m_duplication->ReleaseFrame();
      return false;
    }

    RECT screenRect = {};
    bool haveRect = false;

    // DXGI CROP MODE MODIFICATION:
    // If we are capturing Desktop (DXGI Crop Mode), simply capture the ENTIRE OUTPUT.
    // The cropping is done in App.cpp later using the original window coordinates.
    if (isDesktop) {
        haveRect = true;
        screenRect.left = m_outputRect.left;
        screenRect.top = m_outputRect.top;
        screenRect.right = m_outputRect.right;
        screenRect.bottom = m_outputRect.bottom;
    } else {
        // ... Original window tracking logic ...
        // 1. Try Client Area (Game Content only) - Preferred for gaming
        RECT clientRect = {};
        if (GetClientRect(m_hwnd, &clientRect)) {
            POINT topLeft{clientRect.left, clientRect.top};
            POINT bottomRight{clientRect.right, clientRect.bottom};
            if (ClientToScreen(m_hwnd, &topLeft) && ClientToScreen(m_hwnd, &bottomRight)) {
                 screenRect = {topLeft.x, topLeft.y, bottomRight.x, bottomRight.y};
                 haveRect = (screenRect.right > screenRect.left) && (screenRect.bottom > screenRect.top);
            }
        }
    
        // 2. Fallback to Visual Window Bounds (Frame + Content) if Client Area fails
        if (!haveRect) {
            if (SUCCEEDED(DwmGetWindowAttribute(m_hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, &screenRect, sizeof(screenRect)))) {
                haveRect = true;
            } else if (GetWindowRect(m_hwnd, &screenRect)) {
                haveRect = true;
            }
        }
    }

    if (haveRect) {
        if (m_outputWidth <= 0 || m_outputHeight <= 0) {
          UpdateOutputRect();
        }

        RECT local = {
            screenRect.left - m_outputRect.left,
            screenRect.top - m_outputRect.top,
            screenRect.right - m_outputRect.left,
            screenRect.bottom - m_outputRect.top};

        // Clamp to source texture bounds (not just output bounds)
        int srcW = static_cast<int>(srcDesc.Width);
        int srcH = static_cast<int>(srcDesc.Height);
        local.left = (std::max)(0L, (std::min)(local.left, (LONG)srcW));
        local.right = (std::max)(0L, (std::min)(local.right, (LONG)srcW));
        local.top = (std::max)(0L, (std::min)(local.top, (LONG)srcH));
        local.bottom = (std::max)(0L, (std::min)(local.bottom, (LONG)srcH));

        if (local.right > local.left && local.bottom > local.top) {
          int width = local.right - local.left;
          int height = local.bottom - local.top;

          EnsureCaptureTexture(width, height, srcDesc.Format);
          if (m_captureTexture) {
            D3D11_BOX box = {};
            box.left = static_cast<UINT>(local.left);
            box.top = static_cast<UINT>(local.top);
            box.front = 0;
            box.right = static_cast<UINT>(local.right);
            box.bottom = static_cast<UINT>(local.bottom);
            box.back = 1;
            m_context->CopySubresourceRegion(
                m_captureTexture.Get(), 0, 0, 0, 0, frameTexture.Get(), 0, &box);

            LARGE_INTEGER qpc = {};
            QueryPerformanceCounter(&qpc);
            frame.texture = m_captureTexture;
            frame.width = width;
            frame.height = height;
            frame.qpcTime = qpc.QuadPart;
            if (m_qpcFreq.QuadPart > 0) {
              double qpcTo100ns = 1e7 / static_cast<double>(m_qpcFreq.QuadPart);
              frame.systemTime100ns =
                  static_cast<int64_t>(static_cast<double>(qpc.QuadPart) * qpcTo100ns);
            } else {
              frame.systemTime100ns = 0;
            }
            success = true;
          }
        }
      }
  }

  m_duplication->ReleaseFrame();
  return success;
}

void DupCapture::EnsureCaptureTexture(int width, int height, DXGI_FORMAT format) {
  if (width <= 0 || height <= 0) {
    return;
  }

  DXGI_FORMAT targetFormat = (format == DXGI_FORMAT_UNKNOWN) ? DXGI_FORMAT_B8G8R8A8_UNORM : format;

  if (m_captureTexture) {
    D3D11_TEXTURE2D_DESC desc = {};
    m_captureTexture->GetDesc(&desc);
    if (static_cast<int>(desc.Width) == width && static_cast<int>(desc.Height) == height &&
        desc.Format == targetFormat) {
      return;
    }
  }

  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = static_cast<UINT>(width);
  desc.Height = static_cast<UINT>(height);
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = targetFormat;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

  m_device->CreateTexture2D(&desc, nullptr, &m_captureTexture);
}

bool DupCapture::UpdateOutputRect() {
  if (!m_output) {
    return false;
  }

  DXGI_OUTPUT_DESC desc = {};
  if (FAILED(m_output->GetDesc(&desc))) {
    return false;
  }

  m_outputRect = desc.DesktopCoordinates;
  m_outputWidth = m_outputRect.right - m_outputRect.left;
  m_outputHeight = m_outputRect.bottom - m_outputRect.top;
  return m_outputWidth > 0 && m_outputHeight > 0;
}
