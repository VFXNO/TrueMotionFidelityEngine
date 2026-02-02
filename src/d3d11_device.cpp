#include "d3d11_device.h"

#include <windows.h>
#include <dxgi1_6.h>

#include <cstdint>
#include <vector>
#include <iostream>

#pragma comment(lib, "dxgi.lib")

bool D3D11Device::Initialize(HWND hwnd) {
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
  flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

  // Full DXGI Support Upgrade: 
  // 1. Create Factory first
  // 2. Enumerate Adapters to find High Performance GPU
  // 3. Create Device on that Adapter
  
  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&m_factory));
  if (FAILED(hr)) {
      return false;
  }

  Microsoft::WRL::ComPtr<IDXGIAdapter> targetAdapter;
  Microsoft::WRL::ComPtr<IDXGIFactory6> factory6;
  if (SUCCEEDED(m_factory.As(&factory6))) {
      // DXGI 1.6: Ask for High Performance GPU specifically
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
      if (SUCCEEDED(factory6->EnumAdapterByGpuPreference(
              0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter1)))) {
          targetAdapter = adapter1;
      }
  }

  // Fallback if DXGI 1.6 fails or returns null
  if (!targetAdapter) {
     m_factory->EnumAdapters(0, &targetAdapter);
  }

  D3D_FEATURE_LEVEL featureLevels[] = {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
  };

  D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
  
  // Create device with the specific high-perf adapter
  hr = D3D11CreateDevice(
      targetAdapter.Get(), 
      targetAdapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, // UNKNOWN needed when providing adapter
      nullptr,
      flags,
      featureLevels,
      ARRAYSIZE(featureLevels),
      D3D11_SDK_VERSION,
      &m_device,
      &featureLevel,
      &m_context);

  if (FAILED(hr)) {
    // Last ditch fallback: Try default adapter if specific choice failed
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &m_device,
        &featureLevel,
        &m_context);
        
    if (FAILED(hr)) return false;
  }

  Microsoft::WRL::ComPtr<IDXGIDevice1> dxgiDevice1;
  if (SUCCEEDED(m_device.As(&dxgiDevice1))) {
    // Give our GPU main thread absolute priority
    dxgiDevice1->SetGPUThreadPriority(7);
  }

  // Reload factory from device just to ensure sync (though we created it above)
  Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
  if (FAILED(m_device.As(&dxgiDevice))) return false;

  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  if (FAILED(dxgiDevice->GetAdapter(&adapter))) return false;
  
  // Update m_factory cleanly from the used adapter
  if (FAILED(adapter->GetParent(IID_PPV_ARGS(&m_factory)))) return false;

  Microsoft::WRL::ComPtr<IDXGIFactory5> factory5;
  if (SUCCEEDED(m_factory.As(&factory5))) {
    BOOL allowTearing = FALSE;
    if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                                                &allowTearing,
                                                sizeof(allowTearing)))) {
      m_allowTearing = (allowTearing == TRUE);
    }
  }

  EnumerateMonitors();
  if (m_monitors.empty()) {
    return false;
  }

  RECT rect = m_monitors[0].rect;
  UINT width = static_cast<UINT>(rect.right - rect.left);
  UINT height = static_cast<UINT>(rect.bottom - rect.top);
  if (!CreateSwapChain(hwnd, width, height)) {
    return false;
  }

  return true;
}

void D3D11Device::Shutdown() {
  m_rtv.Reset();
  m_swapChain.Reset();
  m_factory.Reset();
  m_context.Reset();
  m_device.Reset();
  m_monitors.clear();
}

bool D3D11Device::ResizeSwapChain(UINT width, UINT height) {
  if (!m_swapChain) {
    return false;
  }

  m_rtv.Reset();
  UINT flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
  if (m_allowTearing) {
    flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
  }
  HRESULT hr = m_swapChain->ResizeBuffers(
      0,
      width,
      height,
      DXGI_FORMAT_B8G8R8A8_UNORM,
      flags);
  if (FAILED(hr)) {
    return false;
  }
  
  Microsoft::WRL::ComPtr<IDXGISwapChain2> swapChain2;
  if (SUCCEEDED(m_swapChain.As(&swapChain2))) {
    // LATENCY FIX: Strictly enforce 2 frames of latency (Double Buffering)
    swapChain2->SetMaximumFrameLatency(2);
    m_swapChainWaitHandle = swapChain2->GetFrameLatencyWaitableObject();
  }

  UpdateRenderTarget();
  return true;
}

bool D3D11Device::CreateSwapChainForWindow(
    HWND hwnd,
    UINT width,
    UINT height,
    Microsoft::WRL::ComPtr<IDXGISwapChain1>& swapChain,
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView>& rtv) {
  if (!m_factory || !m_device) {
    return false;
  }

  DXGI_SWAP_CHAIN_DESC1 desc = {};
  desc.Width = width;
  desc.Height = height;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 2;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.Scaling = DXGI_SCALING_STRETCH;
  desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
  desc.Flags = m_allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

  Microsoft::WRL::ComPtr<IDXGISwapChain1> localSwapChain;
  HRESULT hr = m_factory->CreateSwapChainForHwnd(
      m_device.Get(),
      hwnd,
      &desc,
      nullptr,
      nullptr,
      &localSwapChain);
  if (FAILED(hr)) {
    return false;
  }

  // LATENCY FIX: Enforce 2 frame latency (Double Buffering)
  // Latency 1 is too strict for the heavy 7x7 interpolation and causes drops on small spikes.
  // Latency 2 is the sweet spot: It absorbs one frame of render-fluctuation while still being very fast.
  Microsoft::WRL::ComPtr<IDXGISwapChain2> swapChain2;
  if (SUCCEEDED(localSwapChain.As(&swapChain2))) {
    swapChain2->SetMaximumFrameLatency(2);
    // If this is the main output swapchain, we will get the waitable object later via GetSwapChainWaitHandle
  }

  m_factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

  Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
  hr = localSwapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
  if (FAILED(hr)) {
    return false;
  }

  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> localRtv;
  hr = m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &localRtv);
  if (FAILED(hr)) {
    return false;
  }

  swapChain = localSwapChain;
  rtv = localRtv;

  // LATENCY FIX: Ensure UI/Preview window also has minimal latency
  // Note: We already set the latency to 2 above (line 158), so we don't need to do it again here.
  // The swap chain object is the same.
  
  return true;
}

bool D3D11Device::SetMonitor(int index, HWND hwnd) {
  if (index < 0 || index >= static_cast<int>(m_monitors.size())) {
    return false;
  }

  RECT rect = m_monitors[index].rect;
  UINT width = static_cast<UINT>(rect.right - rect.left);
  UINT height = static_cast<UINT>(rect.bottom - rect.top);

  SetWindowPos(hwnd, HWND_TOP, rect.left, rect.top, width, height,
               SWP_NOZORDER | SWP_NOACTIVATE);

  return ResizeSwapChain(width, height);
}

float D3D11Device::RefreshHz(int index) const {
  if (index < 0 || index >= static_cast<int>(m_monitors.size())) {
    return 0.0f;
  }

  const auto& rate = m_monitors[index].refreshRate;
  if (rate.Denominator == 0) {
    return 0.0f;
  }
  return static_cast<float>(rate.Numerator) / static_cast<float>(rate.Denominator);
}

float D3D11Device::MaxRefreshHz(int index) const {
  if (index < 0 || index >= static_cast<int>(m_monitors.size())) {
    return 0.0f;
  }

  const auto& rate = m_monitors[index].maxRefreshRate;
  if (rate.Denominator == 0) {
    return 0.0f;
  }
  return static_cast<float>(rate.Numerator) / static_cast<float>(rate.Denominator);
}

RECT D3D11Device::MonitorRect(int index) const {
  if (index < 0 || index >= static_cast<int>(m_monitors.size())) {
    return RECT{};
  }
  return m_monitors[index].rect;
}

HMONITOR D3D11Device::MonitorHandle(int index) const {
  if (index < 0 || index >= static_cast<int>(m_monitors.size())) {
    return nullptr;
  }
  return m_monitors[index].monitor;
}

bool D3D11Device::OutputForMonitor(HMONITOR monitor,
                                   Microsoft::WRL::ComPtr<IDXGIOutput>& output,
                                   RECT& rect) const {
  if (!monitor) {
    return false;
  }

  for (const auto& info : m_monitors) {
    if (info.monitor == monitor) {
      output = info.output;
      rect = info.rect;
      return output != nullptr;
    }
  }

  return false;
}

bool D3D11Device::CreateSwapChain(HWND hwnd, UINT width, UINT height) {
  DXGI_SWAP_CHAIN_DESC1 desc = {};
  desc.Width = width;
  desc.Height = height;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 2; // Double Buffering
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.Scaling = DXGI_SCALING_NONE; // change it to fullscreen scale 
  desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
  desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
  if (m_allowTearing) {
    desc.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
  }

  HRESULT hr = m_factory->CreateSwapChainForHwnd(
      m_device.Get(),
      hwnd,
      &desc,
      nullptr,
      nullptr,
      &m_swapChain);
  if (FAILED(hr)) {
    return false;
  }
  
  Microsoft::WRL::ComPtr<IDXGISwapChain2> swapChain2;
  if (SUCCEEDED(m_swapChain.As(&swapChain2))) {
    // LATENCY FIX: Strict 2 frame latency
    swapChain2->SetMaximumFrameLatency(2);
    m_swapChainWaitHandle = swapChain2->GetFrameLatencyWaitableObject();
  }

  m_factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
  UpdateRenderTarget();
  return true;
}

void D3D11Device::UpdateRenderTarget() {
  Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
  HRESULT hr = m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
  if (FAILED(hr)) {
    return;
  }

  m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_rtv);
}

void D3D11Device::EnumerateMonitors() {
  m_monitors.clear();

  Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
  if (FAILED(m_device.As(&dxgiDevice))) {
    return;
  }

  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  if (FAILED(dxgiDevice->GetAdapter(&adapter))) {
    return;
  }

  for (UINT i = 0;; ++i) {
    Microsoft::WRL::ComPtr<IDXGIOutput> output;
    if (adapter->EnumOutputs(i, &output) == DXGI_ERROR_NOT_FOUND) {
      break;
    }

    DXGI_OUTPUT_DESC desc = {};
    if (FAILED(output->GetDesc(&desc))) {
      continue;
    }

    MonitorInfo info;
    info.name = desc.DeviceName;
    info.rect = desc.DesktopCoordinates;
    info.output = output;
    info.monitor = desc.Monitor;

    DEVMODEW devMode = {};
    devMode.dmSize = sizeof(DEVMODEW);
    if (EnumDisplaySettingsW(desc.DeviceName, ENUM_CURRENT_SETTINGS, &devMode)) {
      if (devMode.dmDisplayFrequency > 1) {
        info.refreshRate.Numerator = devMode.dmDisplayFrequency;
        info.refreshRate.Denominator = 1;
      }
    }

    UINT modeCount = 0;
    output->GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, 0, &modeCount, nullptr);
    if (modeCount > 0) {
      std::vector<DXGI_MODE_DESC> modes(modeCount);
      if (SUCCEEDED(output->GetDisplayModeList(
              DXGI_FORMAT_B8G8R8A8_UNORM, 0, &modeCount, modes.data()))) {
        int width = info.rect.right - info.rect.left;
        int height = info.rect.bottom - info.rect.top;
        DXGI_RATIONAL best = {0, 1};
        for (const auto& mode : modes) {
          if (static_cast<int>(mode.Width) != width ||
              static_cast<int>(mode.Height) != height) {
            continue;
          }

          if (best.Numerator == 0 ||
              static_cast<uint64_t>(mode.RefreshRate.Numerator) * best.Denominator >
                  static_cast<uint64_t>(best.Numerator) * mode.RefreshRate.Denominator) {
            best = mode.RefreshRate;
          }
        }
        if (best.Numerator == 0) {
          best = modes[0].RefreshRate;
        }
        info.maxRefreshRate = best;
        if (info.refreshRate.Numerator == 0) {
          info.refreshRate = best;
        }
      }
    }

    m_monitors.push_back(info);
  }
}
