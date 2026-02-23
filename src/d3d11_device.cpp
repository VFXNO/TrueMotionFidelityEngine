#include "d3d11_device.h"

#include <windows.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#pragma comment(lib, "dxgi.lib")

namespace {

std::string HrToHex(HRESULT hr) {
  std::ostringstream ss;
  ss << "0x" << std::hex << std::uppercase << static_cast<unsigned long>(hr);
  return ss.str();
}

void AppendInitLog(std::ofstream& log, const std::string& line) {
  if (log.is_open()) {
    log << line << "\n";
  }
}

bool LuidEquals(const LUID& a, const LUID& b) {
  return a.HighPart == b.HighPart && a.LowPart == b.LowPart;
}

bool AdapterHasOutputs(IDXGIAdapter1* adapter) {
  if (!adapter) {
    return false;
  }

  Microsoft::WRL::ComPtr<IDXGIOutput> output;
  return SUCCEEDED(adapter->EnumOutputs(0, &output));
}

std::string AdapterName(IDXGIAdapter1* adapter) {
  if (!adapter) {
    return "default";
  }

  DXGI_ADAPTER_DESC1 desc = {};
  if (FAILED(adapter->GetDesc1(&desc))) {
    return "unknown-adapter";
  }

  char utf8[512] = {};
  int written = WideCharToMultiByte(
      CP_UTF8,
      0,
      desc.Description,
      -1,
      utf8,
      static_cast<int>(sizeof(utf8)),
      nullptr,
      nullptr);

  if (written <= 1) {
    return "unnamed-adapter";
  }
  return utf8;
}

std::string DeviceAdapterName(ID3D11Device* device) {
  if (!device) {
    return {};
  }

  Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
  if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDevice)))) {
    return {};
  }

  Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
  if (FAILED(dxgiDevice->GetAdapter(&adapter))) {
    return {};
  }

  Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
  if (FAILED(adapter.As(&adapter1))) {
    return {};
  }

  return AdapterName(adapter1.Get());
}

std::wstring ToLowerAscii(std::wstring value) {
  std::transform(value.begin(), value.end(), value.begin(), [](wchar_t c) {
    if (c >= L'A' && c <= L'Z') {
      return static_cast<wchar_t>(c - L'A' + L'a');
    }
    return c;
  });
  return value;
}

} // namespace

bool D3D11Device::Initialize(HWND hwnd) {
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
  flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

  std::ofstream initLog("init_log.txt", std::ios::app);
  AppendInitLog(initLog, "\n[D3D11Device::Initialize] begin");
  m_hasDxgiOutputs = false;
  m_usingSystemMonitorFallback = false;
  m_activeAdapterName.clear();

  wchar_t prefBuf[64] = {};
  DWORD prefLen = GetEnvironmentVariableW(
      L"TMFE_GPU_PREFERENCE",
      prefBuf,
      static_cast<DWORD>(ARRAYSIZE(prefBuf)));
  std::wstring gpuPref;
  if (prefLen > 0 && prefLen < ARRAYSIZE(prefBuf)) {
    gpuPref = ToLowerAscii(std::wstring(prefBuf));
  }
  const bool preferHighPerformance =
      (gpuPref == L"high_performance" || gpuPref == L"high-performance" ||
       gpuPref == L"high" || gpuPref == L"dgpu");
  const bool requireOutputAdapters = !preferHighPerformance;
  if (preferHighPerformance) {
    AppendInitLog(initLog, "GPU preference: high_performance (TMFE_GPU_PREFERENCE)");
  } else {
    AppendInitLog(initLog, "GPU preference: display_attached (default)");
  }

  HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&m_factory));
  if (FAILED(hr)) {
    AppendInitLog(initLog, "CreateDXGIFactory1 failed: " + HrToHex(hr));
    return false;
  }
  AppendInitLog(initLog, "CreateDXGIFactory1 succeeded");

  Microsoft::WRL::ComPtr<IDXGIFactory2> adapterEnumFactory = m_factory;
  D3D_FEATURE_LEVEL featureLevels[] = {
      D3D_FEATURE_LEVEL_11_1,
      D3D_FEATURE_LEVEL_11_0,
  };
  D3D_FEATURE_LEVEL selectedFeatureLevel = D3D_FEATURE_LEVEL_11_0;

  auto tryCreateDevice = [&](IDXGIAdapter1* adapter1, D3D_DRIVER_TYPE driverType, const std::string& label) -> bool {
    m_device.Reset();
    m_context.Reset();

    IDXGIAdapter* baseAdapter = adapter1;
    HRESULT createHr = D3D11CreateDevice(
        baseAdapter,
        driverType,
        nullptr,
        flags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &m_device,
        &selectedFeatureLevel,
        &m_context);
    AppendInitLog(initLog, label + " D3D11CreateDevice: " + HrToHex(createHr));

#if defined(_DEBUG)
    if (createHr == DXGI_ERROR_SDK_COMPONENT_MISSING &&
        (flags & D3D11_CREATE_DEVICE_DEBUG) != 0) {
      UINT noDebugFlags = flags & ~D3D11_CREATE_DEVICE_DEBUG;
      createHr = D3D11CreateDevice(
          baseAdapter,
          driverType,
          nullptr,
          noDebugFlags,
          featureLevels,
          ARRAYSIZE(featureLevels),
          D3D11_SDK_VERSION,
          &m_device,
          &selectedFeatureLevel,
          &m_context);
      AppendInitLog(initLog, label + " retry without debug layer: " + HrToHex(createHr));
    }
#endif

    if (FAILED(createHr)) {
      return false;
    }

    AppendInitLog(initLog, label + " feature level: " + std::to_string(static_cast<int>(selectedFeatureLevel)));
    return true;
  };

  auto syncFactoryAndMonitors = [&]() -> bool {
    Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
    HRESULT localHr = m_device.As(&dxgiDevice);
    if (FAILED(localHr)) {
      AppendInitLog(initLog, "Query IDXGIDevice failed: " + HrToHex(localHr));
      return false;
    }

    Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
    localHr = dxgiDevice->GetAdapter(&adapter);
    if (FAILED(localHr)) {
      AppendInitLog(initLog, "GetAdapter failed: " + HrToHex(localHr));
      return false;
    }

    localHr = adapter->GetParent(IID_PPV_ARGS(&m_factory));
    if (FAILED(localHr)) {
      AppendInitLog(initLog, "GetParent(IDXGIFactory2) failed: " + HrToHex(localHr));
      return false;
    }

    Microsoft::WRL::ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(m_factory.As(&factory5))) {
      BOOL allowTearing = FALSE;
      if (SUCCEEDED(factory5->CheckFeatureSupport(
              DXGI_FEATURE_PRESENT_ALLOW_TEARING,
              &allowTearing,
              sizeof(allowTearing)))) {
        m_allowTearing = (allowTearing == TRUE);
      }
    }

    EnumerateMonitors();
    m_hasDxgiOutputs = !m_monitors.empty();
    if (!m_hasDxgiOutputs) {
      AppendInitLog(initLog, "No DXGI outputs on selected adapter, using system monitor fallback");
      EnumerateSystemMonitors();
      m_usingSystemMonitorFallback = !m_monitors.empty();
    } else {
      m_usingSystemMonitorFallback = false;
    }

    const std::string deviceAdapterName = DeviceAdapterName(m_device.Get());
    if (!deviceAdapterName.empty()) {
      m_activeAdapterName = deviceAdapterName;
    }

    AppendInitLog(initLog, "EnumerateMonitors count: " + std::to_string(m_monitors.size()));
    AppendInitLog(initLog, "DXGI outputs available: " + std::string(m_hasDxgiOutputs ? "true" : "false"));
    if (!m_activeAdapterName.empty()) {
      AppendInitLog(initLog, "Active adapter: " + m_activeAdapterName);
    }
    return !m_monitors.empty();
  };

  std::vector<Microsoft::WRL::ComPtr<IDXGIAdapter1>> adapterCandidates;

  Microsoft::WRL::ComPtr<IDXGIFactory6> factory6;
  if (SUCCEEDED(adapterEnumFactory.As(&factory6))) {
    for (UINT i = 0;; ++i) {
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
      HRESULT enumHr = factory6->EnumAdapterByGpuPreference(
          i,
          DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
          IID_PPV_ARGS(&adapter1));
      if (enumHr == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      if (SUCCEEDED(enumHr)) {
        adapterCandidates.push_back(adapter1);
      }
    }
  }

  for (UINT i = 0;; ++i) {
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter1;
    HRESULT enumHr = adapterEnumFactory->EnumAdapters1(i, &adapter1);
    if (enumHr == DXGI_ERROR_NOT_FOUND) {
      break;
    }
    if (SUCCEEDED(enumHr)) {
      adapterCandidates.push_back(adapter1);
    }
  }

  bool created = false;
  std::vector<LUID> seenLuids;
  for (const auto& adapter1 : adapterCandidates) {
    if (!adapter1) {
      continue;
    }

    DXGI_ADAPTER_DESC1 desc = {};
    if (FAILED(adapter1->GetDesc1(&desc))) {
      continue;
    }
    if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
      continue;
    }

    bool alreadySeen = false;
    for (const auto& luid : seenLuids) {
      if (LuidEquals(desc.AdapterLuid, luid)) {
        alreadySeen = true;
        break;
      }
    }
    if (alreadySeen) {
      continue;
    }
    seenLuids.push_back(desc.AdapterLuid);

    const bool hasOutputs = AdapterHasOutputs(adapter1.Get());
    const std::string adapterName = AdapterName(adapter1.Get());
    AppendInitLog(initLog,
                  "Trying adapter [" + adapterName + "] hasOutputs=" + (hasOutputs ? "true" : "false"));
    if (requireOutputAdapters && !hasOutputs) {
      continue;
    }

    if (tryCreateDevice(adapter1.Get(), D3D_DRIVER_TYPE_UNKNOWN, "Adapter " + adapterName) &&
        syncFactoryAndMonitors()) {
      created = true;
      break;
    }
  }

  if (!created) {
    AppendInitLog(initLog, "Trying default hardware device");
    if (tryCreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, "Default hardware") &&
        syncFactoryAndMonitors()) {
      created = true;
    }
  }

  if (!created) {
    AppendInitLog(initLog, "Trying WARP fallback device");
    if (tryCreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, "WARP fallback") &&
        syncFactoryAndMonitors()) {
      created = true;
    }
  }

  if (!created) {
    AppendInitLog(initLog, "All D3D11 device initialization paths failed");
    return false;
  }

  Microsoft::WRL::ComPtr<IDXGIDevice1> dxgiDevice1;
  if (SUCCEEDED(m_device.As(&dxgiDevice1))) {
    // Give our GPU main thread absolute priority
    dxgiDevice1->SetGPUThreadPriority(7);
  }

  RECT rect = m_monitors[0].rect;
  UINT width = static_cast<UINT>(rect.right - rect.left);
  UINT height = static_cast<UINT>(rect.bottom - rect.top);
  if (!CreateSwapChain(hwnd, width, height)) {
    AppendInitLog(initLog, "CreateSwapChain failed");
    return false;
  }

  AppendInitLog(initLog, "D3D11Device::Initialize succeeded");
  return true;
}

void D3D11Device::Shutdown() {
  m_rtv.Reset();
  m_swapChain.Reset();
  m_factory.Reset();
  m_context.Reset();
  m_device.Reset();
  m_monitors.clear();
  m_hasDxgiOutputs = false;
  m_usingSystemMonitorFallback = false;
  m_activeAdapterName.clear();
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
  desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

  UINT baseFlags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
  if (m_allowTearing) {
    baseFlags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
  }

  auto tryCreateSwapChain = [&](DXGI_SCALING scaling, UINT flags, const char* tag) -> HRESULT {
    desc.Scaling = scaling;
    desc.Flags = flags;
    m_swapChain.Reset();

    HRESULT localHr = m_factory->CreateSwapChainForHwnd(
        m_device.Get(),
        hwnd,
        &desc,
        nullptr,
        nullptr,
        &m_swapChain);

    if (FAILED(localHr)) {
      std::ofstream log("init_log.txt", std::ios::app);
      if (log.is_open()) {
        log << "CreateSwapChainForHwnd(" << tag << ") failed: " << HrToHex(localHr) << "\n";
      }
    }
    return localHr;
  };

  HRESULT hr = tryCreateSwapChain(DXGI_SCALING_NONE, baseFlags, "SCALING_NONE");
  if (FAILED(hr)) {
    hr = tryCreateSwapChain(DXGI_SCALING_STRETCH, baseFlags, "SCALING_STRETCH");
  }
  if (FAILED(hr)) {
    UINT noLatencyFlags = baseFlags & ~DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    hr = tryCreateSwapChain(DXGI_SCALING_STRETCH, noLatencyFlags, "SCALING_STRETCH_NO_WAITABLE");
  }
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

void D3D11Device::EnumerateSystemMonitors() {
  m_monitors.clear();

  struct EnumState {
    std::vector<MonitorInfo>* monitors = nullptr;
  } state;
  state.monitors = &m_monitors;

  EnumDisplayMonitors(
      nullptr,
      nullptr,
      [](HMONITOR monitor, HDC, LPRECT, LPARAM lParam) -> BOOL {
        auto* enumState = reinterpret_cast<EnumState*>(lParam);
        if (!enumState || !enumState->monitors) {
          return FALSE;
        }

        MONITORINFOEXW infoEx = {};
        infoEx.cbSize = sizeof(infoEx);
        if (!GetMonitorInfoW(monitor, &infoEx)) {
          return TRUE;
        }

        MonitorInfo info;
        info.name = infoEx.szDevice;
        info.rect = infoEx.rcMonitor;
        info.monitor = monitor;

        DEVMODEW devMode = {};
        devMode.dmSize = sizeof(devMode);
        if (EnumDisplaySettingsW(infoEx.szDevice, ENUM_CURRENT_SETTINGS, &devMode) &&
            devMode.dmDisplayFrequency > 1) {
          info.refreshRate.Numerator = devMode.dmDisplayFrequency;
          info.refreshRate.Denominator = 1;
          info.maxRefreshRate = info.refreshRate;
        }

        enumState->monitors->push_back(info);
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&state));
}
