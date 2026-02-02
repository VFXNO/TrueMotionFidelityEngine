#pragma once

#include <d3d11.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <string>
#include <vector>

struct MonitorInfo {
  std::wstring name;
  RECT rect = {};
  DXGI_RATIONAL refreshRate = {0, 1};
  DXGI_RATIONAL maxRefreshRate = {0, 1};
  HMONITOR monitor = nullptr;
  Microsoft::WRL::ComPtr<IDXGIOutput> output;
};

class D3D11Device {
public:
  bool Initialize(HWND hwnd);
  void Shutdown();

  bool ResizeSwapChain(UINT width, UINT height);
  bool SetMonitor(int index, HWND hwnd);
  bool CreateSwapChainForWindow(
      HWND hwnd,
      UINT width,
      UINT height,
      Microsoft::WRL::ComPtr<IDXGISwapChain1>& swapChain,
      Microsoft::WRL::ComPtr<ID3D11RenderTargetView>& rtv);

  ID3D11Device* Device() const { return m_device.Get(); }
  ID3D11DeviceContext* Context() const { return m_context.Get(); }
  IDXGISwapChain1* SwapChain() const { return m_swapChain.Get(); }
  ID3D11RenderTargetView* RenderTargetView() const { return m_rtv.Get(); }
  bool AllowTearing() const { return m_allowTearing; }
  HANDLE GetSwapChainWaitHandle() const { return m_swapChainWaitHandle; }

  const std::vector<MonitorInfo>& Monitors() const { return m_monitors; }
  int MonitorCount() const { return static_cast<int>(m_monitors.size()); }
  float RefreshHz(int index) const;
  float MaxRefreshHz(int index) const;
  RECT MonitorRect(int index) const;
  HMONITOR MonitorHandle(int index) const;
  bool OutputForMonitor(HMONITOR monitor,
                        Microsoft::WRL::ComPtr<IDXGIOutput>& output,
                        RECT& rect) const;

private:
  bool CreateSwapChain(HWND hwnd, UINT width, UINT height);
  void UpdateRenderTarget();
  void EnumerateMonitors();

  Microsoft::WRL::ComPtr<ID3D11Device> m_device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
  Microsoft::WRL::ComPtr<IDXGIFactory2> m_factory;
  Microsoft::WRL::ComPtr<IDXGISwapChain1> m_swapChain;
  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_rtv;
  HANDLE m_swapChainWaitHandle = nullptr;

  std::vector<MonitorInfo> m_monitors;
  bool m_allowTearing = false;
};
