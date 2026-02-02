#pragma once

#include "capture_frame.h"

#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_5.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

class DupCapture {
public:
  bool Initialize(ID3D11Device* device);
  void Shutdown();

  bool StartCapture(HWND hwnd, Microsoft::WRL::ComPtr<IDXGIOutput> output, const RECT& outputRect);
  void StopCapture();

  bool AcquireNextFrame(CapturedFrame& frame);
  bool IsCapturing() const { return m_isCapturing; }

  // Configuration (Mirroring WGC features)
  void SetPollingMode(bool enabled) { m_pollingMode = enabled; }
  void SetPreferNewestFrame(bool enabled) { m_preferNewestFrame = enabled; }
  void SetSpinWaitMode(bool enabled) { m_spinWaitMode = enabled; }
  void SetSpinWaitMs(int ms) { m_spinWaitMs = ms; }
  bool IsSpinWaitMode() const { return m_spinWaitMode; }
  int GetSpinWaitMs() const { return m_spinWaitMs; }

private:
  void EnsureCaptureTexture(int width, int height, DXGI_FORMAT format);
  bool UpdateOutputRect();

  Microsoft::WRL::ComPtr<ID3D11Device> m_device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
  Microsoft::WRL::ComPtr<IDXGIOutput> m_output;
  Microsoft::WRL::ComPtr<IDXGIOutputDuplication> m_duplication;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_captureTexture;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_stagingTexture;
  bool m_useOutput5 = false;

  HWND m_hwnd = nullptr;
  RECT m_outputRect = {};
  int m_outputWidth = 0;
  int m_outputHeight = 0;
  bool m_isCapturing = false;

  // Settings
  bool m_pollingMode = false;       // If false, use blocking wait (up to 50ms)
  bool m_preferNewestFrame = true;  // Skip intermediate frames
  bool m_spinWaitMode = false;      // Busy loop for lowest latency
  int m_spinWaitMs = 10;            // 10ms wait for high-FPS capture (180fps = 5.5ms per frame)

  LARGE_INTEGER m_qpcFreq = {};
  
  // DPI cache
  UINT m_dpiX = 96;
  UINT m_dpiY = 96;
};
