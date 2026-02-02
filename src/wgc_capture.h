#pragma once

#ifndef WINRT_NO_COROUTINES
#define WINRT_NO_COROUTINES
#endif

#include "capture_frame.h"

#include <d3d11.h>
#include <windows.h>
#include <wrl/client.h>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/base.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>

struct WgcCaptureStatistics {
  uint32_t capturedFrames = 0;
  uint32_t droppedFrames = 0;
  int64_t lastFrameTime = 0;
  double avgFrameIntervalMs = 0.0;
  double lastFrameAgeMs = 0.0;  // Time since frame was captured
};

class WgcCapture {
public:
  bool Initialize(ID3D11Device* device);
  void Shutdown();

  bool StartCapture(HWND hwnd);
  bool StartCaptureMonitor(HMONITOR monitor);
  void StopCapture();
  bool RestartCapture();  // Restart current capture session

  bool AcquireNextFrame(CapturedFrame& frame);
  bool AcquireNextFramePolling(CapturedFrame& frame, bool poll);
  bool AcquireLatestFrame(CapturedFrame& frame);
  
  // Spin-wait polling - lower latency but higher CPU
  bool IsCapturing() const { return m_isCapturing; }
  bool HasError() const { return m_hasError; }

  // Low latency mode - uses smaller frame pool
  void SetLowLatencyMode(bool enabled) { m_lowLatencyMode = enabled; }
  bool IsLowLatencyMode() const { return m_lowLatencyMode; }
  
  // Prefer newest frame - always get latest, drop intermediate
  void SetPreferNewestFrame(bool enabled) { m_preferNewestFrame = enabled; }
  bool IsPreferNewestFrame() const { return m_preferNewestFrame; }
  
  // Frame statistics
  int GetDroppedFrameCount() const { return m_droppedFrames; }
  int GetCapturedFrameCount() const { return m_capturedFrames; }
  void ResetStatistics();
  WgcCaptureStatistics GetStatistics() const;
  
  // Get last frame's presentation time for better sync
  int64_t GetLastFrameTime() const { return m_lastFrameTime; }
  double GetLastFrameAgeMs() const { return m_lastFrameAgeMs; }
  int GetCurrentWidth() const { return m_width; }
  int GetCurrentHeight() const { return m_height; }

private:
  void OnFrameArrived(
      winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool const&,
      winrt::Windows::Foundation::IInspectable const&);
  void EnsureCaptureTexture(int width, int height);
  bool ProcessFrame(winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame const& captureFrame, CapturedFrame& frame);
  void UpdateFrameTiming(int64_t frameTime);

  // Dedicated capture thread logic
  void CaptureThreadLoop();
  
  Microsoft::WRL::ComPtr<ID3D11Device> m_device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_captureTexture;

  // Threading synchronization
  std::thread m_captureThread;
  std::mutex m_frameMutex; // Protects access to m_nextFrame
  std::condition_variable m_frameCv;
  std::atomic<bool> m_stopThread{ false };
  bool m_threadActive = false;
  
  // Hand-off slot for the dedicated thread
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame m_nextFrame{ nullptr };
  bool m_nextFrameReady = false;
  bool m_newFrameEvent = false; // Event flag for the capture thread

  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice m_winrtDevice{nullptr};
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool m_framePool{nullptr};
  winrt::Windows::Graphics::Capture::GraphicsCaptureSession m_session{nullptr};
  winrt::Windows::Graphics::Capture::GraphicsCaptureItem m_item{nullptr};

  winrt::event_token m_frameArrivedToken{};
  std::atomic<bool> m_hasNewFrame{false};
  std::atomic<int> m_pendingFrameCount{0};  // Track how many frames are waiting
   std::mutex m_mutex;

   bool m_lowLatencyMode = true;  // Default to low latency
   bool m_preferNewestFrame = true;  // Default to newest frame
   bool m_isCapturing = false;
  bool m_hasError = false;
  int m_width = 0;
  int m_height = 0;
  
  // For session recovery
  HWND m_captureHwnd = nullptr;
  HMONITOR m_captureMonitor = nullptr;
  bool m_isWindowCapture = false;
  
  // Statistics and timing
  int m_droppedFrames = 0;
  int m_capturedFrames = 0;
  int64_t m_lastFrameTime = 0;
  double m_lastFrameAgeMs = 0.0;
  
  // Frame interval tracking for statistics
  int64_t m_prevFrameQpc = 0;
  double m_frameIntervalSum = 0.0;
  int m_frameIntervalCount = 0;
  static constexpr int kMaxIntervalSamples = 60;
};
