#pragma once

#include "d3d11_device.h"
#include "dup_capture.h"
#include "game_capture.h"
#include "interpolator.h"
#include "ui.h"
#include "wgc_capture.h"
#include "window_list.h"

#include <windows.h>

#include <array>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>
#include <wrl/client.h>

class App {
public:
  App();

  bool Initialize(HINSTANCE hInstance);
  int Run();

private:
  bool CreateAppWindow(HINSTANCE hInstance);
  bool CreateUiWindow(HINSTANCE hInstance);
  void RenderUiWindow();
  void UpdateUiSwapChain(UINT width, UINT height);
  void Update();
  void UpdateCapture();
  void RestoreDxgiCropWindow();  // Restore output window after DXGI Crop mode
  
  // Capture methods
  bool ShouldUseWgcForWindowCapture() const;
  bool StartWindowCapture(HWND hwnd);
  bool StartMonitorCapture(HMONITOR monitor);
  void ResetCaptureState();
  void SelectMonitor(int index);
  void RefreshWindowList();
  void Render();
  void ResizeForCapture(int width, int height);
  void RenderUi();
  
  // Input handling methods
  bool IsPreviewPoint(POINT pt) const;
  bool MapPreviewPoint(POINT pt, POINT& outPt) const;
  bool MapOutputPoint(POINT pt, POINT& outPt) const;
  bool HandleOutputMouse(UINT message, WPARAM wParam, LPARAM lParam);
  bool HandleOutputKey(UINT message, WPARAM wParam, LPARAM lParam);
  bool HandlePreviewMouse(UINT message, WPARAM wParam, LPARAM lParam);
  bool HandlePreviewKey(UINT message, WPARAM wParam, LPARAM lParam);
  
  void UpdateOutputWindowMode();
  void UpdateOutputOverlayWindow();
  void FocusCaptureWindow();

  static LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
  static LRESULT CALLBACK UiWndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
  LRESULT HandleMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
  LRESULT HandleUiMessage(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);
  void ExportDiagnostics();

  HINSTANCE m_hInstance = nullptr;
  HWND m_hwnd = nullptr;
  HWND m_uiHwnd = nullptr;

  D3D11Device m_device;
  Microsoft::WRL::ComPtr<IDXGISwapChain1> m_uiSwapChain;
  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> m_uiRtv;
  int m_uiWidth = 480;
  int m_uiHeight = 720;
  bool m_uiVisible = true;
  WgcCapture m_capture;
  DupCapture m_dupCapture;
  GameCapture m_gameCapture;
  Interpolator m_interpolator;
  UiOverlay m_ui;

  std::vector<WindowInfo> m_windows;
  int m_selectedWindow = -1;
  std::string m_captureStatus;
  int m_captureMode = 0;

  int m_selectedMonitor = 0;
  bool m_monitorDirty = false;

  HWND m_captureWindow = nullptr;
  HMONITOR m_captureWindowMonitor = nullptr;
  RECT m_previewRect = {};
  bool m_previewHasImage = false;
  bool m_previewInputActive = false;

  bool m_interpolationEnabled = true;
  bool m_lowLatencyMode = true;
  bool m_neverDropFrames = false;
  bool m_temporalStabilization = true;
  float m_temporalHistoryWeight = 0.2f;
  float m_temporalConfInfluence = 0.6f;
  int m_temporalNeighborhoodSize = 2;
  bool m_textPreservationMode = false;
  float m_textPreservationStrength = 1.0f;
  float m_textPreservationEdgeThreshold = 0.03f;
  bool m_useMotionPrediction = true;
  bool m_limitOutputFps = true;
  bool m_useVsync = false;
  bool m_fullscreenWindowOutput = false;
  bool m_hideCaptureWindow = false;
  bool m_passthroughOverlay = false;
  bool m_overlayMode = true;  // Click-through overlay over game window
  bool m_outputInputEnabled = false;
  bool m_outputForwardMouseMove = true;
  bool m_outputRelativeMouseMode = false;
  bool m_outputConfineCursor = false;
  bool m_outputAutoFocus = true;
  bool m_windowCapturePreferWgc = true;
  bool m_forceDxgiCapture = false; // User override to prefer DXGI even when WGC is typically required
  bool m_windowCaptureUsingWgc = false;
  int m_outputMode = 0;
  int m_outputDisplayMode = 0;
  bool m_outputWindowVisible = true;
  int m_outputWindowMode = -1;
  bool m_outputTopmost = false;
  bool m_uiTopmost = false;
  bool m_captureWindowBehindOutput = false;
  HWND m_zOrderCaptureWindow = nullptr;
  int m_motionModel = 0; // 0=Adaptive, 1=Stable, 2=Balanced, 3=Coverage
  int m_outputMultiplier = 2;
  int m_debugView = 0;
  float m_debugMotionScale = 0.03f;
  float m_debugDiffScale = 2.0f;
  float m_confidencePower = 1.5f;
  float m_motionEdgeScale = 6.0f;
  int m_interpolationQuality = 1; // 0=Standard, 1=High
  float m_delayScale = 1.0f;
  float m_jitterSuppression = 0.2f;
  bool m_forceInterpolation = false;
  
  HANDLE m_waitTimer = nullptr;
  bool m_adaptiveDelay = true;
  int m_targetQueueDepth = 4;
  float m_outputDelayMs = 0.0f;
  float m_lastAlpha = 0.0f;
  bool m_lastInterpolated = false;
  float m_lastIntervalMs = 0.0f;
  float m_lastAvgIntervalMs = 0.0f;
  bool m_lastUnstable = false;
  float m_targetFps = 0.0f;
  int64_t m_lastPresentQpc = 0;
  int64_t m_nextOutputQpc = 0;
  double m_presentAvgInterval = 0.0;
  float m_presentFps = 0.0f;
  int m_captureFrameCount = 0;
  double m_captureFpsTime = 0.0;
  float m_captureFps = 0.0f;
  bool m_forceWgcCapture = false;
  bool m_unlockAppFps = false;     // SKips waitable object sync
  bool m_wgcLowLatencyMode = true; // Default true for lower latency
  bool m_wgcPreferNewest = true;   // Default true for lowest latency
  
  // DXGI Crop mode state
  bool m_dxgiCropModeActive = false;      // Track if DXGI Crop mode is active
  HMONITOR m_dxgiCropOriginalMonitor = nullptr;  // Original output monitor
  RECT m_dxgiCropOriginalRect = {};       // Original window position
  
  // Resizing controls
  int m_resizeWidth = 1920;
  int m_resizeHeight = 1080;
  bool m_resizeLockAspect = true;
  std::string m_resizeStatus;

  // Track overlay position to avoid expensive SetWindowPos calls
  RECT m_lastOverlayRect = {}; 
  
  double m_wgcFrameArrivalTime = 0.0;
  int m_wgcFrameArrivalCount = 0;
  float m_wgcArrivalRate = 0.0f;
  double m_frameIntervalSum = 0.0;
  int m_frameIntervalCount = 0;
  float m_minFrameInterval = 9999.0f;
  float m_maxFrameInterval = 0.0f;
  // ACCURACY: Use int64_t to prevent precision loss (double loses bits for large timestamps)
  std::vector<int64_t> m_frameTimestamps;
  int64_t m_lastSmoothedTime = 0;
  bool m_showUi = true;
  int m_outputStepIndex = 0;
  int m_lastMultiplier = 1;
  int m_pairPrevSlot = -1;
  int m_pairCurrSlot = -1;
  int64_t m_pairPrevTime100ns = 0;
  int64_t m_pairCurrTime100ns = 0;
  bool m_holdEndFrame = false;
  int m_maxQueueSize = 12;
  int m_frameWidth = 0;
  int m_frameHeight = 0;
  int m_outputWidth = 0;
  int m_outputHeight = 0;
  int m_lastOutputWidth = 0;
  int m_lastOutputHeight = 0;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_lastOutputSrv;

  // DXGI Crop mode - texture to hold cropped region
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_cropTexture;
  int m_cropWidth = 0;
  int m_cropHeight = 0;

  static constexpr int kFrameQueueSize = 12;
  std::array<Microsoft::WRL::ComPtr<ID3D11Texture2D>, kFrameQueueSize> m_frameTextures;
  std::array<Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>, kFrameQueueSize> m_frameSrvs;
  std::array<int64_t, kFrameQueueSize> m_frameTime100ns = {};
  std::deque<int> m_frameQueue;
  int m_queueWrite = 0;
  int m_outputMouseIgnore = 0;
  POINT m_lastMousePos = {0, 0};  // For relative mouse mode
  bool m_cursorConfined = false;   // Track cursor confinement state

  LARGE_INTEGER m_qpcFreq = {};
  int64_t m_prevFrameTime100ns = 0;
  int64_t m_currFrameTime100ns = 0;
  double m_avgFrameInterval = 0.0;
  double m_timeOffset100ns = 0.0;
  bool m_timeOffsetValid = false;
  
  // HOTKEYS
  int m_hotkeyToggleOverlay = VK_F9;
  int m_hotkeyToggleUi = VK_F8;
  bool m_toggleOverlayKeyState = false;
  bool m_toggleUiKeyState = false;

  bool m_showStartupAlert = true; // New flag for startup modal
};
