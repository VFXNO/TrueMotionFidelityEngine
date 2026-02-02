#pragma once

#include <d3d11.h>
#include <windows.h>

class UiOverlay {
public:
  bool Initialize(HWND hwnd, ID3D11Device* device, ID3D11DeviceContext* context);
  void Shutdown();

  void BeginFrame();
  void Render();
};
