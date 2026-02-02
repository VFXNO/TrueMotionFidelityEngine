#pragma once

#include <d3d11.h>

#include <cstdint>
#include <wrl/client.h>

struct CapturedFrame {
  Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
  int width = 0;
  int height = 0;
  int64_t qpcTime = 0;
  int64_t systemTime100ns = 0;
};
