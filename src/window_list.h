#pragma once

#include <windows.h>

#include <string>
#include <vector>

struct WindowInfo {
  HWND hwnd = nullptr;
  std::wstring title;
};

std::vector<WindowInfo> EnumerateTopLevelWindows(HWND excludeHwnd);
