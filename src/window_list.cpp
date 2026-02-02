#include "window_list.h"

#include <algorithm>
#include <dwmapi.h>
#include <vector>

namespace {

bool IsAltTabWindow(HWND hwnd) {
  if (!IsWindowVisible(hwnd)) {
    return false;
  }

  BOOL cloaked = FALSE;
  if (SUCCEEDED(DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked)))) {
    if (cloaked) {
      return false;
    }
  }

  LONG exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
  if (exStyle & WS_EX_TOOLWINDOW) {
    return false;
  }

  HWND owner = GetWindow(hwnd, GW_OWNER);
  if (owner != nullptr && !(exStyle & WS_EX_APPWINDOW)) {
    return false;
  }

  int length = GetWindowTextLengthW(hwnd);
  if (length > 0) {
    return true;
  }

  wchar_t className[256] = {};
  if (GetClassNameW(hwnd, className, 256) > 0) {
    return true;
  }

  return false;
}

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
  auto* list = reinterpret_cast<std::vector<WindowInfo>*>(lParam);
  if (!list) {
    return FALSE;
  }

  if (!IsAltTabWindow(hwnd)) {
    return TRUE;
  }

  int length = GetWindowTextLengthW(hwnd);
  std::wstring title;
  if (length > 0) {
    title.assign(length + 1, L'\0');
    GetWindowTextW(hwnd, &title[0], length + 1);
    if (!title.empty() && title.back() == L'\0') {
      title.pop_back();
    }
  } else {
    wchar_t className[256] = {};
    if (GetClassNameW(hwnd, className, 256) > 0) {
      title = std::wstring(L"[") + className + L"]";
    }
  }

  WindowInfo info;
  info.hwnd = hwnd;
  info.title = title;
  list->push_back(info);

  return TRUE;
}

}  // namespace

std::vector<WindowInfo> EnumerateTopLevelWindows(HWND excludeHwnd) {
  std::vector<WindowInfo> windows;

  EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));

  windows.erase(
      std::remove_if(
          windows.begin(),
          windows.end(),
          [excludeHwnd](const WindowInfo& info) { return info.hwnd == excludeHwnd; }),
      windows.end());

  return windows;
}
