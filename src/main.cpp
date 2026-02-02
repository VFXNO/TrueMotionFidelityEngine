#include "app.h"
#include <windows.h>
#include <winrt/base.h>
#include <timeapi.h>

namespace {

void ConfigureProcessPriority() {
  HANDLE process = GetCurrentProcess();
  SetPriorityClass(process, HIGH_PRIORITY_CLASS);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

  PROCESS_POWER_THROTTLING_STATE state = {};
  state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
#ifdef PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION
  state.ControlMask |= PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION;
#endif
  state.StateMask = 0;
  SetProcessInformation(process, ProcessPowerThrottling, &state, sizeof(state));

#ifdef THREAD_POWER_THROTTLING_EXECUTION_SPEED
  THREAD_POWER_THROTTLING_STATE threadState = {};
  threadState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
  threadState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
#ifdef THREAD_POWER_THROTTLING_BACKGROUND
  threadState.ControlMask |= THREAD_POWER_THROTTLING_BACKGROUND;
#endif
  threadState.StateMask = 0;
  SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &threadState, sizeof(threadState));
#endif
}

}  // namespace

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR, int) {
  // Make the application DPI aware to ensure GetClientRect/ClientToScreen match DXGI coordinates
  SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

  timeBeginPeriod(1);
  winrt::init_apartment(winrt::apartment_type::multi_threaded);
  ConfigureProcessPriority();

  App app;
  if (!app.Initialize(hInstance)) {
    timeEndPeriod(1);
    return -1;
  }

  int result = app.Run();
  timeEndPeriod(1);
  return result;
}
