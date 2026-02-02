#include "dll_injector.h"
#include <tlhelp32.h>
#include <fstream>

// Global hook handle for SetWindowsHookEx injection
static HHOOK g_hook = nullptr;
static std::wstring g_dllPath;
static bool g_injected = false;

// Hook procedure - when called in target process context, loads our DLL
static LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam) {
    return CallNextHookEx(g_hook, nCode, wParam, lParam);
}

DllInjector::Result DllInjector::Inject(DWORD processId, const std::wstring& dllPath) {
    // First try the safe method (SetWindowsHookEx) - works better with anti-cheat
    Result result = InjectSafe(processId, dllPath);
    if (result == Result::Success) {
        return result;
    }
    
    // Fall back to CreateRemoteThread method
    HANDLE process = OpenProcess(
        PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | 
        PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ,
        FALSE, processId);
    
    if (!process) {
        return Result::OpenProcessFailed;
    }
    
    result = InjectByHandle(process, dllPath);
    CloseHandle(process);
    return result;
}

DllInjector::Result DllInjector::InjectSafe(DWORD processId, const std::wstring& dllPath) {
    // Log the process ID we're trying to inject into
    FILE* log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "InjectSafe: target processId=%lu\n", processId);
        fclose(log);
    }
    
    // Find a thread in the target process
    DWORD threadId = GetMainThreadId(processId);
    if (threadId == 0) {
        log = fopen("injection_log.txt", "a");
        if (log) {
            fprintf(log, "InjectSafe: GetMainThreadId returned 0 for PID %lu\n", processId);
            fclose(log);
        }
        return Result::ProcessNotFound;
    }
    
    log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "InjectSafe: Found threadId=%lu for processId=%lu\n", threadId, processId);
        fclose(log);
    }
    
    return InjectByHook(threadId, dllPath);
}

DllInjector::Result DllInjector::InjectByHook(DWORD threadId, const std::wstring& dllPath) {
    // Write debug log
    FILE* log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "InjectByHook: threadId=%lu, dllPath=", threadId);
        fwprintf(log, L"%s\n", dllPath.c_str());
        fclose(log);
    }
    
    // Load our hook DLL into this process first to get the hook procedure
    HMODULE hookDll = LoadLibraryW(dllPath.c_str());
    if (!hookDll) {
        DWORD err = GetLastError();
        log = fopen("injection_log.txt", "a");
        if (log) {
            fprintf(log, "LoadLibraryW failed: error=%lu\n", err);
            fclose(log);
        }
        return Result::LoadLibraryFailed;
    }
    
    log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "DLL loaded into self: module=%p\n", hookDll);
        fclose(log);
    }
    
    // The hook DLL should export a dummy hook procedure
    // We use WH_GETMESSAGE because it's called frequently
    HOOKPROC hookProc = (HOOKPROC)GetProcAddress(hookDll, "DummyHookProc");
    if (!hookProc) {
        log = fopen("injection_log.txt", "a");
        if (log) {
            fprintf(log, "DummyHookProc not found, using stub\n");
            fclose(log);
        }
        // Use a stub hook procedure if no export
        hookProc = HookProc;
    } else {
        log = fopen("injection_log.txt", "a");
        if (log) {
            fprintf(log, "Found DummyHookProc at %p\n", hookProc);
            fclose(log);
        }
    }
    
    // Install the hook on the target thread
    // This will cause Windows to load our DLL into the target process
    // when the thread receives a message
    HHOOK hook = SetWindowsHookExW(WH_GETMESSAGE, hookProc, hookDll, threadId);
    
    if (!hook) {
        DWORD err = GetLastError();
        log = fopen("injection_log.txt", "a");
        if (log) {
            fprintf(log, "SetWindowsHookExW failed: error=%lu\n", err);
            fclose(log);
        }
        FreeLibrary(hookDll);
        return Result::CreateThreadFailed;  // Hook installation failed
    }
    
    log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "Hook installed successfully: hook=%p\n", hook);
        fclose(log);
    }
    
    // Try multiple ways to trigger the hook
    // 1. Post thread message
    PostThreadMessageW(threadId, WM_NULL, 0, 0);
    
    // 2. Find a window owned by this thread and send it a message
    HWND hwnd = nullptr;
    while ((hwnd = FindWindowExW(nullptr, hwnd, nullptr, nullptr)) != nullptr) {
        DWORD windowThread = GetWindowThreadProcessId(hwnd, nullptr);
        if (windowThread == threadId) {
            // Found a window on this thread, send it a message to trigger hook
            PostMessageW(hwnd, WM_NULL, 0, 0);
            log = fopen("injection_log.txt", "a");
            if (log) {
                fprintf(log, "Posted WM_NULL to window %p on thread %lu\n", hwnd, threadId);
                fclose(log);
            }
            break;
        }
    }
    
    // Give it more time to process - some games have slow message loops
    for (int i = 0; i < 10; i++) {
        Sleep(100);
        // Keep posting messages to ensure hook triggers
        if (hwnd) {
            PostMessageW(hwnd, WM_NULL, 0, 0);
        }
        PostThreadMessageW(threadId, WM_NULL, 0, 0);
    }
    
    // Remove the hook - the DLL should stay loaded due to DllMain
    UnhookWindowsHookEx(hook);
    FreeLibrary(hookDll);
    
    log = fopen("injection_log.txt", "a");
    if (log) {
        fprintf(log, "Hook injection completed (hook removed, dll freed locally)\n");
        fclose(log);
    }
    
    return Result::Success;
}

DWORD DllInjector::GetMainThreadId(DWORD processId) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return 0;
    }
    
    THREADENTRY32 te;
    te.dwSize = sizeof(te);
    
    DWORD threadId = 0;
    ULONGLONG earliestTime = MAXULONGLONG;
    
    if (Thread32First(snapshot, &te)) {
        do {
            if (te.th32OwnerProcessID == processId) {
                // Get the first thread we find (usually the main thread)
                // For better accuracy, we should get the thread with earliest creation time
                // but this requires additional API calls
                if (threadId == 0) {
                    threadId = te.th32ThreadID;
                }
            }
        } while (Thread32Next(snapshot, &te));
    }
    
    CloseHandle(snapshot);
    return threadId;
}

DllInjector::Result DllInjector::InjectByHandle(HANDLE process, const std::wstring& dllPath) {
    // Calculate size needed for DLL path (in bytes, including null terminator)
    size_t pathSize = (dllPath.length() + 1) * sizeof(wchar_t);
    
    // Allocate memory in target process for the DLL path
    void* remotePath = VirtualAllocEx(process, nullptr, pathSize, 
                                       MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!remotePath) {
        return Result::AllocFailed;
    }
    
    // Write DLL path to target process
    SIZE_T written = 0;
    if (!WriteProcessMemory(process, remotePath, dllPath.c_str(), pathSize, &written)) {
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        return Result::WriteFailed;
    }
    
    // Get LoadLibraryW address (same in all processes due to ASLR randomization per-boot)
    HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
    FARPROC loadLibrary = GetProcAddress(kernel32, "LoadLibraryW");
    
    if (!loadLibrary) {
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        return Result::LoadLibraryFailed;
    }
    
    // Create remote thread to load the DLL
    HANDLE thread = CreateRemoteThread(
        process, nullptr, 0,
        (LPTHREAD_START_ROUTINE)loadLibrary,
        remotePath, 0, nullptr);
    
    if (!thread) {
        VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
        return Result::CreateThreadFailed;
    }
    
    // Wait for thread to complete
    WaitForSingleObject(thread, 5000);
    
    // Check if DLL was loaded
    DWORD exitCode = 0;
    GetExitCodeThread(thread, &exitCode);
    
    CloseHandle(thread);
    VirtualFreeEx(process, remotePath, 0, MEM_RELEASE);
    
    if (exitCode == 0) {
        return Result::LoadLibraryFailed;
    }
    
    return Result::Success;
}

const char* DllInjector::GetErrorString(Result result) {
    switch (result) {
        case Result::Success:
            return "Success";
        case Result::ProcessNotFound:
            return "Process not found";
        case Result::OpenProcessFailed:
            return "Failed to open process (try running as administrator)";
        case Result::AllocFailed:
            return "Failed to allocate memory in target process";
        case Result::WriteFailed:
            return "Failed to write to target process memory";
        case Result::CreateThreadFailed:
            return "Failed to create remote thread";
        case Result::LoadLibraryFailed:
            return "Failed to load DLL in target process";
        default:
            return "Unknown error";
    }
}

bool DllInjector::Is64BitProcess(HANDLE process) {
    BOOL isWow64 = FALSE;
    
    // Check if running on 64-bit Windows
    #ifdef _WIN64
    return !IsWow64Process(process, &isWow64) || !isWow64;
    #else
    SYSTEM_INFO si;
    GetNativeSystemInfo(&si);
    if (si.wProcessorArchitecture != PROCESSOR_ARCHITECTURE_AMD64 &&
        si.wProcessorArchitecture != PROCESSOR_ARCHITECTURE_IA64) {
        return false; // 32-bit Windows
    }
    
    if (!IsWow64Process(process, &isWow64)) {
        return false;
    }
    return !isWow64;
    #endif
}

bool DllInjector::Is64BitProcess(DWORD processId) {
    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, processId);
    if (!process) {
        return false;
    }
    
    bool result = Is64BitProcess(process);
    CloseHandle(process);
    return result;
}
