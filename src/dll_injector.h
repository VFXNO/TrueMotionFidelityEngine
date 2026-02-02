#pragma once

#include <windows.h>
#include <string>

// Inject a DLL into a target process
class DllInjector {
public:
    enum class Result {
        Success,
        ProcessNotFound,
        OpenProcessFailed,
        AllocFailed,
        WriteFailed,
        CreateThreadFailed,
        LoadLibraryFailed
    };
    
    // Inject DLL into process by ID (tries safe method first, then CreateRemoteThread)
    static Result Inject(DWORD processId, const std::wstring& dllPath);
    
    // Inject DLL using SetWindowsHookEx (works better with anti-cheat)
    static Result InjectSafe(DWORD processId, const std::wstring& dllPath);
    
    // Inject DLL using SetWindowsHookEx on a specific thread
    static Result InjectByHook(DWORD threadId, const std::wstring& dllPath);
    
    // Inject DLL into process by handle (CreateRemoteThread method)
    static Result InjectByHandle(HANDLE process, const std::wstring& dllPath);
    
    // Get main thread ID of a process
    static DWORD GetMainThreadId(DWORD processId);
    
    // Get error message for result
    static const char* GetErrorString(Result result);
    
    // Check if process is 64-bit
    static bool Is64BitProcess(HANDLE process);
    static bool Is64BitProcess(DWORD processId);
};