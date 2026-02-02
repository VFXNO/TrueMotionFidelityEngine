#include "shader_utils.h"
#include <fstream>
#include <iostream>
#include <windows.h>
#include <vector>
#include <filesystem>
#include <sstream>

std::string WideToUtf8(const std::wstring& wide) {
  if (wide.empty()) {
    return std::string();
  }
  int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wide[0], (int)wide.size(), NULL, 0, NULL, NULL);
  std::string strTo(size_needed, 0);
  WideCharToMultiByte(CP_UTF8, 0, &wide[0], (int)wide.size(), &strTo[0], size_needed, NULL, NULL);
  return strTo;
}

// Get path for pre-compiled shader (.cso)
static std::wstring GetCompiledShaderPath(const std::wstring& hlslPath) {
  std::filesystem::path p(hlslPath);
  return (p.parent_path() / (p.stem().wstring() + L".cso")).wstring();
}

// Load pre-compiled shader blob (.cso file)
static bool LoadCompiledShader(const std::wstring& csoPath, Microsoft::WRL::ComPtr<ID3DBlob>& shaderBlob) {
  std::ifstream file(csoPath, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return false;
  
  size_t size = file.tellg();
  if (size == 0) return false;
  
  file.seekg(0, std::ios::beg);
  
  HRESULT hr = D3DCreateBlob(size, &shaderBlob);
  if (FAILED(hr)) return false;
  
  file.read(static_cast<char*>(shaderBlob->GetBufferPointer()), size);
  return file.good();
}

// Save compiled shader to .cso file for future loads
static void SaveCompiledShader(const std::wstring& csoPath, ID3DBlob* shaderBlob) {
  std::ofstream file(csoPath, std::ios::binary);
  if (file.is_open()) {
    file.write(static_cast<const char*>(shaderBlob->GetBufferPointer()), shaderBlob->GetBufferSize());
  }
}

bool CompileShaderFromFile(
    const std::wstring& path,
    const char* entryPoint,
    const char* target,
    Microsoft::WRL::ComPtr<ID3DBlob>& shaderBlob,
    std::string* errorMessage) {
  
  // FAST PATH: Load pre-compiled .cso file (from build or previous run)
  std::wstring csoPath = GetCompiledShaderPath(path);
  if (std::filesystem::exists(csoPath)) {
    // Check if .cso is newer than .hlsl (valid cache)
    auto hlslTime = std::filesystem::last_write_time(path);
    auto csoTime = std::filesystem::last_write_time(csoPath);
    
    if (csoTime >= hlslTime) {
      if (LoadCompiledShader(csoPath, shaderBlob)) {
        return true; // Instant load from pre-compiled shader
      }
    }
  }
  
  // FALLBACK: Compile shader at runtime
  UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(_DEBUG)
  flags |= D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
  flags |= D3DCOMPILE_OPTIMIZATION_LEVEL3;
#endif

  Microsoft::WRL::ComPtr<ID3DBlob> errorBlob;
  HRESULT hr = D3DCompileFromFile(
      path.c_str(),
      nullptr,
      D3D_COMPILE_STANDARD_FILE_INCLUDE,
      entryPoint,
      target,
      flags,
      0,
      &shaderBlob,
      &errorBlob);

  if (FAILED(hr)) {
    std::ofstream log("shader_compile_error.txt");
    if (log.is_open()) {
      log << "D3DCompileFromFile failed\n";
      log << "Path: " << WideToUtf8(path) << "\n";
      log << "Entry: " << entryPoint << "\n";
      log << "Target: " << target << "\n";
      log << "HRESULT: 0x" << std::hex << (unsigned long)hr << std::dec << "\n";
      if (errorBlob) {
        const char* msg = static_cast<const char*>(errorBlob->GetBufferPointer());
        log << "Error: " << (msg ? msg : "(null)") << "\n";
      } else {
        log << "Error blob is null\n";
      }
      log.close();
    }
    if (errorMessage) {
      if (errorBlob) {
        const char* msg = static_cast<const char*>(errorBlob->GetBufferPointer());
        errorMessage->assign(msg ? msg : "Shader compilation failed.");
      } else {
        errorMessage->assign("Shader compilation failed.");
      }
    }
    return false;
  }

  // Save to .cso for fast loading next time
  SaveCompiledShader(csoPath, shaderBlob.Get());

  return true;
}
