#pragma once

#include <d3dcompiler.h>
#include <wrl/client.h>

#include <string>

std::string WideToUtf8(const std::wstring& wide);

bool CompileShaderFromFile(
    const std::wstring& path,
    const char* entryPoint,
    const char* target,
    Microsoft::WRL::ComPtr<ID3DBlob>& shaderBlob,
    std::string* errorMessage);
