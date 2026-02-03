#include "interpolator.h"
#include "shader_utils.h"

#include <windows.h>

#include <algorithm>
#include <fstream>

namespace {

struct MotionConstants {
  int radius = 3;
  int usePrediction = 0;
  float pad[2] = {};
};

struct SmoothConstants {
  // Relaxed edge scale (6.0 -> 3.0) to allow blending across minor edges (fixing blocks)
  float edgeScale = 3.0f;
  float confPower = 1.0f;
  float pad[2] = {};
};

struct TemporalConstants {
  float historyWeight = 0.2f;
  float confInfluence = 0.6f;
  int resetHistory = 0;
  int neighborhoodSize = 2; // Was pad
};

struct RefineConstants {
  int radius = 2;
  float motionScale = 2.0f;
  float pad[2] = {};
};

struct InterpConstants {
  float alpha = 0.5f;
  float diffScale = 2.0f;
  float confPower = 1.0f;
  int qualityMode = 0;
  int useDepth = 0;
  float depthScale = 1.0f;
  float depthThreshold = 0.02f;
  float motionSampleScale = 1.5f;
  int useHistory = 0;
  float historyWeight = 0.2f;
  float pad0 = 0.0f;
  float pad1 = 0.0f;
};

struct DebugConstants {
  int mode = 0;
  float motionScale = 0.03f;
  float diffScale = 2.0f;
  float pad = 0.0f;
};

UINT DispatchSize(int size) {
  return (size + 15) / 16;
}

}  // namespace

bool Interpolator::Initialize(ID3D11Device* device, ID3D11DeviceContext* context) {
  std::ofstream log("init_log.txt");
  log << "Interpolator::Initialize started\n";

  if (!device || !context) {
    log << "Error: device or context is null\n";
    log.close();
    return false;
  }

  m_device = device;
  m_context = context;

  std::wstring shaderPath = ShaderPath(L"DownsampleLuma.hlsl");
  log << "Shader path: " << WideToUtf8(shaderPath) << "\n";
  log.flush();

  log << "Calling LoadShaders...\n";
  log.flush();
  if (!LoadShaders()) {
    log << "LoadShaders failed\n";
    log.close();
    return false;
  }

  log << "LoadShaders succeeded\n";
  log << "Creating MotionConstants buffer...\n";
  log.flush();

  D3D11_BUFFER_DESC motionDesc = {};
  motionDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  motionDesc.ByteWidth = sizeof(MotionConstants);
  motionDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&motionDesc, nullptr, &m_motionConstants))) {
    log << "Failed to create MotionConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_BUFFER_DESC smoothDesc = {};
  smoothDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  smoothDesc.ByteWidth = sizeof(SmoothConstants);
  smoothDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&smoothDesc, nullptr, &m_smoothConstants))) {
    log << "Failed to create SmoothConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_BUFFER_DESC temporalDesc = {};
  temporalDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  temporalDesc.ByteWidth = sizeof(TemporalConstants);
  temporalDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&temporalDesc, nullptr, &m_temporalConstants))) {
    log << "Failed to create TemporalConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_BUFFER_DESC refineDesc = {};
  refineDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  refineDesc.ByteWidth = sizeof(RefineConstants);
  refineDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&refineDesc, nullptr, &m_refineConstants))) {
    log << "Failed to create RefineConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_BUFFER_DESC interpDesc = {};
  interpDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  interpDesc.ByteWidth = sizeof(InterpConstants);
  interpDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&interpDesc, nullptr, &m_interpConstants))) {
    log << "Failed to create InterpConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_BUFFER_DESC debugDesc = {};
  debugDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  debugDesc.ByteWidth = sizeof(DebugConstants);
  debugDesc.Usage = D3D11_USAGE_DEFAULT;
  if (FAILED(m_device->CreateBuffer(&debugDesc, nullptr, &m_debugConstants))) {
    log << "Failed to create DebugConstants buffer\n";
    log.close();
    return false;
  }

  D3D11_SAMPLER_DESC samplerDesc = {};
  samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.MinLOD = 0;
  samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
  if (FAILED(m_device->CreateSamplerState(&samplerDesc, &m_linearSampler))) {
    log << "Failed to create SamplerState\n";
    log.close();
    return false;
  }

  log << "Interpolator::Initialize succeeded\n";
  log.close();
  return true;
}

bool Interpolator::Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
  if (inputWidth <= 0 || inputHeight <= 0 || outputWidth <= 0 || outputHeight <= 0) {
    return false;
  }

  m_inputWidth = inputWidth;
  m_inputHeight = inputHeight;
  m_outputWidth = outputWidth;
  m_outputHeight = outputHeight;
  // PACING FIX: Reverting to Half Resolution to ensure render times stay well within 8.3ms budet
  m_lumaWidth = (inputWidth + 1) / 2;
  m_lumaHeight = (inputHeight + 1) / 2;
  // Small buffers are quarter resolution
  m_smallWidth = (m_lumaWidth + 1) / 2;
  m_smallHeight = (m_lumaHeight + 1) / 2;
  if (m_smallWidth < 1) {
    m_smallWidth = 1;
  }
  if (m_smallHeight < 1) {
    m_smallHeight = 1;
  }

  CreateResources();
  return true;
}

void Interpolator::Execute(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    float alpha,
    ID3D11ShaderResourceView* prevDepth,
    ID3D11ShaderResourceView* currDepth) {
  if (!prev || !curr || !m_outputUav) {
    return;
  }
  if (!m_downsampleCs || !m_downsampleLumaCs || !m_motionCs || !m_motionRefineCs ||
      !m_motionSmoothCs || !m_interpolateCs) {
    return;
  }
  if (m_outputWidth <= 0 || m_outputHeight <= 0 || m_lumaWidth <= 0 || m_lumaHeight <= 0) {
    return;
  }

  if (!ComputeMotion(prev, curr)) {
    return;
  }

  MotionConstants motionConstants = {};
  int radius = m_radius;
  if (radius < 1) {
    radius = 1;
  } else if (radius > 16) {
    radius = 16;
  }
  motionConstants.radius = radius;
  m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &motionConstants, 0, 0);

  InterpConstants interpConstants = {};
  float clampedAlpha = alpha;
  if (clampedAlpha < 0.0f) {
    clampedAlpha = 0.0f;
  } else if (clampedAlpha > 1.0f) {
    clampedAlpha = 1.0f;
  }
  interpConstants.alpha = clampedAlpha;

  // DiffScale unused by shader but kept structurally
  float diffScale = 2.0f;
  interpConstants.diffScale = diffScale;

  float confPower = m_confPower;
  if (confPower < 0.25f) {
    confPower = 0.25f;
  } else if (confPower > 4.0f) {
    confPower = 4.0f;
  }
  interpConstants.confPower = confPower;
  interpConstants.qualityMode = m_qualityMode;
  interpConstants.useDepth = (prevDepth && currDepth) ? 1 : 0;
  interpConstants.depthScale = 1.0f;
  interpConstants.depthThreshold = 0.02f;
  interpConstants.motionSampleScale = 1.5f;
  float historyWeight = m_temporalHistoryWeight;
  if (historyWeight < 0.0f) {
    historyWeight = 0.0f;
  } else if (historyWeight > 0.99f) {
    historyWeight = 0.99f;
  }
  interpConstants.useHistory = m_historyValid ? 1 : 0;
  interpConstants.historyWeight = historyWeight;
  m_context->UpdateSubresource(m_interpConstants.Get(), 0, nullptr, &interpConstants, 0, 0);

  ID3D11ShaderResourceView* motionSrv = m_motionSmoothSrv ? m_motionSmoothSrv.Get() : m_motionSrv.Get();
  ID3D11ShaderResourceView* confSrv = m_confidenceSmoothSrv ? m_confidenceSmoothSrv.Get() : m_confidenceSrv.Get();
  if (m_temporalEnabled && m_temporalValid && m_motionTemporalSrv[m_temporalIndex] &&
      m_confidenceTemporalSrv[m_temporalIndex]) {
    motionSrv = m_motionTemporalSrv[m_temporalIndex].Get();
    confSrv = m_confidenceTemporalSrv[m_temporalIndex].Get();
  }
  int historyReadIndex = m_historyIndex;
  ID3D11ShaderResourceView* historySrv =
      (m_historyValid && m_historyColorSrv[historyReadIndex]) ? m_historyColorSrv[historyReadIndex].Get() : nullptr;
  ID3D11ShaderResourceView* interpSrvs[] = {prev, curr, motionSrv, confSrv, prevDepth, currDepth, historySrv};
  ID3D11UnorderedAccessView* interpUavs[] = {m_outputUav.Get()};
  ID3D11Buffer* interpCbs[] = {m_interpConstants.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};
  m_context->CSSetShader(m_interpolateCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 7, interpSrvs);
  m_context->CSSetUnorderedAccessViews(0, 1, interpUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, interpCbs);
  m_context->CSSetSamplers(0, 1, samplers);
  m_context->Dispatch(DispatchSize(m_outputWidth), DispatchSize(m_outputHeight), 1);

  ID3D11ShaderResourceView* nullSrvs[7] = {};
  ID3D11UnorderedAccessView* nullUavs[2] = {};
  ID3D11SamplerState* nullSamplers[1] = {};
  m_context->CSSetShaderResources(0, 7, nullSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
  m_context->CSSetSamplers(0, 1, nullSamplers);
  m_context->CSSetShader(nullptr, nullptr, 0);

  if (m_outputTexture && m_historyColor[0]) {
    int writeIndex = (m_historyIndex + 1) % kHistorySize;
    if (m_historyColor[writeIndex]) {
      m_context->CopyResource(m_historyColor[writeIndex].Get(), m_outputTexture.Get());
      m_historyIndex = writeIndex;
      m_historyValid = true;
    }
  }
}


void Interpolator::Blit(ID3D11ShaderResourceView* src) {
  if (!src || !m_outputUav) {
    return;
  }
  if (!m_copyCs) {
    return;
  }
  if (m_outputWidth <= 0 || m_outputHeight <= 0) {
    return;
  }

  ID3D11ShaderResourceView* srvs[] = {src};
  ID3D11UnorderedAccessView* uavs[] = {m_outputUav.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_copyCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 1, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetSamplers(0, 1, samplers);
  m_context->Dispatch(DispatchSize(m_outputWidth), DispatchSize(m_outputHeight), 1);

  ID3D11ShaderResourceView* nullSrvs[1] = {};
  ID3D11UnorderedAccessView* nullUavs[1] = {};
  ID3D11SamplerState* nullSamplers[1] = {};
  m_context->CSSetShaderResources(0, 1, nullSrvs);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
  m_context->CSSetSamplers(0, 1, nullSamplers);
  m_context->CSSetShader(nullptr, nullptr, 0);

  if (m_outputTexture && m_historyColor[0]) {
    int writeIndex = (m_historyIndex + 1) % kHistorySize;
    if (m_historyColor[writeIndex]) {
      m_context->CopyResource(m_historyColor[writeIndex].Get(), m_outputTexture.Get());
      m_historyIndex = writeIndex;
      m_historyValid = true;
    }
  }
}

void Interpolator::Debug(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    DebugViewMode mode,
    float motionScale,
    float diffScale) {
  if (!prev || !curr || !m_outputUav) {
    return;
  }
  if (!m_debugCs || !m_debugConstants) {
    return;
  }
  if (m_outputWidth <= 0 || m_outputHeight <= 0 || m_lumaWidth <= 0 || m_lumaHeight <= 0) {
    return;
  }

  if (!ComputeMotion(prev, curr)) {
    return;
  }

  DebugConstants debugConstants = {};
  debugConstants.mode = static_cast<int>(mode);
  debugConstants.motionScale = motionScale;
  debugConstants.diffScale = diffScale;
  m_context->UpdateSubresource(m_debugConstants.Get(), 0, nullptr, &debugConstants, 0, 0);

  ID3D11ShaderResourceView* motionSrv = m_motionSmoothSrv ? m_motionSmoothSrv.Get() : m_motionSrv.Get();
  ID3D11ShaderResourceView* confSrv = m_confidenceSmoothSrv ? m_confidenceSmoothSrv.Get() : m_confidenceSrv.Get();
  if (m_temporalEnabled && m_temporalValid && m_motionTemporalSrv[m_temporalIndex] &&
      m_confidenceTemporalSrv[m_temporalIndex]) {
    motionSrv = m_motionTemporalSrv[m_temporalIndex].Get();
    confSrv = m_confidenceTemporalSrv[m_temporalIndex].Get();
  }
  ID3D11ShaderResourceView* srvs[] = {prev, curr, motionSrv, confSrv};
  ID3D11UnorderedAccessView* uavs[] = {m_outputUav.Get()};
  ID3D11Buffer* cbs[] = {m_debugConstants.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_debugCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 4, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, cbs);
  m_context->CSSetSamplers(0, 1, samplers);
  m_context->Dispatch(DispatchSize(m_outputWidth), DispatchSize(m_outputHeight), 1);

  ID3D11ShaderResourceView* nullSrvs[4] = {};
  ID3D11UnorderedAccessView* nullUavs[1] = {};
  ID3D11SamplerState* nullSamplers[1] = {};
  m_context->CSSetShaderResources(0, 4, nullSrvs);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
  m_context->CSSetSamplers(0, 1, nullSamplers);
  m_context->CSSetShader(nullptr, nullptr, 0);
}

bool Interpolator::LoadShaders() {
  Microsoft::WRL::ComPtr<ID3DBlob> blob;
  std::string error;

  if (!CompileShaderFromFile(ShaderPath(L"DownsampleLuma.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "Shader DownsampleLuma.hlsl failed: " << error << std::endl;
      errFile << "Path: " << WideToUtf8(ShaderPath(L"DownsampleLuma.hlsl")) << std::endl;
      errFile.close();
    }
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_downsampleCs))) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "CreateComputeShader DownsampleLuma failed" << std::endl;
      errFile.close();
    }
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"DownsampleLumaR.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "Shader DownsampleLumaR.hlsl failed: " << error << std::endl;
      errFile << "Path: " << WideToUtf8(ShaderPath(L"DownsampleLumaR.hlsl")) << std::endl;
      errFile.close();
    }
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_downsampleLumaCs))) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "CreateComputeShader DownsampleLumaR failed" << std::endl;
      errFile.close();
    }
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"MotionEst.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "Shader MotionEst.hlsl failed: " << error << std::endl;
      errFile << "Path: " << WideToUtf8(ShaderPath(L"MotionEst.hlsl")) << std::endl;
      errFile.close();
    }
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_motionCs))) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "CreateComputeShader MotionEst failed" << std::endl;
      errFile.close();
    }
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"MotionRefine.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    std::ofstream errFile("shader_error.txt");
    if (errFile.is_open()) {
      errFile << "Shader MotionRefine.hlsl failed: " << error << std::endl;
      errFile << "Path: " << WideToUtf8(ShaderPath(L"MotionRefine.hlsl")) << std::endl;
      errFile.close();
    }
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_motionRefineCs))) {
    OutputDebugStringA("CreateComputeShader MotionRefine failed\n");
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"MotionSmooth.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    OutputDebugStringA(("Shader MotionSmooth.hlsl failed: " + error + "\n").c_str());
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_motionSmoothCs))) {
    OutputDebugStringA("CreateComputeShader MotionSmooth failed\n");
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"MotionTemporal.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    OutputDebugStringA(("Shader MotionTemporal.hlsl failed: " + error + "\n").c_str());
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_motionTemporalCs))) {
    OutputDebugStringA("CreateComputeShader MotionTemporal failed\n");
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"Interpolate.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    OutputDebugStringA(("Shader Interpolate.hlsl failed: " + error + "\n").c_str());
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_interpolateCs))) {
    OutputDebugStringA("CreateComputeShader Interpolate failed\n");
    return false;
  }

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"CopyScale.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    OutputDebugStringA(("Shader CopyScale.hlsl failed: " + error + "\n").c_str());
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_copyCs))) {
    OutputDebugStringA("CreateComputeShader CopyScale failed\n");
    return false;
  }
  

  blob.Reset();
  if (!CompileShaderFromFile(ShaderPath(L"DebugView.hlsl"), "CSMain", "cs_5_0", blob, &error)) {
    OutputDebugStringA(("Shader DebugView.hlsl failed: " + error + "\n").c_str());
    return false;
  }
  if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &m_debugCs))) {
    OutputDebugStringA("CreateComputeShader DebugView failed\n");
    return false;
  }

  return true;
}

void Interpolator::CreateResources() {
  m_prevLuma.Reset();
  m_currLuma.Reset();
  m_prevLumaSmall.Reset();
  m_currLumaSmall.Reset();
  m_motion.Reset();
  m_confidence.Reset();
  m_motionCoarse.Reset();
  m_prevMotionCoarse.Reset();
  m_confidenceCoarse.Reset();
  m_motionSmooth.Reset();
  m_confidenceSmooth.Reset();
  for (int i = 0; i < 2; ++i) {
    m_motionTemporal[i].Reset();
    m_confidenceTemporal[i].Reset();
  }
  for (int i = 0; i < kHistorySize; ++i) {
    m_historyColor[i].Reset();
    m_historyColorSrv[i].Reset();
    m_historyColorUav[i].Reset();
  }
  m_outputTexture.Reset();
  m_prevLumaSrv.Reset();
  m_currLumaSrv.Reset();
  m_prevLumaSmallSrv.Reset();
  m_currLumaSmallSrv.Reset();
  m_motionSrv.Reset();
  m_confidenceSrv.Reset();
  m_motionCoarseSrv.Reset();
  m_prevMotionCoarseSrv.Reset();
  m_confidenceCoarseSrv.Reset();
  m_motionSmoothSrv.Reset();
  m_confidenceSmoothSrv.Reset();
  for (int i = 0; i < 2; ++i) {
    m_motionTemporalSrv[i].Reset();
    m_confidenceTemporalSrv[i].Reset();
  }
  m_outputSrv.Reset();
  m_prevLumaUav.Reset();
  m_currLumaUav.Reset();
  m_prevLumaSmallUav.Reset();
  m_currLumaSmallUav.Reset();
  m_motionUav.Reset();
  m_confidenceUav.Reset();
  m_motionCoarseUav.Reset();
  m_prevMotionCoarseUav.Reset();
  m_confidenceCoarseUav.Reset();
  m_motionSmoothUav.Reset();
  m_confidenceSmoothUav.Reset();
  for (int i = 0; i < 2; ++i) {
    m_motionTemporalUav[i].Reset();
    m_confidenceTemporalUav[i].Reset();
  }
  m_outputUav.Reset();

  auto createTexture = [&](int width, int height, DXGI_FORMAT format,
                           Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                           Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv,
                           Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = static_cast<UINT>(width);
    desc.Height = static_cast<UINT>(height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) {
      return;
    }
    if (FAILED(m_device->CreateShaderResourceView(tex.Get(), nullptr, &srv))) {
      tex.Reset();
      return;
    }
    if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) {
      tex.Reset();
      srv.Reset();
      return;
    }
  };

  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_prevLuma, m_prevLumaSrv, m_prevLumaUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_currLuma, m_currLumaSrv, m_currLumaUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_prevLumaSmall, m_prevLumaSmallSrv, m_prevLumaSmallUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_currLumaSmall, m_currLumaSmallSrv, m_currLumaSmallUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motion, m_motionSrv, m_motionUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidence, m_confidenceSrv, m_confidenceUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionCoarse, m_motionCoarseSrv, m_motionCoarseUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16_FLOAT, m_prevMotionCoarse, m_prevMotionCoarseSrv, m_prevMotionCoarseUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceCoarse, m_confidenceCoarseSrv, m_confidenceCoarseUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionSmooth, m_motionSmoothSrv, m_motionSmoothUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceSmooth, m_confidenceSmoothSrv, m_confidenceSmoothUav);
  for (int i = 0; i < 2; ++i) {
    createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTemporal[i],
                  m_motionTemporalSrv[i], m_motionTemporalUav[i]);
    createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTemporal[i],
                  m_confidenceTemporalSrv[i], m_confidenceTemporalUav[i]);
  }
  for (int i = 0; i < kHistorySize; ++i) {
    createTexture(m_outputWidth, m_outputHeight, DXGI_FORMAT_B8G8R8A8_UNORM, m_historyColor[i],
                  m_historyColorSrv[i], m_historyColorUav[i]);
  }
  createTexture(m_outputWidth, m_outputHeight, DXGI_FORMAT_B8G8R8A8_UNORM, m_outputTexture, m_outputSrv, m_outputUav);

  if (!m_outputTexture || !m_outputSrv || !m_outputUav ||
      !m_prevLumaUav || !m_currLumaUav ||
      !m_motionUav || !m_confidenceUav ||
      !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionSmoothUav || !m_confidenceSmoothUav) {
    return;
  }

  m_prevMotionCoarseValid = false;
  m_temporalValid = false;
  m_temporalIndex = 0;
  m_historyValid = false;
  m_historyIndex = 0;
}

bool Interpolator::ComputeMotion(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr) {
  if (!prev || !curr) {
    return false;
  }
  if (!m_downsampleCs || !m_downsampleLumaCs || !m_motionCs || !m_motionRefineCs ||
      !m_motionSmoothCs || !m_motionConstants || !m_refineConstants || !m_smoothConstants) {
    return false;
  }
  if (!m_prevLumaUav || !m_currLumaUav || !m_prevLumaSmallUav || !m_currLumaSmallUav ||
      !m_motionUav || !m_confidenceUav || !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionSmoothUav || !m_confidenceSmoothUav) {
    return false;
  }

  ID3D11ShaderResourceView* srv0[] = {prev};
  ID3D11UnorderedAccessView* uav0[] = {m_prevLumaUav.Get()};
  m_context->CSSetShader(m_downsampleCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 1, srv0);
  m_context->CSSetUnorderedAccessViews(0, 1, uav0, nullptr);
  m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

  ID3D11ShaderResourceView* nullSrv1[1] = {};
  ID3D11UnorderedAccessView* nullUav1[1] = {};
  m_context->CSSetShaderResources(0, 1, nullSrv1);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUav1, nullptr);

  ID3D11ShaderResourceView* srv1[] = {curr};
  ID3D11UnorderedAccessView* uav1[] = {m_currLumaUav.Get()};
  m_context->CSSetShaderResources(0, 1, srv1);
  m_context->CSSetUnorderedAccessViews(0, 1, uav1, nullptr);
  m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

  m_context->CSSetShaderResources(0, 1, nullSrv1);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUav1, nullptr);

  ID3D11ShaderResourceView* lumaSrv0[] = {m_prevLumaSrv.Get()};
  ID3D11UnorderedAccessView* lumaUav0[] = {m_prevLumaSmallUav.Get()};
  m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 1, lumaSrv0);
  m_context->CSSetUnorderedAccessViews(0, 1, lumaUav0, nullptr);
  m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);

  m_context->CSSetShaderResources(0, 1, nullSrv1);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUav1, nullptr);

  ID3D11ShaderResourceView* lumaSrv1[] = {m_currLumaSrv.Get()};
  ID3D11UnorderedAccessView* lumaUav1[] = {m_currLumaSmallUav.Get()};
  m_context->CSSetShaderResources(0, 1, lumaSrv1);
  m_context->CSSetUnorderedAccessViews(0, 1, lumaUav1, nullptr);
  m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);

  m_context->CSSetShaderResources(0, 1, nullSrv1);
  m_context->CSSetUnorderedAccessViews(0, 1, nullUav1, nullptr);

  // FAST MOTION FIX: Unlock the search radius for the coarse pass.
  // Since this runs on a 1/8th resolution buffer (m_small), we can afford a larger search.
  // Previously capped at 6 (radius/2). Now we use minimal 4, max 12?
  // Let's use m_radius directly. Max 8 means 8*4 = 32 pixel reach.
  // We will handle the performance cost in the shader using adaptive stepping.
  int coarseRadius = m_radius; 
  if (coarseRadius < 4) {
    coarseRadius = 4; // Ensure minimum search for stability
  }
  
  if (m_useMotionPrediction && m_prevMotionCoarseValid && m_copyCs) {
      // Backup current coarse motion to previous before overwriting
      // But wait! m_motionCoarse contains "current" result after this shader runs.
      // We must have copied it *before* or we just use double buffering?
      // Actually we just need to save the result of *this* pass for *next* frame.
      // So at the END of this function, we copy m_motionCoarse to m_prevMotionCoarse.
      // Here, m_prevMotionCoarse contains the result from LAST frame.
  }
  
  MotionConstants motionConstants = {};
  motionConstants.radius = coarseRadius;
  motionConstants.usePrediction = (m_useMotionPrediction && m_prevMotionCoarseValid) ? 1 : 0;
  m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &motionConstants, 0, 0);

  ID3D11ShaderResourceView* motionSrvs[] = {m_currLumaSmallSrv.Get(), m_prevLumaSmallSrv.Get(), m_prevMotionCoarseSrv.Get()};
  ID3D11UnorderedAccessView* motionUavs[] = {m_motionCoarseUav.Get(), m_confidenceCoarseUav.Get()};
  ID3D11Buffer* motionCbs[] = {m_motionConstants.Get()};
  m_context->CSSetShader(m_motionCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 3, motionSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, motionUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, motionCbs);
  m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);

  ID3D11ShaderResourceView* nullSrv2[3] = {};
  ID3D11UnorderedAccessView* nullUav2[2] = {};
  ID3D11Buffer* nullCb1[1] = {};
  m_context->CSSetShaderResources(0, 2, nullSrv2);
  m_context->CSSetUnorderedAccessViews(0, 2, nullUav2, nullptr);
  m_context->CSSetConstantBuffers(0, 1, nullCb1);

  RefineConstants refineConstants = {};
  int refineRadius = m_refineRadius;
  
  // QUALITY: Increase refine radius to capture motion of small objects 
  // that were missed by the coarse pass (because they vanished in downsampling).
  // Coarse pass found the "background" motion. We need to look further to find the "object" motion.
  // Old cap was 4. New cap 12.
  if (refineRadius < 2) {
    refineRadius = 2;
  } else if (refineRadius > 12) {
    refineRadius = 12;
  }
  refineConstants.radius = refineRadius;
  refineConstants.motionScale = static_cast<float>(m_lumaWidth) / static_cast<float>(m_smallWidth);
  m_context->UpdateSubresource(m_refineConstants.Get(), 0, nullptr, &refineConstants, 0, 0);

  // QUALITY: Use Full-Resolution Luma for refinement (safer than raw RGB)
  // Revert to original binding scheme: t0=curr(future), t1=prev(past) for Forward Motion
  ID3D11ShaderResourceView* refineSrvs[] = {
      m_currLumaSrv.Get(),
      m_prevLumaSrv.Get(),
      m_motionCoarseSrv.Get(),
      m_confidenceCoarseSrv.Get()};
  ID3D11UnorderedAccessView* refineUavs[] = {m_motionUav.Get(), m_confidenceUav.Get()};
  ID3D11Buffer* refineCbs[] = {m_refineConstants.Get()};
  ID3D11SamplerState* refineSamplers[] = {m_linearSampler.Get()};
  m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 4, refineSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, refineUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, refineCbs);
  m_context->CSSetSamplers(0, 1, refineSamplers);
  m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

  ID3D11ShaderResourceView* nullSrv4[4] = {};
  m_context->CSSetShaderResources(0, 4, nullSrv4);
  m_context->CSSetUnorderedAccessViews(0, 2, nullUav2, nullptr);
  m_context->CSSetConstantBuffers(0, 1, nullCb1);
  ID3D11SamplerState* nullSamplers[1] = {};
  m_context->CSSetSamplers(0, 1, nullSamplers);

  SmoothConstants smoothConstants = {};
  float edgeScale = m_smoothEdgeScale;
  if (edgeScale < 0.5f) {
    edgeScale = 0.5f;
  } else if (edgeScale > 20.0f) {
    edgeScale = 20.0f;
  }
  float confPower = m_smoothConfPower;
  if (confPower < 0.25f) {
    confPower = 0.25f;
  } else if (confPower > 4.0f) {
    confPower = 4.0f;
  }
  smoothConstants.edgeScale = edgeScale;
  smoothConstants.confPower = confPower;
  m_context->UpdateSubresource(m_smoothConstants.Get(), 0, nullptr, &smoothConstants, 0, 0);

  ID3D11ShaderResourceView* smoothSrvs[] = {m_motionSrv.Get(), m_confidenceSrv.Get(), m_prevLumaSrv.Get()};
  ID3D11UnorderedAccessView* smoothUavs[] = {m_motionSmoothUav.Get(), m_confidenceSmoothUav.Get()};
  ID3D11Buffer* smoothCbs[] = {m_smoothConstants.Get()};
  m_context->CSSetShader(m_motionSmoothCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 3, smoothSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, smoothUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, smoothCbs);
  m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

  ID3D11ShaderResourceView* nullSrvs[3] = {};
  m_context->CSSetShaderResources(0, 3, nullSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, nullUav2, nullptr);
  m_context->CSSetConstantBuffers(0, 1, nullCb1);
  m_context->CSSetShader(nullptr, nullptr, 0);

  if (m_temporalEnabled && m_motionTemporalCs && m_temporalConstants) {
    TemporalConstants temporalConstants = {};
    float historyWeight = m_temporalHistoryWeight;
    // Allow full range as requested by user (up to 0.99)
    if (historyWeight < 0.0f) {
      historyWeight = 0.0f;
    } else if (historyWeight > 0.99f) {
      historyWeight = 0.99f;
    }
    float confInfluence = m_temporalConfInfluence;
    if (confInfluence < 0.0f) {
      confInfluence = 0.0f;
    } else if (confInfluence > 1.0f) {
      confInfluence = 1.0f;
    }
    temporalConstants.historyWeight = historyWeight;
    temporalConstants.confInfluence = confInfluence;
    temporalConstants.neighborhoodSize = m_temporalNeighborhoodSize;
    
    bool resetHistory = m_temporalValid ? 0 : 1;
    if (!m_temporalValid) {
      resetHistory = 1;
    }
    temporalConstants.resetHistory = resetHistory;
    m_context->UpdateSubresource(m_temporalConstants.Get(), 0, nullptr, &temporalConstants, 0, 0);

    int readIndex = m_temporalIndex;
    int writeIndex = 1 - m_temporalIndex;

    ID3D11ShaderResourceView* temporalSrvs[] = {
        m_motionSmoothSrv.Get(),
        m_confidenceSmoothSrv.Get(),
        m_motionTemporalSrv[readIndex].Get(),
        m_confidenceTemporalSrv[readIndex].Get(),
        m_prevLumaSrv.Get(),
        m_currLumaSrv.Get()};
    ID3D11UnorderedAccessView* temporalUavs[] = {
        m_motionTemporalUav[writeIndex].Get(),
        m_confidenceTemporalUav[writeIndex].Get()};
    ID3D11Buffer* temporalCbs[] = {m_temporalConstants.Get()};
    m_context->CSSetShader(m_motionTemporalCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 6, temporalSrvs);
    m_context->CSSetUnorderedAccessViews(0, 2, temporalUavs, nullptr);
    m_context->CSSetConstantBuffers(0, 1, temporalCbs);
    m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

    ID3D11ShaderResourceView* nullSrvs4[6] = {};
    m_context->CSSetShaderResources(0, 6, nullSrvs4);
    m_context->CSSetUnorderedAccessViews(0, 2, nullUav2, nullptr);
    m_context->CSSetConstantBuffers(0, 1, nullCb1);
    m_context->CSSetShader(nullptr, nullptr, 0);

    m_temporalIndex = writeIndex;
    m_temporalValid = true;
  } else {
    m_temporalValid = false;
    m_temporalIndex = 0;
  }
  
  if (m_useMotionPrediction) {
      // Copy m_motionCoarse to m_prevMotionCoarse for next frame
      m_context->CopyResource(m_prevMotionCoarse.Get(), m_motionCoarse.Get());
      m_prevMotionCoarseValid = true;
  } else {
      m_prevMotionCoarseValid = false;
  }
  
  return true;
}

std::wstring Interpolator::ShaderPath(const wchar_t* filename) const {
  wchar_t path[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring exePath(path);
  size_t pos = exePath.find_last_of(L"\\/");
  if (pos == std::wstring::npos) {
    return filename;
  }
  std::wstring dir = exePath.substr(0, pos);
  return dir + L"\\shaders\\" + filename;
}
