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
  int useBackward = 0;
  float backwardScale = 1.0f;
};

struct InterpConstants {
  float alpha = 0.5f;
  float diffScale = 2.0f;
  float confPower = 1.0f;
  int qualityMode = 0;
  int useHistory = 0;
  float historyWeight = 0.2f;
  float textProtect = 0.0f;
  float edgeThreshold = 0.0f;
  float motionSampleScale = 2.0f;
  float pad[3] = {};
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
  
  // Tiny buffers are 1/8th resolution (1/4 of Small)
  m_tinyWidth = (m_smallWidth + 1) / 2;
  m_tinyHeight = (m_smallHeight + 1) / 2;
  if (m_tinyWidth < 1) m_tinyWidth = 1;
  if (m_tinyHeight < 1) m_tinyHeight = 1;

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
  // Minimal mode always uses the cheapest warp sampling path (bilinear).
  interpConstants.qualityMode = m_useMinimalMotionPipeline ? 0 : m_qualityMode;
  float historyWeight = m_temporalHistoryWeight;
  if (historyWeight < 0.0f) {
    historyWeight = 0.0f;
  } else if (historyWeight > 0.99f) {
    historyWeight = 0.99f;
  }
  interpConstants.useHistory = (!m_useMinimalMotionPipeline && m_historyValid) ? 1 : 0;
  interpConstants.historyWeight = historyWeight;
  float textProtect = m_textProtectStrength;
  if (textProtect < 0.0f) {
    textProtect = 0.0f;
  } else if (textProtect > 1.0f) {
    textProtect = 1.0f;
  }
  float edgeThreshold = m_textEdgeThreshold;
  if (edgeThreshold < 0.0f) {
    edgeThreshold = 0.0f;
  } else if (edgeThreshold > 1.0f) {
    edgeThreshold = 1.0f;
  }
  interpConstants.textProtect = textProtect;
  interpConstants.edgeThreshold = edgeThreshold;
  if (m_useMinimalMotionPipeline && m_tinyWidth > 0) {
    // Minimal pipeline: motion comes from tiny pyramid level.
    interpConstants.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_tinyWidth);
  } else {
    // Full pipeline: motion comes from full luma level.
    interpConstants.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_lumaWidth);
  }
  m_context->UpdateSubresource(m_interpConstants.Get(), 0, nullptr, &interpConstants, 0, 0);

  ID3D11ShaderResourceView* motionSrv =
      (m_useMinimalMotionPipeline && m_motionTinySrv) ? m_motionTinySrv.Get() : m_motionSrv.Get();
  ID3D11ShaderResourceView* confSrv =
      (m_useMinimalMotionPipeline && m_confidenceTinySrv) ? m_confidenceTinySrv.Get() : m_confidenceSrv.Get();
  ID3D11ShaderResourceView* backwardMotionSrv =
      (m_useMinimalMotionPipeline && m_motionTinyBackwardSrv) ? m_motionTinyBackwardSrv.Get() : nullptr;
  ID3D11ShaderResourceView* backwardConfSrv =
      (m_useMinimalMotionPipeline && m_confidenceTinyBackwardSrv) ? m_confidenceTinyBackwardSrv.Get() : nullptr;
  int historyReadIndex = m_historyIndex;
  ID3D11ShaderResourceView* historySrv =
      (m_historyValid && m_historyColorSrv[historyReadIndex]) ? m_historyColorSrv[historyReadIndex].Get() : nullptr;
  ID3D11ShaderResourceView* interpSrvs[] = {prev, curr, motionSrv, confSrv, backwardMotionSrv, backwardConfSrv, historySrv};
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

  if (!m_useMinimalMotionPipeline && m_outputTexture && m_historyColor[0]) {
    int writeIndex = (m_historyIndex + 1) % kHistorySize;
    if (m_historyColor[writeIndex]) {
      m_context->CopyResource(m_historyColor[writeIndex].Get(), m_outputTexture.Get());
      m_historyIndex = writeIndex;
      m_historyValid = true;
    }
  } else if (m_useMinimalMotionPipeline) {
    m_historyValid = false;
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

  if (!m_useMinimalMotionPipeline && m_outputTexture && m_historyColor[0]) {
    int writeIndex = (m_historyIndex + 1) % kHistorySize;
    if (m_historyColor[writeIndex]) {
      m_context->CopyResource(m_historyColor[writeIndex].Get(), m_outputTexture.Get());
      m_historyIndex = writeIndex;
      m_historyValid = true;
    }
  } else if (m_useMinimalMotionPipeline) {
    m_historyValid = false;
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

  ID3D11ShaderResourceView* motionSrv = nullptr;
  ID3D11ShaderResourceView* confSrv = nullptr;
  if (m_useMinimalMotionPipeline) {
    motionSrv = m_motionTinySrv.Get();
    confSrv = m_confidenceTinySrv.Get();
  } else {
    motionSrv = m_motionSmoothSrv ? m_motionSmoothSrv.Get() : m_motionSrv.Get();
    confSrv = m_confidenceSmoothSrv ? m_confidenceSmoothSrv.Get() : m_confidenceSrv.Get();
    if (m_temporalEnabled && m_temporalValid && m_motionTemporalSrv[m_temporalIndex] &&
        m_confidenceTemporalSrv[m_temporalIndex]) {
      motionSrv = m_motionTemporalSrv[m_temporalIndex].Get();
      confSrv = m_confidenceTemporalSrv[m_temporalIndex].Get();
    }
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
  m_prevLuma.Reset(); m_prevLumaSrv.Reset(); m_prevLumaUav.Reset();
  m_currLuma.Reset(); m_currLumaSrv.Reset(); m_currLumaUav.Reset();
  
  m_prevLumaSmall.Reset(); m_prevLumaSmallSrv.Reset(); m_prevLumaSmallUav.Reset();
  m_currLumaSmall.Reset(); m_currLumaSmallSrv.Reset(); m_currLumaSmallUav.Reset();
  
  m_prevLumaTiny.Reset(); m_prevLumaTinySrv.Reset(); m_prevLumaTinyUav.Reset();
  m_currLumaTiny.Reset(); m_currLumaTinySrv.Reset(); m_currLumaTinyUav.Reset();
  
  m_motion.Reset(); m_motionSrv.Reset(); m_motionUav.Reset();
  m_confidence.Reset(); m_confidenceSrv.Reset(); m_confidenceUav.Reset();
  
  m_motionCoarse.Reset(); m_motionCoarseSrv.Reset(); m_motionCoarseUav.Reset();
  m_prevMotionCoarse.Reset(); m_prevMotionCoarseSrv.Reset(); m_prevMotionCoarseUav.Reset();
  
  m_motionTiny.Reset(); m_motionTinySrv.Reset(); m_motionTinyUav.Reset();
  m_motionTinyBackward.Reset(); m_motionTinyBackwardSrv.Reset(); m_motionTinyBackwardUav.Reset();
  m_confidenceTiny.Reset(); m_confidenceTinySrv.Reset(); m_confidenceTinyUav.Reset();
  m_confidenceTinyBackward.Reset(); m_confidenceTinyBackwardSrv.Reset(); m_confidenceTinyBackwardUav.Reset();
  m_confidenceCoarse.Reset(); m_confidenceCoarseSrv.Reset(); m_confidenceCoarseUav.Reset();
  
  m_motionSmooth.Reset(); m_motionSmoothSrv.Reset(); m_motionSmoothUav.Reset();
  m_confidenceSmooth.Reset(); m_confidenceSmoothSrv.Reset(); m_confidenceSmoothUav.Reset();
  
  for (int i = 0; i < 2; ++i) {
    m_motionTemporal[i].Reset(); m_motionTemporalSrv[i].Reset(); m_motionTemporalUav[i].Reset();
    m_confidenceTemporal[i].Reset(); m_confidenceTemporalSrv[i].Reset(); m_confidenceTemporalUav[i].Reset();
  }
  
  for (int i = 0; i < kHistorySize; ++i) {
    m_historyColor[i].Reset(); m_historyColorSrv[i].Reset(); m_historyColorUav[i].Reset();
  }
  
  m_outputTexture.Reset(); m_outputSrv.Reset(); m_outputUav.Reset();

  auto createTexture = [&](int width, int height, DXGI_FORMAT format,
                           Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                           Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv,
                           Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
    if (width <= 0 || height <= 0) return; 
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = static_cast<UINT>(width);
    desc.Height = static_cast<UINT>(height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) return;
    if (FAILED(m_device->CreateShaderResourceView(tex.Get(), nullptr, &srv))) { tex.Reset(); return; }
    if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { tex.Reset(); srv.Reset(); return; }
  };

  // Luma (Half Resolution)
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_prevLuma, m_prevLumaSrv, m_prevLumaUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_currLuma, m_currLumaSrv, m_currLumaUav);
  
  // Small (Quarter Resolution)
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_prevLumaSmall, m_prevLumaSmallSrv, m_prevLumaSmallUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_currLumaSmall, m_currLumaSmallSrv, m_currLumaSmallUav);
  
  // Tiny (1/8th Resolution relative to Luma, or 1/16th relative to Output)
  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_prevLumaTiny, m_prevLumaTinySrv, m_prevLumaTinyUav);
  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_currLumaTiny, m_currLumaTinySrv, m_currLumaTinyUav);

  // Motion Fields
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motion, m_motionSrv, m_motionUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidence, m_confidenceSrv, m_confidenceUav);
  
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionCoarse, m_motionCoarseSrv, m_motionCoarseUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16_FLOAT, m_prevMotionCoarse, m_prevMotionCoarseSrv, m_prevMotionCoarseUav);
  createTexture(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceCoarse, m_confidenceCoarseSrv, m_confidenceCoarseUav);

  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTiny, m_motionTinySrv, m_motionTinyUav);
  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTinyBackward, m_motionTinyBackwardSrv, m_motionTinyBackwardUav);
  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTiny, m_confidenceTinySrv, m_confidenceTinyUav);
  createTexture(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTinyBackward, m_confidenceTinyBackwardSrv, m_confidenceTinyBackwardUav);

  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionSmooth, m_motionSmoothSrv, m_motionSmoothUav);
  createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceSmooth, m_confidenceSmoothSrv, m_confidenceSmoothUav);
  
  for (int i = 0; i < 2; ++i) {
    createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTemporal[i], m_motionTemporalSrv[i], m_motionTemporalUav[i]);
    createTexture(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTemporal[i], m_confidenceTemporalSrv[i], m_confidenceTemporalUav[i]);
  }
  for (int i = 0; i < kHistorySize; ++i) {
    createTexture(m_outputWidth, m_outputHeight, DXGI_FORMAT_B8G8R8A8_UNORM, m_historyColor[i], m_historyColorSrv[i], m_historyColorUav[i]);
  }
  createTexture(m_outputWidth, m_outputHeight, DXGI_FORMAT_B8G8R8A8_UNORM, m_outputTexture, m_outputSrv, m_outputUav);

  if (!m_outputTexture || !m_outputSrv || !m_outputUav ||
      !m_prevLumaUav || !m_currLumaUav ||
      !m_motionUav || !m_confidenceUav ||
      !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionTinyUav || !m_motionTinyBackwardUav ||
      !m_confidenceTinyUav || !m_confidenceTinyBackwardUav ||
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
  // Check all required views including new Pyramid resources
  if (!m_prevLumaUav || !m_currLumaUav || 
      !m_prevLumaSmallUav || !m_currLumaSmallUav ||
      !m_prevLumaTinyUav || !m_currLumaTinyUav ||
      !m_motionUav || !m_confidenceUav ||
      !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionTinyUav || !m_motionTinyBackwardUav ||
      !m_confidenceTinyUav || !m_confidenceTinyBackwardUav) {
    return false;
  }
  if (!m_motionCs || !m_motionRefineCs || !m_motionSmoothCs || 
      !m_motionConstants || !m_refineConstants || !m_smoothConstants) {
    return false;
  }

  // Helper arrays for cleanup
  ID3D11ShaderResourceView* nullSrvs[6] = {};
  ID3D11UnorderedAccessView* nullUavs[2] = {};
  ID3D11Buffer* nullCbs[1] = {};
  ID3D11SamplerState* nullSamplers[1] = {};

  // Model-driven search radii (replaces manual Search/Refine Radius).
  int model = m_motionModel;
  if (model < 0) model = 0;
  else if (model > 3) model = 3;

  int tinyRadiusForward = 3;
  int tinyRadiusBackward = 2;
  int refineSmallRadius = 3;
  int refineFullRadius = 2;

  if (m_useMinimalMotionPipeline) {
    // Minimal mode keeps only tiny forward/backward search.
    // Slightly stronger forward search improves warp quality with modest extra cost.
    tinyRadiusForward = 2;
    tinyRadiusBackward = 1;
  } else if (model == 0) { // Adaptive
    bool hasPrediction = m_useMotionPrediction && m_prevMotionCoarseValid;
    if (!hasPrediction) {
      tinyRadiusForward = 2;
      tinyRadiusBackward = 2;
      refineSmallRadius = 2;
      refineFullRadius = 1;
    }
    if (!m_temporalEnabled) {
      tinyRadiusForward = std::min(4, tinyRadiusForward + 1);
      refineSmallRadius = std::min(4, refineSmallRadius + 1);
      refineFullRadius = std::min(3, refineFullRadius + 1);
    }
  } else if (model == 1) { // Stable
    tinyRadiusForward = 2;
    tinyRadiusBackward = 2;
    refineSmallRadius = 2;
    refineFullRadius = 1;
  } else if (model == 3) { // Coverage
    tinyRadiusForward = 4;
    tinyRadiusBackward = 3;
    refineSmallRadius = 4;
    refineFullRadius = 3;
  }

  // -----------------------------------------------------------------------
  // 1. DOWNSAMPLE CHAIN
  // -----------------------------------------------------------------------
  
  // A. Full -> Small (1/4)
  // ----------------------
  {
      // Convert Prev -> Luma Small
      ID3D11ShaderResourceView* srv[] = {prev};
      ID3D11UnorderedAccessView* uav[] = {m_prevLumaUav.Get()};
      m_context->CSSetShader(m_downsampleCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 1, srv);
      m_context->CSSetUnorderedAccessViews(0, 1, uav, nullptr);
      m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);
      
      // Clear
      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);

      // Convert Curr -> Luma Small
      srv[0] = curr;
      uav[0] = m_currLumaUav.Get();
      m_context->CSSetShaderResources(0, 1, srv);
      m_context->CSSetUnorderedAccessViews(0, 1, uav, nullptr);
      m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

      // Clear
      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
      
      // Prev Luma Full -> Prev Luma Small
      ID3D11ShaderResourceView* lumaSrv[] = {m_prevLumaSrv.Get()};
      ID3D11UnorderedAccessView* lumaUav[] = {m_prevLumaSmallUav.Get()};
      m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 1, lumaSrv);
      m_context->CSSetUnorderedAccessViews(0, 1, lumaUav, nullptr);
      m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);

      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);

      // Curr Luma Full -> Curr Luma Small
      lumaSrv[0] = m_currLumaSrv.Get();
      lumaUav[0] = m_currLumaSmallUav.Get();
      m_context->CSSetShaderResources(0, 1, lumaSrv);
      m_context->CSSetUnorderedAccessViews(0, 1, lumaUav, nullptr);
      m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);
      
      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
  }

  // B. Small -> Tiny (1/16)
  // -----------------------
  {
      // Prev Luma Small -> Prev Luma Tiny
      ID3D11ShaderResourceView* inputs[] = {m_prevLumaSmallSrv.Get()};
      ID3D11UnorderedAccessView* outputs[] = {m_prevLumaTinyUav.Get()};
      
      // Re-use downsampleLumaCs (logic is identical: averaging 2x2 block from input)
      m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0); 
      m_context->CSSetShaderResources(0, 1, inputs);
      m_context->CSSetUnorderedAccessViews(0, 1, outputs, nullptr);
      m_context->Dispatch(DispatchSize(m_tinyWidth), DispatchSize(m_tinyHeight), 1);
      
      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);

      // Curr Luma Small -> Curr Luma Tiny
      inputs[0] = m_currLumaSmallSrv.Get();
      outputs[0] = m_currLumaTinyUav.Get();
      m_context->CSSetShaderResources(0, 1, inputs);
      m_context->CSSetUnorderedAccessViews(0, 1, outputs, nullptr);
      m_context->Dispatch(DispatchSize(m_tinyWidth), DispatchSize(m_tinyHeight), 1);
      
      m_context->CSSetShaderResources(0, 1, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 1, nullUavs, nullptr);
  }

  // -----------------------------------------------------------------------
  // 2. MOTION ESTIMATION PYRAMID
  // -----------------------------------------------------------------------

  // PASS 1: TINY Coarse Search
  // --------------------------
  {
      MotionConstants tinyConstants = {};
      tinyConstants.radius = tinyRadiusForward;
      tinyConstants.usePrediction = (!m_useMinimalMotionPipeline && m_useMotionPrediction && m_prevMotionCoarseValid) ? 1 : 0;
      tinyConstants.pad[0] = 0.5f; // Prediction Scale: Coarse (1/4) -> Tiny (1/8) is 0.5x
      m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &tinyConstants, 0, 0);

      // Use Previous Coarse Motion as prediction (if valid)
      ID3D11ShaderResourceView* predSrv = m_prevMotionCoarseValid ? m_prevMotionCoarseSrv.Get() : nullptr;
      
      // CORRECT BINDING FOR FORWARD MOTION: {Curr, Prev, Pred}
      ID3D11ShaderResourceView* srvs[] = {m_currLumaTinySrv.Get(), m_prevLumaTinySrv.Get(), predSrv};
      ID3D11UnorderedAccessView* uavs[] = {m_motionTinyUav.Get(), m_confidenceTinyUav.Get()};
      ID3D11Buffer* cbs[] = {m_motionConstants.Get()};
      ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};
      
      m_context->CSSetShader(m_motionCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 3, srvs);
      m_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
      m_context->CSSetConstantBuffers(0, 1, cbs);
      m_context->CSSetSamplers(0, 1, samplers);
      m_context->Dispatch(DispatchSize(m_tinyWidth), DispatchSize(m_tinyHeight), 1);
      
      // Cleanup
      m_context->CSSetShaderResources(0, 3, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
      ID3D11SamplerState* nullSamplers[] = {nullptr};
      m_context->CSSetSamplers(0, 1, nullSamplers);
  }

  // PASS 1B: TINY Backward Search (Prev -> Curr) for consistency
  // -------------------------------------------------------------
  {
      MotionConstants tinyBackwardConstants = {};
      tinyBackwardConstants.radius = tinyRadiusBackward;
      tinyBackwardConstants.usePrediction = 0;
      tinyBackwardConstants.pad[0] = 1.0f;
      m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &tinyBackwardConstants, 0, 0);

      ID3D11ShaderResourceView* srvs[] = {m_prevLumaTinySrv.Get(), m_currLumaTinySrv.Get(), nullptr};
      ID3D11UnorderedAccessView* uavs[] = {m_motionTinyBackwardUav.Get(), m_confidenceTinyBackwardUav.Get()};
      ID3D11Buffer* cbs[] = {m_motionConstants.Get()};
      ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

      m_context->CSSetShader(m_motionCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 3, srvs);
      m_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
      m_context->CSSetConstantBuffers(0, 1, cbs);
      m_context->CSSetSamplers(0, 1, samplers);
      m_context->Dispatch(DispatchSize(m_tinyWidth), DispatchSize(m_tinyHeight), 1);

      m_context->CSSetShaderResources(0, 3, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
      ID3D11SamplerState* nullSamplers[] = {nullptr};
      m_context->CSSetSamplers(0, 1, nullSamplers);
  }

  if (m_useMinimalMotionPipeline) {
    // Minimal pipeline requested: keep only downsample pyramid + forward/backward motion + warp.
    m_temporalValid = false;
    m_temporalIndex = 0;
    m_prevMotionCoarseValid = false;
    return true;
  }

  // PASS 2: MEDIUM Refine
  // ---------------------
  {
      RefineConstants refineSmallConstants = {};
      refineSmallConstants.radius = refineSmallRadius;
      // Scale up the motion vector from Tiny->Small (usually x4)
      refineSmallConstants.motionScale = static_cast<float>(m_smallWidth) / static_cast<float>(m_tinyWidth);
      refineSmallConstants.useBackward = 1;
      refineSmallConstants.backwardScale = refineSmallConstants.motionScale;
      m_context->UpdateSubresource(m_refineConstants.Get(), 0, nullptr, &refineSmallConstants, 0, 0);

      // Inputs: {CurrSmall, PrevSmall, MotionTinyFwd, ConfTinyFwd, MotionTinyBwd, ConfTinyBwd}
      // Note: We MUST use {Curr, Prev} order for Refine shader to calculate Forward vectors (Prev->Curr).
      ID3D11ShaderResourceView* srvs[] = {
          m_currLumaSmallSrv.Get(), 
          m_prevLumaSmallSrv.Get(), 
          m_motionTinySrv.Get(),
          m_confidenceTinySrv.Get(),
          m_motionTinyBackwardSrv.Get(),
          m_confidenceTinyBackwardSrv.Get()
      };
      // Outputs: {MotionSmall, ConfSmall}
      ID3D11UnorderedAccessView* uavs[] = {m_motionCoarseUav.Get(), m_confidenceCoarseUav.Get()};
      ID3D11Buffer* cbs[] = {m_refineConstants.Get()};
      ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};
      
      m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 6, srvs);
      m_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
      m_context->CSSetConstantBuffers(0, 1, cbs);
      m_context->CSSetSamplers(0, 1, samplers);
      m_context->Dispatch(DispatchSize(m_smallWidth), DispatchSize(m_smallHeight), 1);
      
      // Cleanup
      m_context->CSSetShaderResources(0, 6, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
      m_context->CSSetSamplers(0, 1, nullSamplers);
  }

  // PASS 3: FULL Refine
  // -------------------
  {
      RefineConstants refineConstants = {};
      refineConstants.radius = refineFullRadius;
      // Scale up the motion vector from Small->Full (usually x4)
      refineConstants.motionScale = static_cast<float>(m_lumaWidth) / static_cast<float>(m_smallWidth);
      refineConstants.useBackward = 0;
      refineConstants.backwardScale = 1.0f;
      m_context->UpdateSubresource(m_refineConstants.Get(), 0, nullptr, &refineConstants, 0, 0);

      // Inputs: {CurrFull, PrevFull, MotionSmall, ConfSmall, DummyBwdMotion, DummyBwdConf}
      // Note: Use {Curr, Prev} for Forward Vectors.
      ID3D11ShaderResourceView* srvs[] = {
          m_currLumaSrv.Get(),
          m_prevLumaSrv.Get(),
          m_motionCoarseSrv.Get(),
          m_confidenceCoarseSrv.Get(),
          m_motionTinyBackwardSrv.Get(),
          m_confidenceTinyBackwardSrv.Get()
      };
      // Outputs: {MotionFull, ConfFull}
      ID3D11UnorderedAccessView* uavs[] = {m_motionUav.Get(), m_confidenceUav.Get()};
      ID3D11Buffer* cbs[] = {m_refineConstants.Get()};
      ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

      m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
      m_context->CSSetShaderResources(0, 6, srvs);
      m_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
      m_context->CSSetConstantBuffers(0, 1, cbs);
      m_context->CSSetSamplers(0, 1, samplers);
      m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

      // Cleanup
      m_context->CSSetShaderResources(0, 6, nullSrvs);
      m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
      m_context->CSSetSamplers(0, 1, nullSamplers);
  }

  // -----------------------------------------------------------------------
  // 3. POST PROCESSING (Smooth, Temporal)
  // -----------------------------------------------------------------------

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

  ID3D11ShaderResourceView* smoothSrvs[] = {m_motionSrv.Get(), m_confidenceSrv.Get(), m_currLumaSrv.Get()};
  ID3D11UnorderedAccessView* smoothUavs[] = {m_motionSmoothUav.Get(), m_confidenceSmoothUav.Get()};
  ID3D11Buffer* smoothCbs[] = {m_smoothConstants.Get()};
  m_context->CSSetShader(m_motionSmoothCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 3, smoothSrvs);
  m_context->CSSetUnorderedAccessViews(0, 2, smoothUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, smoothCbs);
  m_context->Dispatch(DispatchSize(m_lumaWidth), DispatchSize(m_lumaHeight), 1);

  ID3D11ShaderResourceView* nullSrvs3[3] = {};
  m_context->CSSetShaderResources(0, 3, nullSrvs3);
  m_context->CSSetUnorderedAccessViews(0, 2, nullUavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, nullCbs);
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
    ID3D11UnorderedAccessView* nullUavs4[2] = {};
    ID3D11Buffer* nullCbs4[1] = {};
    m_context->CSSetShaderResources(0, 6, nullSrvs4);
    m_context->CSSetUnorderedAccessViews(0, 2, nullUavs4, nullptr);
    m_context->CSSetConstantBuffers(0, 1, nullCbs4);
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
