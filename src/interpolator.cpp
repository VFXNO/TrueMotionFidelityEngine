// ============================================================================
// Interpolator v2 - Rewritten motion estimation & frame interpolation pipeline
//
// Changes from v1:
//   - Cleaner pipeline with helper methods (Dispatch, ClearCS)
//   - ZNCC-based motion estimation (MotionEst.hlsl) instead of SAD
//   - Lucas-Kanade gradient-descent refinement (MotionRefine.hlsl) instead
//     of brute-force fractional search
//   - Joint bilateral spatial smoothing (MotionSmooth.hlsl)
//   - AABB-clamped temporal accumulation (MotionTemporal.hlsl) preventing
//     ghosting while maintaining jitter suppression
//   - Bidirectional pure motion-compensated warp (Interpolate.hlsl)
//   - Same public API as v1 for drop-in replacement
// ============================================================================

#include "interpolator.h"
#include "shader_utils.h"

#include <windows.h>

#include <algorithm>
#include <fstream>

namespace {

// -----------------------------------------------------------------------
// Constant buffer structures (must match HLSL cbuffer layouts)
// -----------------------------------------------------------------------
struct MotionConstants {
  int   radius         = 3;
  int   usePrediction  = 0;
  float predictionScale = 1.0f;
  float pad            = 0.0f;
};

struct RefineConstants {
  int   radius        = 2;
  float motionScale   = 2.0f;
  int   useBackward   = 0;
  float backwardScale = 1.0f;
  float attnLearnRate = 0.08f;
  float attnPriorMix  = 0.45f;
  float attnStability = 0.35f;
  float pad0          = 0.0f;
};

struct SmoothConstants {
  float edgeScale = 6.0f;
  float confPower = 1.0f;
  float pad[2]    = {};
};

struct InterpConstants {
  float alpha            = 0.5f;
  float diffScale        = 2.0f;
  float confPower        = 1.0f;
  int   qualityMode      = 0;
  int   _reserved0       = 0;
  float _reserved1       = 0.0f;
  float _reserved2       = 0.0f;
  float _reserved3       = 0.0f;
  float motionSampleScale = 2.0f;
  float pad[3]           = {};
};

struct DebugConstants {
  int   mode        = 0;
  float motionScale = 0.03f;
  float diffScale   = 2.0f;
  float pad         = 0.0f;
};

UINT DivUp(int size) {
  return static_cast<UINT>((size + 15) / 16);
}

}  // namespace

// -----------------------------------------------------------------------
// Helper: dispatch and clear compute state
// -----------------------------------------------------------------------
void Interpolator::Dispatch(int w, int h) {
  m_context->Dispatch(DivUp(w), DivUp(h), 1);
}

void Interpolator::ClearCS(int srvCount, int uavCount) {
  ID3D11ShaderResourceView*  nullSrvs[8] = {};
  ID3D11UnorderedAccessView* nullUavs[8] = {};
  ID3D11SamplerState*        nullSamp[1] = {};
  m_context->CSSetShaderResources(0, (srvCount > 8) ? 8 : srvCount, nullSrvs);
  m_context->CSSetUnorderedAccessViews(0, (uavCount > 8) ? 8 : uavCount, nullUavs, nullptr);
  m_context->CSSetSamplers(0, 1, nullSamp);
  m_context->CSSetShader(nullptr, nullptr, 0);
}

// -----------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------
bool Interpolator::Initialize(ID3D11Device* device, ID3D11DeviceContext* context) {
  std::ofstream log("init_log.txt", std::ios::app);
  log << "Interpolator::Initialize started\n";

  if (!device || !context) {
    log << "Error: device or context is null\n";
    return false;
  }

  m_device  = device;
  m_context = context;

  if (!LoadShaders()) {
    log << "LoadShaders failed\n";
    return false;
  }
  log << "LoadShaders succeeded\n";

  // Create constant buffers
  auto makeCB = [&](UINT size, Microsoft::WRL::ComPtr<ID3D11Buffer>& buf, const char* name) -> bool {
    D3D11_BUFFER_DESC desc = {};
    desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    desc.ByteWidth = size;
    desc.Usage     = D3D11_USAGE_DEFAULT;
    if (FAILED(m_device->CreateBuffer(&desc, nullptr, &buf))) {
      log << "Failed to create " << name << " buffer\n";
      return false;
    }
    return true;
  };

  if (!makeCB(sizeof(MotionConstants),   m_motionConstants,   "MotionConstants"))   return false;
  if (!makeCB(sizeof(RefineConstants),   m_refineConstants,   "RefineConstants"))   return false;
  if (!makeCB(sizeof(SmoothConstants),   m_smoothConstants,   "SmoothConstants"))   return false;
  if (!makeCB(sizeof(InterpConstants),   m_interpConstants,   "InterpConstants"))   return false;
  if (!makeCB(sizeof(DebugConstants),    m_debugConstants,    "DebugConstants"))    return false;

  D3D11_SAMPLER_DESC samplerDesc = {};
  samplerDesc.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
  samplerDesc.MinLOD   = 0;
  samplerDesc.MaxLOD   = D3D11_FLOAT32_MAX;
  if (FAILED(m_device->CreateSamplerState(&samplerDesc, &m_linearSampler))) {
    log << "Failed to create SamplerState\n";
    return false;
  }

  log << "Interpolator::Initialize succeeded\n";
  return true;
}

// -----------------------------------------------------------------------
// Resize
// -----------------------------------------------------------------------
bool Interpolator::Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
  if (inputWidth <= 0 || inputHeight <= 0 || outputWidth <= 0 || outputHeight <= 0)
    return false;

  m_inputWidth  = inputWidth;
  m_inputHeight = inputHeight;
  m_outputWidth  = outputWidth;
  m_outputHeight = outputHeight;

  // Half resolution luma
  m_lumaWidth  = (inputWidth  + 1) / 2;
  m_lumaHeight = (inputHeight + 1) / 2;

  // Quarter resolution
  m_smallWidth  = std::max(1, (m_lumaWidth  + 1) / 2);
  m_smallHeight = std::max(1, (m_lumaHeight + 1) / 2);

  // Eighth resolution
  m_tinyWidth  = std::max(1, (m_smallWidth  + 1) / 2);
  m_tinyHeight = std::max(1, (m_smallHeight + 1) / 2);

  CreateResources();
  return true;
}

// -----------------------------------------------------------------------
// Execute: run the full pipeline and produce an interpolated frame
// -----------------------------------------------------------------------
void Interpolator::Execute(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    float alpha,
    ID3D11ShaderResourceView* /*prevDepth*/,
    ID3D11ShaderResourceView* /*currDepth*/) {
  if (!prev || !curr || !m_outputUav) return;
  if (!m_downsampleCs || !m_downsampleLumaCs || !m_motionCs ||
      !m_motionRefineCs || !m_motionSmoothCs || !m_interpolateCs)
    return;
  if (m_outputWidth <= 0 || m_outputHeight <= 0 || m_lumaWidth <= 0 || m_lumaHeight <= 0)
    return;

  // --- Compute motion field ---
  if (!ComputeMotion(prev, curr)) return;

  // --- Build interpolation constants ---
  InterpConstants ic = {};
  ic.alpha     = std::clamp(alpha, 0.0f, 1.0f);
  ic.diffScale = 2.0f;
  ic.confPower = std::clamp(m_confPower, 0.25f, 4.0f);
  ic.qualityMode = m_useMinimalMotionPipeline ? 0 : m_qualityMode;

  // History / text-preservation removed — pure warp only
  ic._reserved0 = 0;
  ic._reserved1 = 0.0f;
  ic._reserved2 = 0.0f;
  ic._reserved3 = 0.0f;

  if (m_useMinimalMotionPipeline && m_tinyWidth > 0) {
    ic.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_tinyWidth);
  } else {
    ic.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_lumaWidth);
  }
  m_context->UpdateSubresource(m_interpConstants.Get(), 0, nullptr, &ic, 0, 0);

  // Select motion/confidence SRVs based on pipeline mode
  ID3D11ShaderResourceView* motionSrv = nullptr;
  ID3D11ShaderResourceView* confSrv   = nullptr;
  ID3D11ShaderResourceView* bwdMotionSrv = nullptr;
  ID3D11ShaderResourceView* bwdConfSrv   = nullptr;

  if (m_useMinimalMotionPipeline) {
    motionSrv    = m_motionTinySrv.Get();
    confSrv      = m_confidenceTinySrv.Get();
    bwdMotionSrv = m_motionTinyBackwardSrv.Get();
    bwdConfSrv   = m_confidenceTinyBackwardSrv.Get();
  } else {
    // Use the best available: smooth > raw
    if (m_motionSmoothSrv) {
      motionSrv = m_motionSmoothSrv.Get();
      confSrv   = m_confidenceSmoothSrv.Get();
    } else {
      motionSrv = m_motionSrv.Get();
      confSrv   = m_confidenceSrv.Get();
    }
    // No backward MV in full pipeline (consistency built into refine)
    bwdMotionSrv = nullptr;
    bwdConfSrv   = nullptr;
  }

  // --- Dispatch interpolation ---
  ID3D11ShaderResourceView* srvs[] = {
      prev, curr, motionSrv, confSrv, bwdMotionSrv, bwdConfSrv,
      m_prevLumaSrv.Get(), m_currLumaSrv.Get(),
      m_prevFeature2Srv.Get(), m_currFeature2Srv.Get(),
      m_prevFeature3Srv.Get(), m_currFeature3Srv.Get()
  };
  ID3D11UnorderedAccessView* uavs[] = {m_outputUav.Get()};
  ID3D11Buffer* cbs[] = {m_interpConstants.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_interpolateCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 12, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, cbs);
  m_context->CSSetSamplers(0, 1, samplers);
  Dispatch(m_outputWidth, m_outputHeight);
  ClearCS(12, 1);
}

// -----------------------------------------------------------------------
// InterpolateOnly: re-warp with new alpha, reusing cached motion field
// Skips the entire motion estimation pipeline (downsample, ME, refine,
// smooth, temporal).  Use when the source pair has NOT changed.
// -----------------------------------------------------------------------
void Interpolator::InterpolateOnly(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    float alpha) {
  if (!prev || !curr || !m_outputUav || !m_interpolateCs) return;
  if (m_outputWidth <= 0 || m_outputHeight <= 0) return;

  // --- Build interpolation constants (same layout as Execute) ---
  InterpConstants ic = {};
  ic.alpha     = std::clamp(alpha, 0.0f, 1.0f);
  ic.diffScale = 2.0f;
  ic.confPower = std::clamp(m_confPower, 0.25f, 4.0f);
  ic.qualityMode = m_useMinimalMotionPipeline ? 0 : m_qualityMode;
  ic._reserved0 = 0;
  ic._reserved1 = 0.0f;
  ic._reserved2 = 0.0f;
  ic._reserved3 = 0.0f;
  if (m_useMinimalMotionPipeline && m_tinyWidth > 0) {
    ic.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_tinyWidth);
  } else {
    ic.motionSampleScale = static_cast<float>(m_inputWidth) / static_cast<float>(m_lumaWidth);
  }
  m_context->UpdateSubresource(m_interpConstants.Get(), 0, nullptr, &ic, 0, 0);

  // Select cached motion/confidence SRVs (same logic as Execute)
  ID3D11ShaderResourceView* motionSrv = nullptr;
  ID3D11ShaderResourceView* confSrv   = nullptr;
  ID3D11ShaderResourceView* bwdMotionSrv = nullptr;
  ID3D11ShaderResourceView* bwdConfSrv   = nullptr;
  if (m_useMinimalMotionPipeline) {
    motionSrv    = m_motionTinySrv.Get();
    confSrv      = m_confidenceTinySrv.Get();
    bwdMotionSrv = m_motionTinyBackwardSrv.Get();
    bwdConfSrv   = m_confidenceTinyBackwardSrv.Get();
  } else {
    if (m_motionSmoothSrv) {
      motionSrv = m_motionSmoothSrv.Get();
      confSrv   = m_confidenceSmoothSrv.Get();
    } else {
      motionSrv = m_motionSrv.Get();
      confSrv   = m_confidenceSrv.Get();
    }
    bwdMotionSrv = nullptr;
    bwdConfSrv   = nullptr;
  }

  ID3D11ShaderResourceView* srvs[] = {
      prev, curr, motionSrv, confSrv, bwdMotionSrv, bwdConfSrv,
      m_prevLumaSrv.Get(), m_currLumaSrv.Get(),
      m_prevFeature2Srv.Get(), m_currFeature2Srv.Get(),
      m_prevFeature3Srv.Get(), m_currFeature3Srv.Get()
  };
  ID3D11UnorderedAccessView* uavs[] = {m_outputUav.Get()};
  ID3D11Buffer* cbs[] = {m_interpConstants.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_interpolateCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 12, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetConstantBuffers(0, 1, cbs);
  m_context->CSSetSamplers(0, 1, samplers);
  Dispatch(m_outputWidth, m_outputHeight);
  ClearCS(12, 1);
}

// -----------------------------------------------------------------------
// Blit: simple copy/scale pass-through
// -----------------------------------------------------------------------
void Interpolator::Blit(ID3D11ShaderResourceView* src) {
  if (!src || !m_outputUav || !m_copyCs) return;
  if (m_outputWidth <= 0 || m_outputHeight <= 0) return;

  ID3D11ShaderResourceView* srvs[]     = {src};
  ID3D11UnorderedAccessView* uavs[]    = {m_outputUav.Get()};
  ID3D11SamplerState* samplers[]       = {m_linearSampler.Get()};

  m_context->CSSetShader(m_copyCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 1, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetSamplers(0, 1, samplers);
  Dispatch(m_outputWidth, m_outputHeight);
  ClearCS(1, 1);
}

// -----------------------------------------------------------------------
// Debug: visualize motion/confidence
// -----------------------------------------------------------------------
void Interpolator::Debug(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    DebugViewMode mode,
    float motionScale,
    float diffScale) {
  if (!prev || !curr || !m_outputUav || !m_debugCs || !m_debugConstants) return;
  if (m_outputWidth <= 0 || m_outputHeight <= 0 || m_lumaWidth <= 0 || m_lumaHeight <= 0) return;

  if (!ComputeMotion(prev, curr)) return;

  DebugConstants dc = {};
  dc.mode        = static_cast<int>(mode);
  dc.motionScale = motionScale;
  dc.diffScale   = diffScale;
  m_context->UpdateSubresource(m_debugConstants.Get(), 0, nullptr, &dc, 0, 0);

  // Pick best motion SRVs
  ID3D11ShaderResourceView* motionSrv = nullptr;
  ID3D11ShaderResourceView* confSrv   = nullptr;
  if (m_useMinimalMotionPipeline) {
    motionSrv = m_motionTinySrv.Get();
    confSrv   = m_confidenceTinySrv.Get();
  } else {
    motionSrv = m_motionSmoothSrv ? m_motionSmoothSrv.Get() : m_motionSrv.Get();
    confSrv   = m_confidenceSmoothSrv ? m_confidenceSmoothSrv.Get() : m_confidenceSrv.Get();
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
  Dispatch(m_outputWidth, m_outputHeight);
  ClearCS(4, 1);
}

// -----------------------------------------------------------------------
// LoadShaders
// -----------------------------------------------------------------------
bool Interpolator::LoadShaders() {
  Microsoft::WRL::ComPtr<ID3DBlob> blob;
  std::string error;

  auto loadCS = [&](const wchar_t* file, Microsoft::WRL::ComPtr<ID3D11ComputeShader>& cs) -> bool {
    blob.Reset();
    if (!CompileShaderFromFile(ShaderPath(file), "CSMain", "cs_5_0", blob, &error)) {
      std::ofstream errFile("shader_error.txt");
      errFile << "Shader " << WideToUtf8(std::wstring(file)) << " failed: " << error << std::endl;
      errFile << "Path: " << WideToUtf8(ShaderPath(file)) << std::endl;
      return false;
    }
    if (FAILED(m_device->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &cs))) {
      std::ofstream errFile("shader_error.txt");
      errFile << "CreateComputeShader " << WideToUtf8(std::wstring(file)) << " failed" << std::endl;
      return false;
    }
    return true;
  };

  if (!loadCS(L"DownsampleLuma.hlsl",  m_downsampleCs))     return false;
  if (!loadCS(L"DownsampleLumaR.hlsl", m_downsampleLumaCs)) return false;
  if (!loadCS(L"MotionEst.hlsl",       m_motionCs))         return false;
  if (!loadCS(L"MotionRefine.hlsl",    m_motionRefineCs))   return false;
  if (!loadCS(L"MotionSmooth.hlsl",    m_motionSmoothCs))   return false;

  if (!loadCS(L"Interpolate.hlsl",     m_interpolateCs))    return false;
  if (!loadCS(L"CopyScale.hlsl",       m_copyCs))           return false;
  if (!loadCS(L"DebugView.hlsl",       m_debugCs))          return false;

  return true;
}

// -----------------------------------------------------------------------
// CreateResources
// -----------------------------------------------------------------------
void Interpolator::CreateResources() {
  // Reset all textures
  m_prevLuma.Reset(); m_prevLumaSrv.Reset(); m_prevLumaUav.Reset();
  m_currLuma.Reset(); m_currLumaSrv.Reset(); m_currLumaUav.Reset();
  m_prevLumaSmall.Reset(); m_prevLumaSmallSrv.Reset(); m_prevLumaSmallUav.Reset();
  m_currLumaSmall.Reset(); m_currLumaSmallSrv.Reset(); m_currLumaSmallUav.Reset();
  m_prevLumaTiny.Reset(); m_prevLumaTinySrv.Reset(); m_prevLumaTinyUav.Reset();
  m_currLumaTiny.Reset(); m_currLumaTinySrv.Reset(); m_currLumaTinyUav.Reset();

  m_prevFeature2.Reset(); m_prevFeature2Srv.Reset(); m_prevFeature2Uav.Reset();
  m_currFeature2.Reset(); m_currFeature2Srv.Reset(); m_currFeature2Uav.Reset();
  m_prevFeature2Small.Reset(); m_prevFeature2SmallSrv.Reset(); m_prevFeature2SmallUav.Reset();
  m_currFeature2Small.Reset(); m_currFeature2SmallSrv.Reset(); m_currFeature2SmallUav.Reset();
  m_prevFeature2Tiny.Reset(); m_prevFeature2TinySrv.Reset(); m_prevFeature2TinyUav.Reset();
  m_currFeature2Tiny.Reset(); m_currFeature2TinySrv.Reset(); m_currFeature2TinyUav.Reset();

  m_prevFeature3.Reset(); m_prevFeature3Srv.Reset(); m_prevFeature3Uav.Reset();
  m_currFeature3.Reset(); m_currFeature3Srv.Reset(); m_currFeature3Uav.Reset();
  m_prevFeature3Small.Reset(); m_prevFeature3SmallSrv.Reset(); m_prevFeature3SmallUav.Reset();
  m_currFeature3Small.Reset(); m_currFeature3SmallSrv.Reset(); m_currFeature3SmallUav.Reset();
  m_prevFeature3Tiny.Reset(); m_prevFeature3TinySrv.Reset(); m_prevFeature3TinyUav.Reset();
  m_currFeature3Tiny.Reset(); m_currFeature3TinySrv.Reset(); m_currFeature3TinyUav.Reset();

  m_motion.Reset(); m_motionSrv.Reset(); m_motionUav.Reset();
  m_confidence.Reset(); m_confidenceSrv.Reset(); m_confidenceUav.Reset();
  m_motionCoarse.Reset(); m_motionCoarseSrv.Reset(); m_motionCoarseUav.Reset();
  m_motionTiny.Reset(); m_motionTinySrv.Reset(); m_motionTinyUav.Reset();
  m_motionTinyBackward.Reset(); m_motionTinyBackwardSrv.Reset(); m_motionTinyBackwardUav.Reset();
  m_confidenceTiny.Reset(); m_confidenceTinySrv.Reset(); m_confidenceTinyUav.Reset();
  m_confidenceTinyBackward.Reset(); m_confidenceTinyBackwardSrv.Reset(); m_confidenceTinyBackwardUav.Reset();
  m_confidenceCoarse.Reset(); m_confidenceCoarseSrv.Reset(); m_confidenceCoarseUav.Reset();
  m_motionSmooth.Reset(); m_motionSmoothSrv.Reset(); m_motionSmoothUav.Reset();
  m_confidenceSmooth.Reset(); m_confidenceSmoothSrv.Reset(); m_confidenceSmoothUav.Reset();
  m_attnSmall1.Reset(); m_attnSmall1Uav.Reset();
  m_attnSmall2.Reset(); m_attnSmall2Uav.Reset();
  m_attnSmall3.Reset(); m_attnSmall3Uav.Reset();
  m_attnFull1.Reset(); m_attnFull1Uav.Reset();
  m_attnFull2.Reset(); m_attnFull2Uav.Reset();
  m_attnFull3.Reset(); m_attnFull3Uav.Reset();

  m_outputTexture.Reset(); m_outputSrv.Reset(); m_outputUav.Reset();

  // Helper lambda to create texture + SRV + UAV
  auto createTex = [&](int w, int h, DXGI_FORMAT fmt,
                       Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                       Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv,
                       Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
    if (w <= 0 || h <= 0) return;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width      = static_cast<UINT>(w);
    desc.Height     = static_cast<UINT>(h);
    desc.MipLevels  = 1;
    desc.ArraySize  = 1;
    desc.Format     = fmt;
    desc.SampleDesc.Count = 1;
    desc.Usage      = D3D11_USAGE_DEFAULT;
    desc.BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) return;
    if (FAILED(m_device->CreateShaderResourceView(tex.Get(), nullptr, &srv)))  { tex.Reset(); return; }
    if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { tex.Reset(); srv.Reset(); return; }
  };

  auto createUavTex = [&](int w, int h, DXGI_FORMAT fmt,
                          Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                          Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
    if (w <= 0 || h <= 0) return;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = static_cast<UINT>(w);
    desc.Height = static_cast<UINT>(h);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = fmt;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) return;
    if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { tex.Reset(); return; }
  };

  // Luma pyramid (now storing 4-channel CNN features)
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevLuma, m_prevLumaSrv, m_prevLumaUav);
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currLuma, m_currLumaSrv, m_currLumaUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevLumaSmall, m_prevLumaSmallSrv, m_prevLumaSmallUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currLumaSmall, m_currLumaSmallSrv, m_currLumaSmallUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevLumaTiny, m_prevLumaTinySrv, m_prevLumaTinyUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currLumaTiny, m_currLumaTinySrv, m_currLumaTinyUav);

  // Feature2 pyramid (Channels 5-8)
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature2, m_prevFeature2Srv, m_prevFeature2Uav);
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature2, m_currFeature2Srv, m_currFeature2Uav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature2Small, m_prevFeature2SmallSrv, m_prevFeature2SmallUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature2Small, m_currFeature2SmallSrv, m_currFeature2SmallUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature2Tiny, m_prevFeature2TinySrv, m_prevFeature2TinyUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature2Tiny, m_currFeature2TinySrv, m_currFeature2TinyUav);

  // Feature3 pyramid (Channels 9-12)
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature3, m_prevFeature3Srv, m_prevFeature3Uav);
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature3, m_currFeature3Srv, m_currFeature3Uav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature3Small, m_prevFeature3SmallSrv, m_prevFeature3SmallUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature3Small, m_currFeature3SmallSrv, m_currFeature3SmallUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_prevFeature3Tiny, m_prevFeature3TinySrv, m_prevFeature3TinyUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_currFeature3Tiny, m_currFeature3TinySrv, m_currFeature3TinyUav);

  // Motion fields
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motion, m_motionSrv, m_motionUav);
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidence, m_confidenceSrv, m_confidenceUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionCoarse, m_motionCoarseSrv, m_motionCoarseUav);
  createTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceCoarse, m_confidenceCoarseSrv, m_confidenceCoarseUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTiny, m_motionTinySrv, m_motionTinyUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionTinyBackward, m_motionTinyBackwardSrv, m_motionTinyBackwardUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTiny, m_confidenceTinySrv, m_confidenceTinyUav);
  createTex(m_tinyWidth, m_tinyHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceTinyBackward, m_confidenceTinyBackwardSrv, m_confidenceTinyBackwardUav);

  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16_FLOAT, m_motionSmooth, m_motionSmoothSrv, m_motionSmoothUav);
  createTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16_FLOAT, m_confidenceSmooth, m_confidenceSmoothSrv, m_confidenceSmoothUav);

  // Attention priors are writable state buffers (UAV-only), one float4 per feature set
  createUavTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnSmall1, m_attnSmall1Uav);
  createUavTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnSmall2, m_attnSmall2Uav);
  createUavTex(m_smallWidth, m_smallHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnSmall3, m_attnSmall3Uav);
  createUavTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnFull1, m_attnFull1Uav);
  createUavTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnFull2, m_attnFull2Uav);
  createUavTex(m_lumaWidth, m_lumaHeight, DXGI_FORMAT_R16G16B16A16_FLOAT, m_attnFull3, m_attnFull3Uav);

  if (m_context) {
    const float baseW1[4] = {0.15f, 0.10f, 0.10f, 0.20f};
    const float baseW2[4] = {0.10f, 0.10f, 0.15f, 0.10f};
    const float baseW3[4] = {0.10f, 0.10f, 0.10f, 0.10f};
    if (m_attnSmall1Uav) m_context->ClearUnorderedAccessViewFloat(m_attnSmall1Uav.Get(), baseW1);
    if (m_attnSmall2Uav) m_context->ClearUnorderedAccessViewFloat(m_attnSmall2Uav.Get(), baseW2);
    if (m_attnSmall3Uav) m_context->ClearUnorderedAccessViewFloat(m_attnSmall3Uav.Get(), baseW3);
    if (m_attnFull1Uav) m_context->ClearUnorderedAccessViewFloat(m_attnFull1Uav.Get(), baseW1);
    if (m_attnFull2Uav) m_context->ClearUnorderedAccessViewFloat(m_attnFull2Uav.Get(), baseW2);
    if (m_attnFull3Uav) m_context->ClearUnorderedAccessViewFloat(m_attnFull3Uav.Get(), baseW3);
  }

  createTex(m_outputWidth, m_outputHeight, DXGI_FORMAT_B8G8R8A8_UNORM, m_outputTexture, m_outputSrv, m_outputUav);

  // Validate critical resources
  if (!m_outputTexture || !m_outputSrv || !m_outputUav ||
      !m_prevLumaUav || !m_currLumaUav ||
      !m_motionUav || !m_confidenceUav ||
      !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionTinyUav || !m_motionTinyBackwardUav ||
      !m_confidenceTinyUav || !m_confidenceTinyBackwardUav ||
      !m_motionSmoothUav || !m_confidenceSmoothUav ||
      !m_attnSmall1Uav || !m_attnSmall2Uav || !m_attnSmall3Uav ||
      !m_attnFull1Uav || !m_attnFull2Uav || !m_attnFull3Uav) {
    return;
  }
}

// -----------------------------------------------------------------------
// ComputeMotion: the core motion estimation pyramid
// -----------------------------------------------------------------------
bool Interpolator::ComputeMotion(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr) {
  if (!prev || !curr) return false;
  if (!m_prevLumaUav || !m_currLumaUav ||
      !m_prevLumaSmallUav || !m_currLumaSmallUav ||
      !m_prevLumaTinyUav || !m_currLumaTinyUav ||
      !m_motionUav || !m_confidenceUav ||
      !m_motionCoarseUav || !m_confidenceCoarseUav ||
      !m_motionTinyUav || !m_motionTinyBackwardUav ||
      !m_confidenceTinyUav || !m_confidenceTinyBackwardUav ||
      !m_attnSmall1Uav || !m_attnSmall2Uav || !m_attnSmall3Uav ||
      !m_attnFull1Uav || !m_attnFull2Uav || !m_attnFull3Uav)
    return false;
  if (!m_motionCs || !m_motionRefineCs || !m_motionSmoothCs ||
      !m_motionConstants || !m_refineConstants || !m_smoothConstants)
    return false;

  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  // Model-driven search radii
  int model = std::clamp(m_motionModel, 0, 3);
  int tinyRadiusFwd = 12, tinyRadiusBwd = 12;
  int refineSmallR = 8, refineFullR = 6;
  float attnLearnRate = 0.08f;
  float attnPriorMix = 0.45f;
  float attnStability = 0.35f;

  if (m_useMinimalMotionPipeline) {
    tinyRadiusFwd = 4; tinyRadiusBwd = 4;
    attnLearnRate = 0.03f;
    attnPriorMix = 0.30f;
    attnStability = 0.65f;
  } else if (model == 0) { // Adaptive
    tinyRadiusFwd = 16; tinyRadiusBwd = 16; refineSmallR = 12; refineFullR = 8;
    attnLearnRate = 0.09f;
    attnPriorMix = 0.55f;
    attnStability = 0.28f;
  } else if (model == 1) { // Stable
    tinyRadiusFwd = 8; tinyRadiusBwd = 8; refineSmallR = 6; refineFullR = 4;
    attnLearnRate = 0.04f;
    attnPriorMix = 0.65f;
    attnStability = 0.70f;
  } else if (model == 3) { // Coverage
    tinyRadiusFwd = 24; tinyRadiusBwd = 24; refineSmallR = 16; refineFullR = 12;
    attnLearnRate = 0.11f;
    attnPriorMix = 0.40f;
    attnStability = 0.22f;
  }

  // =======================================================================
  // STAGE 1: DOWNSAMPLE PYRAMID
  // =======================================================================

  // Full -> Half luma (prev)
  {
    ID3D11ShaderResourceView* s[] = {prev};
    ID3D11UnorderedAccessView* u[] = {m_prevLumaUav.Get(), m_prevFeature2Uav.Get(), m_prevFeature3Uav.Get()};
    m_context->CSSetShader(m_downsampleCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 1, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_lumaWidth, m_lumaHeight);
    ClearCS(1, 3);
  }
  // Full -> Half luma (curr)
  {
    ID3D11ShaderResourceView* s[] = {curr};
    ID3D11UnorderedAccessView* u[] = {m_currLumaUav.Get(), m_currFeature2Uav.Get(), m_currFeature3Uav.Get()};
    m_context->CSSetShader(m_downsampleCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 1, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_lumaWidth, m_lumaHeight);
    ClearCS(1, 3);
  }
  // Half -> Quarter (prev)
  {
    ID3D11ShaderResourceView* s[] = {m_prevLumaSrv.Get(), m_prevFeature2Srv.Get(), m_prevFeature3Srv.Get()};
    ID3D11UnorderedAccessView* u[] = {m_prevLumaSmallUav.Get(), m_prevFeature2SmallUav.Get(), m_prevFeature3SmallUav.Get()};
    m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_smallWidth, m_smallHeight);
    ClearCS(3, 3);
  }
  // Half -> Quarter (curr)
  {
    ID3D11ShaderResourceView* s[] = {m_currLumaSrv.Get(), m_currFeature2Srv.Get(), m_currFeature3Srv.Get()};
    ID3D11UnorderedAccessView* u[] = {m_currLumaSmallUav.Get(), m_currFeature2SmallUav.Get(), m_currFeature3SmallUav.Get()};
    m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_smallWidth, m_smallHeight);
    ClearCS(3, 3);
  }
  // Quarter -> Eighth (prev)
  {
    ID3D11ShaderResourceView* s[] = {m_prevLumaSmallSrv.Get(), m_prevFeature2SmallSrv.Get(), m_prevFeature3SmallSrv.Get()};
    ID3D11UnorderedAccessView* u[] = {m_prevLumaTinyUav.Get(), m_prevFeature2TinyUav.Get(), m_prevFeature3TinyUav.Get()};
    m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_tinyWidth, m_tinyHeight);
    ClearCS(3, 3);
  }
  // Quarter -> Eighth (curr)
  {
    ID3D11ShaderResourceView* s[] = {m_currLumaSmallSrv.Get(), m_currFeature2SmallSrv.Get(), m_currFeature3SmallSrv.Get()};
    ID3D11UnorderedAccessView* u[] = {m_currLumaTinyUav.Get(), m_currFeature2TinyUav.Get(), m_currFeature3TinyUav.Get()};
    m_context->CSSetShader(m_downsampleLumaCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 3, u, nullptr);
    Dispatch(m_tinyWidth, m_tinyHeight);
    ClearCS(3, 3);
  }

  // =======================================================================
  // STAGE 2: MOTION ESTIMATION (Tiny level - forward)
  // =======================================================================
  {
    MotionConstants mc = {};
    mc.radius = tinyRadiusFwd;
    mc.usePrediction = 0;
    mc.predictionScale = 0.5f; // coarse -> tiny scale
    m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &mc, 0, 0);

    ID3D11ShaderResourceView* predSrv = nullptr;
    ID3D11ShaderResourceView* s[] = {m_currLumaTinySrv.Get(), m_prevLumaTinySrv.Get(), predSrv};
    ID3D11UnorderedAccessView* u[] = {m_motionTinyUav.Get(), m_confidenceTinyUav.Get()};
    ID3D11Buffer* cbs[] = {m_motionConstants.Get()};

    m_context->CSSetShader(m_motionCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 2, u, nullptr);
    m_context->CSSetConstantBuffers(0, 1, cbs);
    m_context->CSSetSamplers(0, 1, samplers);
    Dispatch(m_tinyWidth, m_tinyHeight);
    ClearCS(3, 2);
  }

  // =======================================================================
  // STAGE 2B: MOTION ESTIMATION (Tiny level - backward for consistency)
  // =======================================================================
  {
    MotionConstants mc = {};
    mc.radius = tinyRadiusBwd;
    mc.usePrediction = 0;
    mc.predictionScale = 1.0f;
    m_context->UpdateSubresource(m_motionConstants.Get(), 0, nullptr, &mc, 0, 0);

    ID3D11ShaderResourceView* s[] = {m_prevLumaTinySrv.Get(), m_currLumaTinySrv.Get(), nullptr};
    ID3D11UnorderedAccessView* u[] = {m_motionTinyBackwardUav.Get(), m_confidenceTinyBackwardUav.Get()};
    ID3D11Buffer* cbs[] = {m_motionConstants.Get()};

    m_context->CSSetShader(m_motionCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 2, u, nullptr);
    m_context->CSSetConstantBuffers(0, 1, cbs);
    m_context->CSSetSamplers(0, 1, samplers);
    Dispatch(m_tinyWidth, m_tinyHeight);
    ClearCS(3, 2);
  }

  // --- Minimal pipeline stops here ---
  if (m_useMinimalMotionPipeline) {
    return true;
  }

  // =======================================================================
  // STAGE 3: REFINEMENT (Quarter level)
  // =======================================================================
  {
    RefineConstants rc = {};
    rc.radius      = refineSmallR;
    rc.motionScale = static_cast<float>(m_smallWidth) / static_cast<float>(m_tinyWidth);
    rc.useBackward = 1;
    rc.backwardScale = rc.motionScale;
    rc.attnLearnRate = attnLearnRate;
    rc.attnPriorMix = attnPriorMix;
    rc.attnStability = attnStability;
    m_context->UpdateSubresource(m_refineConstants.Get(), 0, nullptr, &rc, 0, 0);

    ID3D11ShaderResourceView* s[] = {
        m_currLumaSmallSrv.Get(), m_prevLumaSmallSrv.Get(),
        m_currFeature2SmallSrv.Get(), m_prevFeature2SmallSrv.Get(),
        m_currFeature3SmallSrv.Get(), m_prevFeature3SmallSrv.Get(),
        m_motionTinySrv.Get(), m_confidenceTinySrv.Get(),
        m_motionTinyBackwardSrv.Get(), m_confidenceTinyBackwardSrv.Get()
    };
    ID3D11UnorderedAccessView* u[] = {
        m_motionCoarseUav.Get(),
        m_confidenceCoarseUav.Get(),
        m_attnSmall1Uav.Get(),
        m_attnSmall2Uav.Get(),
        m_attnSmall3Uav.Get()
    };
    ID3D11Buffer* cbs[] = {m_refineConstants.Get()};

    m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 10, s);
    m_context->CSSetUnorderedAccessViews(0, 5, u, nullptr);
    m_context->CSSetConstantBuffers(0, 1, cbs);
    m_context->CSSetSamplers(0, 1, samplers);
    Dispatch(m_smallWidth, m_smallHeight);
    ClearCS(10, 5);
  }

  // =======================================================================
  // STAGE 4: REFINEMENT (Half level)
  // =======================================================================
  {
    RefineConstants rc = {};
    rc.radius      = refineFullR;
    rc.motionScale = static_cast<float>(m_lumaWidth) / static_cast<float>(m_smallWidth);
    rc.useBackward = 0;
    rc.backwardScale = 1.0f;
    rc.attnLearnRate = attnLearnRate;
    rc.attnPriorMix = attnPriorMix;
    rc.attnStability = attnStability;
    m_context->UpdateSubresource(m_refineConstants.Get(), 0, nullptr, &rc, 0, 0);

    ID3D11ShaderResourceView* s[] = {
        m_currLumaSrv.Get(), m_prevLumaSrv.Get(),
        m_currFeature2Srv.Get(), m_prevFeature2Srv.Get(),
        m_currFeature3Srv.Get(), m_prevFeature3Srv.Get(),
        m_motionCoarseSrv.Get(), m_confidenceCoarseSrv.Get(),
        m_motionTinyBackwardSrv.Get(), m_confidenceTinyBackwardSrv.Get()
    };
    ID3D11UnorderedAccessView* u[] = {
        m_motionUav.Get(),
        m_confidenceUav.Get(),
        m_attnFull1Uav.Get(),
        m_attnFull2Uav.Get(),
        m_attnFull3Uav.Get()
    };
    ID3D11Buffer* cbs[] = {m_refineConstants.Get()};

    m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 10, s);
    m_context->CSSetUnorderedAccessViews(0, 5, u, nullptr);
    m_context->CSSetConstantBuffers(0, 1, cbs);
    m_context->CSSetSamplers(0, 1, samplers);
    Dispatch(m_lumaWidth, m_lumaHeight);
    ClearCS(10, 5);
  }

  // =======================================================================
  // STAGE 5: SPATIAL SMOOTHING (Joint Bilateral)
  // =======================================================================
  {
    SmoothConstants sc = {};
    sc.edgeScale = std::clamp(m_smoothEdgeScale, 0.5f, 20.0f);
    sc.confPower = std::clamp(m_smoothConfPower, 0.25f, 4.0f);
    m_context->UpdateSubresource(m_smoothConstants.Get(), 0, nullptr, &sc, 0, 0);

    ID3D11ShaderResourceView* s[] = {m_motionSrv.Get(), m_confidenceSrv.Get(), m_currLumaSrv.Get()};
    ID3D11UnorderedAccessView* u[] = {m_motionSmoothUav.Get(), m_confidenceSmoothUav.Get()};
    ID3D11Buffer* cbs[] = {m_smoothConstants.Get()};

    m_context->CSSetShader(m_motionSmoothCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 3, s);
    m_context->CSSetUnorderedAccessViews(0, 2, u, nullptr);
    m_context->CSSetConstantBuffers(0, 1, cbs);
    Dispatch(m_lumaWidth, m_lumaHeight);
    ClearCS(3, 2);
  }

  return true;
}

// -----------------------------------------------------------------------
// ShaderPath
// -----------------------------------------------------------------------
std::wstring Interpolator::ShaderPath(const wchar_t* filename) const {
  wchar_t path[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, path, MAX_PATH);
  std::wstring exePath(path);
  size_t pos = exePath.find_last_of(L"\\/");
  if (pos == std::wstring::npos) return filename;
  return exePath.substr(0, pos) + L"\\shaders\\" + filename;
}
