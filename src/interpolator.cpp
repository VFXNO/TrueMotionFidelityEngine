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
#include <sstream>

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

// IFNet-Lite + FusionNet-Lite weights - matches HLSL cbuffer AttentionWeightsCB
// Total: 128 trainable parameters + 1 flag + 3 padding = 132 floats = 528 bytes
struct AttentionWeights {
  // === IFNet-Lite: 12->8->16 MLP ===
  // Hidden weights: 8 units x 4 floats (weight-shared)
  float mlpW_h0[4], mlpW_h1[4], mlpW_h2[4], mlpW_h3[4];
  float mlpW_h4[4], mlpW_h5[4], mlpW_h6[4], mlpW_h7[4];
  // Output weights: 4 shared vectors
  float mlpW_out0[4], mlpW_out1[4], mlpW_out2[4], mlpW_out3[4];
  // Hidden biases (8 values in 2 float4)
  float mlpBias_h0[4], mlpBias_h1[4];
  // Output biases (16 values in 4 float4)
  float mlpBias_out0[4], mlpBias_out1[4], mlpBias_out2[4], mlpBias_out3[4];
  // Base weights
  float baseW1[4], baseW2[4], baseW3[4];
  // === FusionNet-Lite: 12->6->4 Synthesis MLP ===
  float synthW_h0[4], synthW_h1[4], synthW_h2[4];
  float synthW_h3[4], synthW_h4[4], synthW_h5[4];
  float synthW_out0[4], synthW_out1[4];
  float synthBias_h0[4], synthBias_h1[4];
  float synthBias_out[4];
  // Control flag + padding
  float useCustomWeights;
  float pad[3];
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
  if (!makeCB(sizeof(AttentionWeights),   m_attentionWeights,  "AttentionWeights"))  return false;

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
  // Bind InterpCB (b0) + AttentionWeightsCB (b1) for FusionNet-Lite synthesis
  ID3D11Buffer* cbs[] = {m_interpConstants.Get(), m_attentionWeights.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_interpolateCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 12, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetConstantBuffers(0, 2, cbs);
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
  // Bind InterpCB (b0) + AttentionWeightsCB (b1) for FusionNet-Lite synthesis
  ID3D11Buffer* cbs[] = {m_interpConstants.Get(), m_attentionWeights.Get()};
  ID3D11SamplerState* samplers[] = {m_linearSampler.Get()};

  m_context->CSSetShader(m_interpolateCs.Get(), nullptr, 0);
  m_context->CSSetShaderResources(0, 12, srvs);
  m_context->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);
  m_context->CSSetConstantBuffers(0, 2, cbs);
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
    ID3D11Buffer* cbs[] = {m_refineConstants.Get(), m_attentionWeights.Get()};

    m_context->CSSetShader(m_motionRefineCs.Get(), nullptr, 0);
    m_context->CSSetShaderResources(0, 10, s);
    m_context->CSSetUnorderedAccessViews(0, 5, u, nullptr);
    m_context->CSSetConstantBuffers(0, 2, cbs);
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

// -----------------------------------------------------------------------
// LoadAttentionWeights - Load MLP weights from JSON file
// -----------------------------------------------------------------------
bool Interpolator::LoadAttentionWeights(const wchar_t* path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }

  // Initialize with default weights so missing keys don't result in zeros
  AttentionWeights weights = {};
  
  // === IFNet-Lite defaults (12->8->16 MLP) ===
  // Hidden layer weights: 8 units
  float mlpH0[] = { 1.12f, -0.31f, 0.48f, 0.86f };
  float mlpH1[] = { -0.49f, 0.84f, 0.24f, -0.29f };
  float mlpH2[] = { 0.34f, 0.27f, -0.44f, 0.75f };
  float mlpH3[] = { 0.58f, -0.72f, 0.18f, 0.31f };
  float mlpH4[] = { -0.27f, 0.41f, 0.95f, -0.33f };
  float mlpH5[] = { 0.73f, 0.11f, -0.29f, 0.63f };
  float mlpH6[] = { 0.43f, -0.55f, 0.71f, -0.21f };
  float mlpH7[] = { -0.62f, 0.38f, 0.15f, 0.47f };
  memcpy(weights.mlpW_h0, mlpH0, 16); memcpy(weights.mlpW_h1, mlpH1, 16);
  memcpy(weights.mlpW_h2, mlpH2, 16); memcpy(weights.mlpW_h3, mlpH3, 16);
  memcpy(weights.mlpW_h4, mlpH4, 16); memcpy(weights.mlpW_h5, mlpH5, 16);
  memcpy(weights.mlpW_h6, mlpH6, 16); memcpy(weights.mlpW_h7, mlpH7, 16);
  
  // Output layer weights: 4 shared vectors
  float mlpO0[] = { 0.10f, -0.05f, 0.02f, 0.07f };
  float mlpO1[] = { -0.03f, 0.01f, 0.05f, -0.04f };
  float mlpO2[] = { 0.00f, 0.03f, -0.02f, 0.06f };
  float mlpO3[] = { 0.05f, -0.02f, 0.04f, -0.01f };
  memcpy(weights.mlpW_out0, mlpO0, 16); memcpy(weights.mlpW_out1, mlpO1, 16);
  memcpy(weights.mlpW_out2, mlpO2, 16); memcpy(weights.mlpW_out3, mlpO3, 16);
  
  // Hidden biases (8 values in 2 float4)
  float biasH0[] = { 0.03f, -0.01f, 0.02f, 0.04f };
  float biasH1[] = { -0.02f, 0.01f, 0.02f, -0.01f };
  memcpy(weights.mlpBias_h0, biasH0, 16);
  memcpy(weights.mlpBias_h1, biasH1, 16);
  
  // Output biases (16 values in 4 float4)
  float biasO0[] = { -0.02f, 0.01f, 0.0f, 0.0f };
  float biasO1[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float biasO2[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float biasO3[] = { 0.0f, 0.0f, -2.0f, 0.0f };  // residual=0, occlusion~0.12, quality=0.5
  memcpy(weights.mlpBias_out0, biasO0, 16); memcpy(weights.mlpBias_out1, biasO1, 16);
  memcpy(weights.mlpBias_out2, biasO2, 16); memcpy(weights.mlpBias_out3, biasO3, 16);
  
  // Base weights
  float bW1[] = { 0.15f, 0.1f, 0.1f, 0.2f };
  float bW2[] = { 0.1f, 0.1f, 0.15f, 0.1f };
  float bW3[] = { 0.1f, 0.1f, 0.1f, 0.1f };
  memcpy(weights.baseW1, bW1, 16); memcpy(weights.baseW2, bW2, 16);
  memcpy(weights.baseW3, bW3, 16);
  
  // === FusionNet-Lite defaults (12->6->4 Synthesis MLP) ===
  float sH0[] = { 0.5f, -0.3f, 0.2f, 0.4f };
  float sH1[] = { -0.2f, 0.6f, 0.1f, -0.3f };
  float sH2[] = { 0.3f, 0.1f, -0.4f, 0.5f };
  float sH3[] = { 0.4f, -0.5f, 0.3f, 0.2f };
  float sH4[] = { -0.1f, 0.4f, 0.6f, -0.2f };
  float sH5[] = { 0.5f, 0.2f, -0.3f, 0.4f };
  memcpy(weights.synthW_h0, sH0, 16); memcpy(weights.synthW_h1, sH1, 16);
  memcpy(weights.synthW_h2, sH2, 16); memcpy(weights.synthW_h3, sH3, 16);
  memcpy(weights.synthW_h4, sH4, 16); memcpy(weights.synthW_h5, sH5, 16);
  float sO0[] = { 0.3f, 0.2f, 0.4f, 0.1f };
  float sO1[] = { 0.2f, 0.3f, 0.3f, 0.2f };
  memcpy(weights.synthW_out0, sO0, 16); memcpy(weights.synthW_out1, sO1, 16);
  float sBH0[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float sBH1[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float sBO[] = { 0.0f, 0.0f, 1.4f, -0.85f };  // blend=0.5, detail=0.5, conf=0.8, sharp=0.3
  memcpy(weights.synthBias_h0, sBH0, 16); memcpy(weights.synthBias_h1, sBH1, 16);
  memcpy(weights.synthBias_out, sBO, 16);
  
  // Simple JSON parsing
  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
  
  // Helper to parse a 4-element float array
  auto parseArray = [&](const char* name, float* out) {
    std::string searchStr = std::string("\"") + name + "\"";
    size_t pos = content.find(searchStr);
    if (pos == std::string::npos) {
      searchStr = name;
      pos = content.find(searchStr);
    }
    if (pos != std::string::npos) {
      size_t arrStart = content.find('[', pos);
      size_t arrEnd = content.find(']', arrStart);
      if (arrStart != std::string::npos && arrEnd != std::string::npos) {
        std::string arr = content.substr(arrStart + 1, arrEnd - arrStart - 1);
        for (char& c : arr) { if (c == ',') c = ' '; }
        std::istringstream iss(arr);
        float val;
        int i = 0;
        while (iss >> val && i < 4) { out[i++] = val; }
      }
    }
  };
  
  // Parse IFNet-Lite hidden weights (8 units)
  parseArray("mlpW_h0", weights.mlpW_h0);
  parseArray("mlpW_h1", weights.mlpW_h1);
  parseArray("mlpW_h2", weights.mlpW_h2);
  parseArray("mlpW_h3", weights.mlpW_h3);
  parseArray("mlpW_h4", weights.mlpW_h4);
  parseArray("mlpW_h5", weights.mlpW_h5);
  parseArray("mlpW_h6", weights.mlpW_h6);
  parseArray("mlpW_h7", weights.mlpW_h7);
  
  // Parse output weights (4 vectors)
  parseArray("mlpW_out0", weights.mlpW_out0);
  parseArray("mlpW_out1", weights.mlpW_out1);
  parseArray("mlpW_out2", weights.mlpW_out2);
  parseArray("mlpW_out3", weights.mlpW_out3);
  
  // Parse biases (backward-compatible: try new names first, fall back to old)
  parseArray("mlpBias_h0", weights.mlpBias_h0);
  parseArray("mlpBias_h1", weights.mlpBias_h1);
  // Backward compat: old "mlpBias_hidden" maps to mlpBias_h0
  if (content.find("\"mlpBias_hidden\"") != std::string::npos) {
    parseArray("mlpBias_hidden", weights.mlpBias_h0);
  }
  parseArray("mlpBias_out0", weights.mlpBias_out0);
  parseArray("mlpBias_out1", weights.mlpBias_out1);
  parseArray("mlpBias_out2", weights.mlpBias_out2);
  parseArray("mlpBias_out3", weights.mlpBias_out3);
  
  // Parse base weights
  parseArray("baseW1", weights.baseW1);
  parseArray("baseW2", weights.baseW2);
  parseArray("baseW3", weights.baseW3);
  
  // Parse FusionNet-Lite synthesis weights (optional, defaults are fine)
  parseArray("synthW_h0", weights.synthW_h0);
  parseArray("synthW_h1", weights.synthW_h1);
  parseArray("synthW_h2", weights.synthW_h2);
  parseArray("synthW_h3", weights.synthW_h3);
  parseArray("synthW_h4", weights.synthW_h4);
  parseArray("synthW_h5", weights.synthW_h5);
  parseArray("synthW_out0", weights.synthW_out0);
  parseArray("synthW_out1", weights.synthW_out1);
  parseArray("synthBias_h0", weights.synthBias_h0);
  parseArray("synthBias_h1", weights.synthBias_h1);
  parseArray("synthBias_out", weights.synthBias_out);
  
  // Set custom weights flag
  weights.useCustomWeights = 1.0f;

  if (!m_attentionWeights) {
    return false;
  }

  // Update the constant buffer
  m_context->UpdateSubresource(m_attentionWeights.Get(), 0, nullptr, &weights, 0, 0);
  m_useCustomWeights = true;
  
  return true;
}

// -----------------------------------------------------------------------
// SaveAttentionWeights - Save MLP weights to JSON file
// -----------------------------------------------------------------------
bool Interpolator::SaveAttentionWeights(const wchar_t* path) const {
  // Default weights (matching IFNet-Lite + FusionNet-Lite HLSL defaults)
  AttentionWeights weights = {};
  
  // IFNet-Lite hidden weights (8 units)
  float mlpH0[] = { 1.12f, -0.31f, 0.48f, 0.86f };
  float mlpH1[] = { -0.49f, 0.84f, 0.24f, -0.29f };
  float mlpH2[] = { 0.34f, 0.27f, -0.44f, 0.75f };
  float mlpH3[] = { 0.58f, -0.72f, 0.18f, 0.31f };
  float mlpH4[] = { -0.27f, 0.41f, 0.95f, -0.33f };
  float mlpH5[] = { 0.73f, 0.11f, -0.29f, 0.63f };
  float mlpH6[] = { 0.43f, -0.55f, 0.71f, -0.21f };
  float mlpH7[] = { -0.62f, 0.38f, 0.15f, 0.47f };
  memcpy(weights.mlpW_h0, mlpH0, 16); memcpy(weights.mlpW_h1, mlpH1, 16);
  memcpy(weights.mlpW_h2, mlpH2, 16); memcpy(weights.mlpW_h3, mlpH3, 16);
  memcpy(weights.mlpW_h4, mlpH4, 16); memcpy(weights.mlpW_h5, mlpH5, 16);
  memcpy(weights.mlpW_h6, mlpH6, 16); memcpy(weights.mlpW_h7, mlpH7, 16);
  
  float mlpO0[] = { 0.10f, -0.05f, 0.02f, 0.07f };
  float mlpO1[] = { -0.03f, 0.01f, 0.05f, -0.04f };
  float mlpO2[] = { 0.00f, 0.03f, -0.02f, 0.06f };
  float mlpO3[] = { 0.05f, -0.02f, 0.04f, -0.01f };
  memcpy(weights.mlpW_out0, mlpO0, 16); memcpy(weights.mlpW_out1, mlpO1, 16);
  memcpy(weights.mlpW_out2, mlpO2, 16); memcpy(weights.mlpW_out3, mlpO3, 16);
  
  float biasH0[] = { 0.03f, -0.01f, 0.02f, 0.04f };
  float biasH1[] = { -0.02f, 0.01f, 0.02f, -0.01f };
  memcpy(weights.mlpBias_h0, biasH0, 16); memcpy(weights.mlpBias_h1, biasH1, 16);
  
  float biasO0[] = { -0.02f, 0.01f, 0.0f, 0.0f };
  float biasO1[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float biasO2[] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float biasO3[] = { 0.0f, 0.0f, -2.0f, 0.0f };
  memcpy(weights.mlpBias_out0, biasO0, 16); memcpy(weights.mlpBias_out1, biasO1, 16);
  memcpy(weights.mlpBias_out2, biasO2, 16); memcpy(weights.mlpBias_out3, biasO3, 16);
  
  float bW1[] = { 0.15f, 0.1f, 0.1f, 0.2f };
  float bW2[] = { 0.1f, 0.1f, 0.15f, 0.1f };
  float bW3[] = { 0.1f, 0.1f, 0.1f, 0.1f };
  memcpy(weights.baseW1, bW1, 16); memcpy(weights.baseW2, bW2, 16);
  memcpy(weights.baseW3, bW3, 16);
  
  // FusionNet-Lite synthesis defaults (zeroed out, will use defaults in shader)
  
  weights.useCustomWeights = 0.0f;
  
  std::ofstream file(path);
  if (!file.is_open()) return false;
  
  auto writeArray = [&](const char* name, float* w) {
    file << "  \"" << name << "\": [" << w[0] << ", " << w[1] << ", " << w[2] << ", " << w[3] << "]";
  };
  
  file << "{\n";
  
  // Save IFNet-Lite hidden weights (8 units)
  float* hiddenWeights[] = { weights.mlpW_h0, weights.mlpW_h1, weights.mlpW_h2,
                              weights.mlpW_h3, weights.mlpW_h4, weights.mlpW_h5,
                              weights.mlpW_h6, weights.mlpW_h7 };
  for (int i = 0; i < 8; i++) {
    char name[16];
    sprintf_s(name, "mlpW_h%d", i);
    writeArray(name, hiddenWeights[i]);
    file << ",\n";
  }
  
  // Save output weights (4 vectors)
  float* outputWeights[] = { weights.mlpW_out0, weights.mlpW_out1, weights.mlpW_out2, weights.mlpW_out3 };
  for (int i = 0; i < 4; i++) {
    char name[16];
    sprintf_s(name, "mlpW_out%d", i);
    writeArray(name, outputWeights[i]);
    file << ",\n";
  }
  
  // Save biases
  writeArray("mlpBias_h0", weights.mlpBias_h0); file << ",\n";
  writeArray("mlpBias_h1", weights.mlpBias_h1); file << ",\n";
  writeArray("mlpBias_out0", weights.mlpBias_out0); file << ",\n";
  writeArray("mlpBias_out1", weights.mlpBias_out1); file << ",\n";
  writeArray("mlpBias_out2", weights.mlpBias_out2); file << ",\n";
  writeArray("mlpBias_out3", weights.mlpBias_out3); file << ",\n";
  
  // Save base weights
  writeArray("baseW1", weights.baseW1); file << ",\n";
  writeArray("baseW2", weights.baseW2); file << ",\n";
  writeArray("baseW3", weights.baseW3); file << "\n";
  
  file << "}\n";
  
  return true;
}

// -----------------------------------------------------------------------
// ExportTrainedWeights - Export EMA-trained weights from GPU textures
// -----------------------------------------------------------------------
bool Interpolator::ExportTrainedWeights(const wchar_t* path) {
  // Use small resolution textures (used in both minimal and full pipeline)
  Microsoft::WRL::ComPtr<ID3D11Texture2D> tex1 = m_attnSmall1;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> tex2 = m_attnSmall2;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> tex3 = m_attnSmall3;
  
  if (!tex1 || !tex2 || !tex3) {
    // Try full resolution
    tex1 = m_attnFull1.Get();
    tex2 = m_attnFull2.Get();
    tex3 = m_attnFull3.Get();
  }
  
  if (!tex1 || !tex2 || !tex3) {
    return false;
  }
  
  // Flush GPU to ensure data is written
  m_context->Flush();
  
  // Get texture dimensions
  D3D11_TEXTURE2D_DESC desc = {};
  tex1->GetDesc(&desc);
  
  if (desc.Width == 0 || desc.Height == 0) {
    return false;
  }
  
  // Create staging textures for reading
  D3D11_TEXTURE2D_DESC stagingDesc = {};
  stagingDesc.Width = desc.Width;
  stagingDesc.Height = desc.Height;
  stagingDesc.MipLevels = 1;
  stagingDesc.ArraySize = 1;
  stagingDesc.Format = desc.Format;
  stagingDesc.SampleDesc.Count = 1;
  stagingDesc.SampleDesc.Quality = 0;
  stagingDesc.Usage = D3D11_USAGE_STAGING;
  stagingDesc.BindFlags = 0;
  stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  stagingDesc.MiscFlags = 0;
  stagingDesc.BindFlags = 0;
  stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  stagingDesc.MiscFlags = 0;
  
  Microsoft::WRL::ComPtr<ID3D11Texture2D> staging1, staging2, staging3;
  HRESULT hr;
  hr = m_device->CreateTexture2D(&stagingDesc, nullptr, &staging1);
  if (FAILED(hr)) return false;
  hr = m_device->CreateTexture2D(&stagingDesc, nullptr, &staging2);
  if (FAILED(hr)) return false;
  hr = m_device->CreateTexture2D(&stagingDesc, nullptr, &staging3);
  if (FAILED(hr)) return false;
  
  // Copy to staging
  m_context->CopyResource(staging1.Get(), tex1.Get());
  m_context->CopyResource(staging2.Get(), tex2.Get());
  m_context->CopyResource(staging3.Get(), tex3.Get());
  
  // Map and read
  D3D11_MAPPED_SUBRESOURCE mapped1 = {};
  D3D11_MAPPED_SUBRESOURCE mapped2 = {};
  D3D11_MAPPED_SUBRESOURCE mapped3 = {};
  
  hr = m_context->Map(staging1.Get(), 0, D3D11_MAP_READ, 0, &mapped1);
  if (FAILED(hr)) return false;
  
  hr = m_context->Map(staging2.Get(), 0, D3D11_MAP_READ, 0, &mapped2);
  if (FAILED(hr)) { m_context->Unmap(staging1.Get(), 0); return false; }
  
  hr = m_context->Map(staging3.Get(), 0, D3D11_MAP_READ, 0, &mapped3);
  if (FAILED(hr)) { m_context->Unmap(staging1.Get(), 0); m_context->Unmap(staging2.Get(), 0); return false; }
  
  // Average all pixels
  float sumW1[4] = {0,0,0,0};
  float sumW2[4] = {0,0,0,0};
  float sumW3[4] = {0,0,0,0};
  float* data1 = (float*)mapped1.pData;
  float* data2 = (float*)mapped2.pData;
  float* data3 = (float*)mapped3.pData;
  
  int pixelCount = desc.Width * desc.Height;
  int rowPitch1 = mapped1.RowPitch / sizeof(float);
  int rowPitch2 = mapped2.RowPitch / sizeof(float);
  int rowPitch3 = mapped3.RowPitch / sizeof(float);
  
  for (int y = 0; y < (int)desc.Height; y++) {
    for (int x = 0; x < (int)desc.Width; x++) {
      int idx1 = y * rowPitch1 + x * 4;
      int idx2 = y * rowPitch2 + x * 4;
      int idx3 = y * rowPitch3 + x * 4;
      
      sumW1[0] += data1[idx1 + 0];
      sumW1[1] += data1[idx1 + 1];
      sumW1[2] += data1[idx1 + 2];
      sumW1[3] += data1[idx1 + 3];
      
      sumW2[0] += data2[idx2 + 0];
      sumW2[1] += data2[idx2 + 1];
      sumW2[2] += data2[idx2 + 2];
      sumW2[3] += data2[idx2 + 3];
      
      sumW3[0] += data3[idx3 + 0];
      sumW3[1] += data3[idx3 + 1];
      sumW3[2] += data3[idx3 + 2];
      sumW3[3] += data3[idx3 + 3];
    }
  }
  
  m_context->Unmap(staging1.Get(), 0);
  m_context->Unmap(staging2.Get(), 0);
  m_context->Unmap(staging3.Get(), 0);
  
  // Normalize by pixel count
  for (int i = 0; i < 4; i++) {
    sumW1[i] /= pixelCount;
    sumW2[i] /= pixelCount;
    sumW3[i] /= pixelCount;
  }
  
  // Normalize each weight vector to sum to 1
  auto normalize = [](float* w) {
    float sum = w[0] + w[1] + w[2] + w[3];
    if (sum > 0.0001f) {
      w[0] /= sum; w[1] /= sum; w[2] /= sum; w[3] /= sum;
    }
  };
  normalize(sumW1);
  normalize(sumW2);
  normalize(sumW3);
  
  // Write to JSON
  std::ofstream file(path);
  if (!file.is_open()) {
    return false;
  }
  
  file << "{\n";
  file << "  \"baseW1\": [" << sumW1[0] << ", " << sumW1[1] << ", " << sumW1[2] << ", " << sumW1[3] << "],\n";
  file << "  \"baseW2\": [" << sumW2[0] << ", " << sumW2[1] << ", " << sumW2[2] << ", " << sumW2[3] << "],\n";
  file << "  \"baseW3\": [" << sumW3[0] << ", " << sumW3[1] << ", " << sumW3[2] << ", " << sumW3[3] << "]\n";
  file << "}\n";
  
  return true;
}
