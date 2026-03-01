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

#ifdef USE_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <dxgi.h>
#endif

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

#ifdef USE_VULKAN
  // Try loading Vulkan compute shaders (non-fatal if fails)
  {
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "LoadVulkanShaders ENTER\n";
    vklog.flush();
  }
  if (LoadVulkanShaders()) {
    m_useVulkan = true;
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "After Vulkan check, m_useVulkan=1\n";
    vklog.flush();
  } else {
    m_useVulkan = false;
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "LoadVulkanShaders failed, falling back to D3D11 compute\n";
    vklog.flush();
  }
#endif

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
  if (m_outputWidth <= 0 || m_outputHeight <= 0 || m_lumaWidth <= 0 || m_lumaHeight <= 0)
    return;

  if (!m_downsampleCs || !m_downsampleLumaCs || !m_motionCs ||
      !m_motionRefineCs || !m_motionSmoothCs || !m_interpolateCs)
    return;

#ifdef USE_VULKAN
  // Full Vulkan PWC-Net pipeline: downsample → cost_volume → flow_decoder → interpolate
  if (m_useVulkan && !m_useMinimalMotionPipeline && m_vkResCreated && m_vkFullPipeline) {
    if (VulkanFullDispatch(prev, curr, std::clamp(alpha, 0.0f, 1.0f))) {
      return;
    }
  }
#endif

  // --- Compute motion field (D3D11 — battle-tested pipeline) ---
  if (!ComputeMotion(prev, curr)) return;

#ifdef USE_VULKAN
  // Hybrid fallback: D3D11 motion + Vulkan interpolation
  if (m_useVulkan && !m_useMinimalMotionPipeline && m_vkResCreated) {
    if (VulkanDispatchInterpolate(prev, curr, std::clamp(alpha, 0.0f, 1.0f))) {
      return;
    }
  }
#endif

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

#ifdef USE_VULKAN
  // Fast Vulkan re-warp: shared textures already have correct data from
  // the first Execute() call.  Only alpha changes — skip ALL D3D11 copies.
  if (m_useVulkan && !m_useMinimalMotionPipeline && m_vkResCreated && m_vkZeroCopy) {
    if (VulkanReWarp(std::clamp(alpha, 0.0f, 1.0f))) {
      return;
    }
  }
#endif

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
  {
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "CreateResources: START, m_useVulkan=" << m_useVulkan
          << " outputW=" << m_outputWidth << " outputH=" << m_outputHeight
          << " lumaW=" << m_lumaWidth << " lumaH=" << m_lumaHeight << "\n";
    vklog.flush();
  }

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
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "CreateResources: FAILED validation - one or more UAVs are null\n";
    vklog.flush();
    return;
  }

  {
    std::ofstream vklog("vulkan_debug.txt", std::ios::app);
    vklog << "CreateResources: SUCCESS\n";
    vklog.flush();
  }

#ifdef USE_VULKAN
  if (m_useVulkan) {
    CreateVulkanResources();
  }
#endif
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

// -----------------------------------------------------------------------
// LoadVulkanShaders: Load SPIR-V compute shaders and create Vulkan pipelines
// -----------------------------------------------------------------------
#ifdef USE_VULKAN

static std::vector<char> ReadSPIRVFile(const std::wstring& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return {};
  size_t size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  file.read(data.data(), size);
  return data;
}

static bool CreateVulkanPipeline(
    VkDevice device,
    const std::vector<char>& spirv,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
    uint32_t pushConstantSize,
    VkPipeline& outPipeline,
    VkPipelineLayout& outLayout,
    VkDescriptorSetLayout& outSetLayout,
    const char* name,
    std::ofstream& log) {

  if (spirv.empty() || spirv.size() % 4 != 0) {
    log << "Invalid SPIR-V data for " << name << " size=" << spirv.size() << "\n";
    log.flush();
    return false;
  }

  // Create shader module
  VkShaderModuleCreateInfo moduleInfo = {};
  moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleInfo.codeSize = spirv.size();
  moduleInfo.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

  log << "Creating shader module, size=" << spirv.size() << "\n";
  log.flush();

  VkShaderModule shaderModule = VK_NULL_HANDLE;
  if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    log << "vkCreateShaderModule failed for " << name << "\n";
    log.flush();
    return false;
  }
  log << "Shader module created successfully\n";
  log.flush();

  // Create descriptor set layout
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &outSetLayout) != VK_SUCCESS) {
    vkDestroyShaderModule(device, shaderModule, nullptr);
    log << "Descriptor set layout creation failed for " << name << "\n";
    log.flush();
    return false;
  }
  log << "Descriptor set layout created\n";
  log.flush();

  // Create pipeline layout with push constants
  VkPushConstantRange pushRange = {};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = pushConstantSize;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &outSetLayout;
  if (pushConstantSize > 0) {
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
  }

  if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(device, outSetLayout, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);
    log << "Pipeline layout creation failed for " << name << "\n";
    log.flush();
    return false;
  }
  log << "Pipeline layout created\n";
  log.flush();

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.layout = outLayout;
  pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shaderModule;
  pipelineInfo.stage.pName = "main";

  log << "About to create compute pipeline: " << name << "\n";
  log.flush();

  VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outPipeline);
  
  // Shader module can be destroyed after pipeline creation
  vkDestroyShaderModule(device, shaderModule, nullptr);

  if (result != VK_SUCCESS) {
    vkDestroyPipelineLayout(device, outLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, outSetLayout, nullptr);
    outLayout = VK_NULL_HANDLE;
    outSetLayout = VK_NULL_HANDLE;
    log << "Compute pipeline creation failed for " << name << " result=" << result << "\n";
    log.flush();
    return false;
  }

  log << "Pipeline created: " << name << "\n";
  log.flush();
  return true;
}

bool Interpolator::LoadVulkanShaders() {
  std::ofstream log("vulkan_debug.txt", std::ios::app);

  if (!m_renderDevice) {
    log << "LoadVulkanShaders: No render device\n";
    log.flush();
    return false;
  }

  VkDevice vkDevice = m_renderDevice->GetVkDevice();
  if (vkDevice == VK_NULL_HANDLE) {
    log << "LoadVulkanShaders: VkDevice is null\n";
    log.flush();
    return false;
  }

  log << "Passed Vulkan check\n";
  log.flush();

  // Build shader directory path
  wchar_t exePath[MAX_PATH] = {};
  GetModuleFileNameW(nullptr, exePath, MAX_PATH);
  std::wstring exeDir(exePath);
  auto pos = exeDir.find_last_of(L"\\/");
  if (pos != std::wstring::npos) exeDir = exeDir.substr(0, pos);
  std::wstring shaderDir = exeDir + L"\\shaders\\vulkan\\";

  // Convert to narrow for logging
  char narrowDir[512] = {};
  WideCharToMultiByte(CP_UTF8, 0, shaderDir.c_str(), -1, narrowDir, sizeof(narrowDir), nullptr, nullptr);
  log << "Shader dir: " << narrowDir << "\n";
  log.flush();

  // Helper to build descriptor bindings
  auto makeSamplerBinding = [](uint32_t binding) -> VkDescriptorSetLayoutBinding {
    VkDescriptorSetLayoutBinding b = {};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
  };
  auto makeStorageImageBinding = [](uint32_t binding) -> VkDescriptorSetLayoutBinding {
    VkDescriptorSetLayoutBinding b = {};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
  };

  // Set layouts are stored per-pipeline for descriptor set allocation

  // --- Feature Pyramid: binding 0=sampler, 2-7=storage images ---
  {
    std::wstring path = shaderDir + L"feature_pyramid.spv";
    auto spirv = ReadSPIRVFile(path);
    char narrowPath[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrowPath, sizeof(narrowPath), nullptr, nullptr);
    log << "Loading: " << narrowPath << "\n";
    log.flush();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      makeSamplerBinding(0),
      makeStorageImageBinding(2), makeStorageImageBinding(3),
      makeStorageImageBinding(4), makeStorageImageBinding(5),
      makeStorageImageBinding(6), makeStorageImageBinding(7)
    };

    if (!CreateVulkanPipeline(vkDevice, spirv, bindings, 120,
        m_vkFeaturePyramidPipeline, m_vkFeaturePyramidLayout, m_vkFeaturePyramidSetLayout,
        "FeaturePyramid", log)) {
      return false;
    }
  }

  // --- Cost Volume: binding 0-12=samplers, 13=storage image ---
  {
    std::wstring path = shaderDir + L"cost_volume.spv";
    auto spirv = ReadSPIRVFile(path);
    char narrowPath[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrowPath, sizeof(narrowPath), nullptr, nullptr);
    log << "Loading: " << narrowPath << "\n";
    log.flush();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      makeSamplerBinding(0), makeSamplerBinding(1),
      makeSamplerBinding(2), makeSamplerBinding(3),
      makeSamplerBinding(4), makeSamplerBinding(5),
      makeSamplerBinding(6), makeSamplerBinding(7),
      makeSamplerBinding(8), makeSamplerBinding(9),
      makeSamplerBinding(10), makeSamplerBinding(11),
      makeSamplerBinding(12),
      makeStorageImageBinding(13)
    };

    if (!CreateVulkanPipeline(vkDevice, spirv, bindings, 56,
        m_vkCostVolumePipeline, m_vkCostVolumeLayout, m_vkCostVolumeSetLayout,
        "CostVolume", log)) {
      return false;
    }
  }

  // --- Flow Decoder: binding 0-3=samplers, 4-5=storage images ---
  {
    std::wstring path = shaderDir + L"flow_decoder.spv";
    auto spirv = ReadSPIRVFile(path);
    char narrowPath[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrowPath, sizeof(narrowPath), nullptr, nullptr);
    log << "Loading: " << narrowPath << "\n";
    log.flush();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      makeSamplerBinding(0), makeSamplerBinding(1),
      makeSamplerBinding(2), makeSamplerBinding(3),
      makeStorageImageBinding(4), makeStorageImageBinding(5)
    };

    if (!CreateVulkanPipeline(vkDevice, spirv, bindings, 56,
        m_vkFlowDecoderPipeline, m_vkFlowDecoderLayout, m_vkFlowDecoderSetLayout,
        "FlowDecoder", log)) {
      return false;
    }
  }

  // --- Interpolate: binding 0-9=samplers, 10=storage image ---
  {
    std::wstring path = shaderDir + L"interpolate.spv";
    auto spirv = ReadSPIRVFile(path);
    char narrowPath[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrowPath, sizeof(narrowPath), nullptr, nullptr);
    log << "Loading: " << narrowPath << "\n";
    log.flush();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      makeSamplerBinding(0), makeSamplerBinding(1),
      makeSamplerBinding(2), makeSamplerBinding(3),
      makeSamplerBinding(4), makeSamplerBinding(5),
      makeSamplerBinding(6), makeSamplerBinding(7),
      makeSamplerBinding(8), makeSamplerBinding(9),
      makeStorageImageBinding(10)
    };

    if (!CreateVulkanPipeline(vkDevice, spirv, bindings, 60,
        m_vkInterpolatePipeline, m_vkInterpolateLayout, m_vkInterpolateSetLayout,
        "Interpolate", log)) {
      return false;
    }
  }

  // --- Downsample: binding 0=sampler, 2-5=storage images, 6-8=samplers ---
  {
    std::wstring path = shaderDir + L"downsample.spv";
    auto spirv = ReadSPIRVFile(path);
    char narrowPath[512] = {};
    WideCharToMultiByte(CP_UTF8, 0, path.c_str(), -1, narrowPath, sizeof(narrowPath), nullptr, nullptr);
    log << "Loading: " << narrowPath << "\n";
    log.flush();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
      makeSamplerBinding(0),
      makeStorageImageBinding(2), makeStorageImageBinding(3),
      makeStorageImageBinding(4), makeStorageImageBinding(5),
      makeSamplerBinding(6), makeSamplerBinding(7), makeSamplerBinding(8)
    };

    if (!CreateVulkanPipeline(vkDevice, spirv, bindings, 88,
        m_vkDownsamplePipeline, m_vkDownsampleLayout, m_vkDownsampleSetLayout,
        "Downsample", log)) {
      return false;
    }
  }

  log << "All Vulkan shaders loaded successfully\n";
  log.flush();
  return true;
}

// -----------------------------------------------------------------------
// Vulkan Compute Dispatch Implementation
// -----------------------------------------------------------------------

static uint32_t FindVkMemoryType(VkPhysicalDevice phys, uint32_t filter, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((filter & (1 << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  return UINT32_MAX;
}

bool Interpolator::CreateVkImg(VkImg& img, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
  VkDevice dev = m_renderDevice->GetVkDevice();
  VkPhysicalDevice phys = m_renderDevice->GetVkPhysicalDevice();

  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = fmt;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(dev, &ci, nullptr, &img.image) != VK_SUCCESS) return false;

  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(dev, img.image, &req);
  VkMemoryAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindVkMemoryType(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX) {
    vkDestroyImage(dev, img.image, nullptr); img.image = VK_NULL_HANDLE; return false;
  }
  if (vkAllocateMemory(dev, &ai, nullptr, &img.memory) != VK_SUCCESS) {
    vkDestroyImage(dev, img.image, nullptr); img.image = VK_NULL_HANDLE; return false;
  }
  vkBindImageMemory(dev, img.image, img.memory, 0);

  VkImageViewCreateInfo vi = {};
  vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vi.image = img.image;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = fmt;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(dev, &vi, nullptr, &img.view) != VK_SUCCESS) {
    vkFreeMemory(dev, img.memory, nullptr);
    vkDestroyImage(dev, img.image, nullptr);
    img = {}; return false;
  }
  img.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  return true;
}

void Interpolator::DestroyVkImg(VkImg& img) {
  if (!m_renderDevice) return;
  VkDevice dev = m_renderDevice->GetVkDevice();
  if (img.view) vkDestroyImageView(dev, img.view, nullptr);
  if (img.image) vkDestroyImage(dev, img.image, nullptr);
  if (img.memory) vkFreeMemory(dev, img.memory, nullptr);
  img = {};
}

// -----------------------------------------------------------------------
// CreateSharedImg: Creates a D3D11 texture with SHARED flag and
// imports it into Vulkan via VK_KHR_external_memory_win32 (KMT handle).
// The same GPU memory is visible to both APIs — zero CPU copies.
// -----------------------------------------------------------------------
bool Interpolator::CreateSharedImg(SharedImg& s, uint32_t w, uint32_t h,
    DXGI_FORMAT d3dFmt, VkFormat vkFmt, VkImageUsageFlags vkUsage) {
  VkDevice dev = m_renderDevice->GetVkDevice();
  VkPhysicalDevice phys = m_renderDevice->GetVkPhysicalDevice();
  s = {};
  s.w = w; s.h = h;

  // 1. Create D3D11 texture with legacy SHARED flag (KMT handle, universally supported)
  D3D11_TEXTURE2D_DESC desc = {};
  desc.Width = w; desc.Height = h;
  desc.MipLevels = 1; desc.ArraySize = 1;
  desc.Format = d3dFmt;
  desc.SampleDesc.Count = 1;
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
  desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
  HRESULT hr = m_device->CreateTexture2D(&desc, nullptr, &s.d3dTex);
  if (FAILED(hr)) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "CreateSharedImg: CreateTexture2D failed, hr=0x" << std::hex << hr
        << " fmt=" << std::dec << d3dFmt << " " << w << "x" << h << "\n";
    log.flush();
    return false;
  }

  // 2. Get the KMT shared handle
  Microsoft::WRL::ComPtr<IDXGIResource> dxgiRes;
  if (FAILED(s.d3dTex.As(&dxgiRes))) { s.d3dTex.Reset(); return false; }
  if (FAILED(dxgiRes->GetSharedHandle(&s.ntHandle))) {
    s.d3dTex.Reset(); return false;
  }

  // 3. Create VkImage with external memory info (KMT handle type)
  VkExternalMemoryImageCreateInfo extCI = {};
  extCI.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
  extCI.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;

  VkImageCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ci.pNext = &extCI;
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = vkFmt;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1; ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = vkUsage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (vkCreateImage(dev, &ci, nullptr, &s.vkImage) != VK_SUCCESS) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "CreateSharedImg: vkCreateImage failed " << w << "x" << h << " fmt=" << vkFmt << "\n";
    log.flush();
    s.d3dTex.Reset(); return false;
  }

  // 4. Import D3D11 KMT shared handle as VkDeviceMemory
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(dev, s.vkImage, &req);

  VkImportMemoryWin32HandleInfoKHR importInfo = {};
  importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
  importInfo.handle = s.ntHandle;

  VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
  dedicatedInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicatedInfo.image = s.vkImage;
  importInfo.pNext = &dedicatedInfo;

  VkMemoryAllocateInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.pNext = &importInfo;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = FindVkMemoryType(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (ai.memoryTypeIndex == UINT32_MAX) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "CreateSharedImg: no suitable memory type, bits=0x" << std::hex << req.memoryTypeBits << "\n";
    log.flush();
    vkDestroyImage(dev, s.vkImage, nullptr); s.vkImage = VK_NULL_HANDLE;
    s.d3dTex.Reset(); return false;
  }
  VkResult vr = vkAllocateMemory(dev, &ai, nullptr, &s.vkMemory);
  if (vr != VK_SUCCESS) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "CreateSharedImg: vkAllocateMemory failed, result=" << vr << "\n";
    log.flush();
    vkDestroyImage(dev, s.vkImage, nullptr); s.vkImage = VK_NULL_HANDLE;
    s.d3dTex.Reset(); return false;
  }
  if (vkBindImageMemory(dev, s.vkImage, s.vkMemory, 0) != VK_SUCCESS) {
    vkFreeMemory(dev, s.vkMemory, nullptr); s.vkMemory = VK_NULL_HANDLE;
    vkDestroyImage(dev, s.vkImage, nullptr); s.vkImage = VK_NULL_HANDLE;
    s.d3dTex.Reset(); return false;
  }

  // 5. Create VkImageView
  VkImageViewCreateInfo vi = {};
  vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  vi.image = s.vkImage;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = vkFmt;
  vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  if (vkCreateImageView(dev, &vi, nullptr, &s.vkView) != VK_SUCCESS) {
    vkFreeMemory(dev, s.vkMemory, nullptr);
    vkDestroyImage(dev, s.vkImage, nullptr);
    s = {}; return false;
  }

  return true;
}

void Interpolator::DestroySharedImg(SharedImg& s) {
  if (!m_renderDevice) return;
  VkDevice dev = m_renderDevice->GetVkDevice();
  if (s.vkView) vkDestroyImageView(dev, s.vkView, nullptr);
  if (s.vkImage) vkDestroyImage(dev, s.vkImage, nullptr);
  if (s.vkMemory) vkFreeMemory(dev, s.vkMemory, nullptr);
  // KMT handles are not closeable (not NT handles)
  s = {};
}

void Interpolator::CreateVulkanResources() {
  std::ofstream log("vulkan_debug.txt", std::ios::app);
  log << "CreateVulkanResources: START\n"; log.flush();
  if (!m_renderDevice || !m_renderDevice->IsVulkan()) return;

  DestroyVulkanResources();

  VkDevice dev = m_renderDevice->GetVkDevice();
  uint32_t oW = (uint32_t)m_outputWidth, oH = (uint32_t)m_outputHeight;
  uint32_t iW = (uint32_t)m_inputWidth, iH = (uint32_t)m_inputHeight;
  uint32_t hW = (uint32_t)m_lumaWidth, hH = (uint32_t)m_lumaHeight;
  if (!oW || !oH || !iW || !iH || !hW || !hH) return;

  log << "  input=" << iW << "x" << iH << " output=" << oW << "x" << oH
      << " luma=" << hW << "x" << hH << "\n";
  log.flush();

  // Try zero-copy shared textures (D3D11<->Vulkan, same GPU memory)
  VkImageUsageFlags sampU = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  VkImageUsageFlags storU = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                          | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  // Intermediate images need both sampled (for next stage) and storage (for writing)
  VkImageUsageFlags intU = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

  bool zc = true;
  zc &= CreateSharedImg(m_sharedPrev, iW, iH,
      DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, sampU);
  zc &= CreateSharedImg(m_sharedCurr, iW, iH,
      DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, sampU);
  zc &= CreateSharedImg(m_sharedMotion, hW, hH,
      DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, sampU);
  zc &= CreateSharedImg(m_sharedConf, hW, hH,
      DXGI_FORMAT_R16_FLOAT, VK_FORMAT_R16_SFLOAT, sampU);
  zc &= CreateSharedImg(m_sharedFeatPrev, hW, hH,
      DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, sampU);
  zc &= CreateSharedImg(m_sharedFeatCurr, hW, hH,
      DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, sampU);
  zc &= CreateSharedImg(m_sharedOutput, oW, oH,
      DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, storU);

  m_vkZeroCopy = zc;
  if (zc) {
    log << "CreateVulkanResources: ZERO-COPY shared textures OK\n"; log.flush();
  } else {
    log << "CreateVulkanResources: Shared texture creation FAILED, cannot proceed\n"; log.flush();
    DestroyVulkanResources();
    return;
  }

  // ---- Vulkan-only intermediate images for full pipeline ----
  bool fullOk = true;
  fullOk &= CreateVkImg(m_vkFeatPrev, hW, hH, VK_FORMAT_R16G16B16A16_SFLOAT, intU);
  fullOk &= CreateVkImg(m_vkFeatCurr, hW, hH, VK_FORMAT_R16G16B16A16_SFLOAT, intU);
  fullOk &= CreateVkImg(m_vkCostVol,  hW, hH, VK_FORMAT_R16G16B16A16_SFLOAT, intU);
  fullOk &= CreateVkImg(m_vkFlowOut,  hW, hH, VK_FORMAT_R16G16_SFLOAT, intU);
  fullOk &= CreateVkImg(m_vkConfOut,  hW, hH, VK_FORMAT_R16_SFLOAT, intU);
  fullOk &= CreateVkImg(m_vkDummy, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, intU);
  if (fullOk) {
    log << "CreateVulkanResources: Intermediate VkImages OK\n"; log.flush();
  } else {
    log << "CreateVulkanResources: Intermediate image creation failed (non-fatal)\n"; log.flush();
  }

  // Samplers
  {
    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &si, nullptr, &m_vkLinearSampler) != VK_SUCCESS) {
      DestroyVulkanResources(); return;
    }
  }
  {
    VkSamplerCreateInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(dev, &si, nullptr, &m_vkPointSampler) != VK_SUCCESS) {
      DestroyVulkanResources(); return;
    }
  }

  // Fence
  {
    VkFenceCreateInfo fi = {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(dev, &fi, nullptr, &m_vkComputeFence) != VK_SUCCESS) {
      DestroyVulkanResources(); return;
    }
  }

  VkDescriptorPool pool = m_renderDevice->GetVkDescriptorPool();

  // Allocate interpolation descriptor set (for D3D11 motion + VK interpolate path)
  if (m_vkInterpolateSetLayout) {
    VkDescriptorSetAllocateInfo da = {};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &m_vkInterpolateSetLayout;
    if (vkAllocateDescriptorSets(dev, &da, &m_vkInterpolateSet) != VK_SUCCESS) {
      log << "CreateVulkanResources: InterpolateSet alloc failed\n";
      m_vkInterpolateSet = VK_NULL_HANDLE;
    }
  }

  // ---- Allocate descriptor sets for full Vulkan pipeline ----
  auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet& outSet, const char* name) {
    if (!layout) { log << "  " << name << ": no layout\n"; return; }
    VkDescriptorSetAllocateInfo da = {};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &layout;
    if (vkAllocateDescriptorSets(dev, &da, &outSet) != VK_SUCCESS) {
      log << "  " << name << ": alloc FAILED\n"; outSet = VK_NULL_HANDLE;
    }
  };

  allocSet(m_vkDownsampleSetLayout, m_vkDownsamplePrevSet, "DownsamplePrev");
  allocSet(m_vkDownsampleSetLayout, m_vkDownsampleCurrSet, "DownsampleCurr");
  allocSet(m_vkCostVolumeSetLayout, m_vkCostVolumeFullSet, "CostVolumeFull");
  allocSet(m_vkFlowDecoderSetLayout, m_vkFlowDecoderFullSet, "FlowDecoderFull");
  allocSet(m_vkInterpolateSetLayout, m_vkInterpolateFullSet, "InterpolateFull");

  // ---- Pre-configure descriptor sets (all bindings set once, no per-frame updates) ----
  if (fullOk && m_vkDownsamplePrevSet && m_vkDownsampleCurrSet &&
      m_vkCostVolumeFullSet && m_vkFlowDecoderFullSet && m_vkInterpolateFullSet) {

    VkDescriptorImageInfo dummySamp = {m_vkLinearSampler, m_vkDummy.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo dummyStor = {VK_NULL_HANDLE, m_vkDummy.view, VK_IMAGE_LAYOUT_GENERAL};

    // -- Downsample Prev: binding 0=sharedPrev, 2=featPrev(write), 3-5=dummy(write), 6-8=dummy(samp) --
    {
      VkDescriptorImageInfo si[8];
      si[0] = {m_vkLinearSampler, m_sharedPrev.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[1] = {VK_NULL_HANDLE, m_vkFeatPrev.view, VK_IMAGE_LAYOUT_GENERAL}; // binding 2
      si[2] = dummyStor; // binding 3
      si[3] = dummyStor; // binding 4
      si[4] = dummyStor; // binding 5
      si[5] = dummySamp; // binding 6
      si[6] = dummySamp; // binding 7
      si[7] = dummySamp; // binding 8

      VkWriteDescriptorSet ws[8] = {};
      int bindings[] = {0, 2, 3, 4, 5, 6, 7, 8};
      VkDescriptorType types[] = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      };
      for (int i = 0; i < 8; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = m_vkDownsamplePrevSet;
        ws[i].dstBinding = bindings[i];
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = types[i];
        ws[i].pImageInfo = &si[i];
      }
      vkUpdateDescriptorSets(dev, 8, ws, 0, nullptr);
    }

    // -- Downsample Curr: binding 0=sharedCurr, 2=featCurr(write), 3-5=dummy, 6-8=dummy --
    {
      VkDescriptorImageInfo si[8];
      si[0] = {m_vkLinearSampler, m_sharedCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[1] = {VK_NULL_HANDLE, m_vkFeatCurr.view, VK_IMAGE_LAYOUT_GENERAL};
      si[2] = dummyStor; si[3] = dummyStor; si[4] = dummyStor;
      si[5] = dummySamp; si[6] = dummySamp; si[7] = dummySamp;

      VkWriteDescriptorSet ws[8] = {};
      int bindings[] = {0, 2, 3, 4, 5, 6, 7, 8};
      VkDescriptorType types[] = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
      };
      for (int i = 0; i < 8; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = m_vkDownsampleCurrSet;
        ws[i].dstBinding = bindings[i];
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = types[i];
        ws[i].pImageInfo = &si[i];
      }
      vkUpdateDescriptorSets(dev, 8, ws, 0, nullptr);
    }

    // -- Cost Volume: bindings 0-11=samplers(features), 12=prev flow sampler, 13=costVol(write) --
    {
      VkDescriptorImageInfo featPrevSamp = {m_vkLinearSampler, m_vkFeatPrev.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkDescriptorImageInfo featCurrSamp = {m_vkLinearSampler, m_vkFeatCurr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

      VkDescriptorImageInfo si[14];
      // binding 0=feat0_level0(prev), 1=feat1_level0(curr), 2-11=dummies
      si[0] = featPrevSamp;
      si[1] = featCurrSamp;
      for (int i = 2; i < 12; i++) si[i] = dummySamp; // unused levels
      si[12] = dummySamp; // flowPrev (no previous flow for first iteration)
      si[13] = {VK_NULL_HANDLE, m_vkCostVol.view, VK_IMAGE_LAYOUT_GENERAL};

      VkWriteDescriptorSet ws[14] = {};
      for (int i = 0; i < 13; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = m_vkCostVolumeFullSet;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[i].pImageInfo = &si[i];
      }
      ws[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[13].dstSet = m_vkCostVolumeFullSet;
      ws[13].dstBinding = 13;
      ws[13].descriptorCount = 1;
      ws[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      ws[13].pImageInfo = &si[13];
      vkUpdateDescriptorSets(dev, 14, ws, 0, nullptr);
    }

    // -- Flow Decoder: 0=costVol, 1=featPrev, 2=featCurr, 3=prevFlow(dummy),
    //                  4=flowOut(write), 5=confOut(write) --
    {
      VkDescriptorImageInfo si[6];
      si[0] = {m_vkLinearSampler, m_vkCostVol.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[1] = {m_vkLinearSampler, m_vkFeatPrev.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[2] = {m_vkLinearSampler, m_vkFeatCurr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[3] = dummySamp; // no previous flow
      si[4] = {VK_NULL_HANDLE, m_vkFlowOut.view, VK_IMAGE_LAYOUT_GENERAL};
      si[5] = {VK_NULL_HANDLE, m_vkConfOut.view, VK_IMAGE_LAYOUT_GENERAL};

      VkWriteDescriptorSet ws[6] = {};
      VkDescriptorType types[] = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
      };
      for (int i = 0; i < 6; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = m_vkFlowDecoderFullSet;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = types[i];
        ws[i].pImageInfo = &si[i];
      }
      vkUpdateDescriptorSets(dev, 6, ws, 0, nullptr);
    }

    // -- Interpolate Full: uses VK intermediates for motion/conf/feat instead of shared --
    // binding 0=prev, 1=curr, 2=flowFwd, 3=flowFwd(bwd=same), 4=confFwd, 5=confFwd,
    //         6=featPrev, 7=featCurr, 8=linear(curr), 9=point(flow), 10=output
    {
      VkDescriptorImageInfo si[10];
      si[0] = {m_vkLinearSampler, m_sharedPrev.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[1] = {m_vkLinearSampler, m_sharedCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[2] = {m_vkLinearSampler, m_vkFlowOut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[3] = {m_vkLinearSampler, m_vkFlowOut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[4] = {m_vkLinearSampler, m_vkConfOut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[5] = {m_vkLinearSampler, m_vkConfOut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[6] = {m_vkLinearSampler, m_vkFeatPrev.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[7] = {m_vkLinearSampler, m_vkFeatCurr.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[8] = {m_vkLinearSampler, m_sharedCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      si[9] = {m_vkPointSampler, m_vkFlowOut.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

      VkDescriptorImageInfo oi = {};
      oi.imageView = m_sharedOutput.vkView;
      oi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

      VkWriteDescriptorSet ws[11] = {};
      for (int i = 0; i < 10; i++) {
        ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet = m_vkInterpolateFullSet;
        ws[i].dstBinding = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[i].pImageInfo = &si[i];
      }
      ws[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[10].dstSet = m_vkInterpolateFullSet;
      ws[10].dstBinding = 10;
      ws[10].descriptorCount = 1;
      ws[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      ws[10].pImageInfo = &oi;
      vkUpdateDescriptorSets(dev, 11, ws, 0, nullptr);
    }

    m_vkFullPipeline = true;
    log << "CreateVulkanResources: Full VK pipeline descriptor sets configured\n"; log.flush();
  } else {
    log << "CreateVulkanResources: Full VK pipeline NOT available (intermediates or sets failed)\n";
    log.flush();
  }

  m_vkResCreated = true;
  log << "CreateVulkanResources: SUCCESS (zero-copy=" << m_vkZeroCopy
      << " fullPipeline=" << m_vkFullPipeline << ")\n"; log.flush();
}

void Interpolator::DestroyVulkanResources() {
  if (!m_renderDevice) return;
  VkDevice dev = m_renderDevice->GetVkDevice();
  if (!dev) return;
  vkDeviceWaitIdle(dev);

  DestroySharedImg(m_sharedPrev); DestroySharedImg(m_sharedCurr);
  DestroySharedImg(m_sharedMotion); DestroySharedImg(m_sharedConf);
  DestroySharedImg(m_sharedFeatPrev); DestroySharedImg(m_sharedFeatCurr);
  DestroySharedImg(m_sharedOutput);
  DestroyVkImg(m_vkImgOutput);

  // Destroy full pipeline intermediates
  DestroyVkImg(m_vkFeatPrev); DestroyVkImg(m_vkFeatCurr);
  DestroyVkImg(m_vkCostVol);
  DestroyVkImg(m_vkFlowOut); DestroyVkImg(m_vkConfOut);
  DestroyVkImg(m_vkDummy);

  if (m_vkLinearSampler) { vkDestroySampler(dev, m_vkLinearSampler, nullptr); m_vkLinearSampler = VK_NULL_HANDLE; }
  if (m_vkPointSampler) { vkDestroySampler(dev, m_vkPointSampler, nullptr); m_vkPointSampler = VK_NULL_HANDLE; }
  if (m_vkComputeFence) { vkDestroyFence(dev, m_vkComputeFence, nullptr); m_vkComputeFence = VK_NULL_HANDLE; }

  // Descriptor sets are freed when pool is reset/destroyed — just null them
  m_vkInterpolateSet = VK_NULL_HANDLE;
  m_vkDownsamplePrevSet = VK_NULL_HANDLE;
  m_vkDownsampleCurrSet = VK_NULL_HANDLE;
  m_vkCostVolumeFullSet = VK_NULL_HANDLE;
  m_vkFlowDecoderFullSet = VK_NULL_HANDLE;
  m_vkInterpolateFullSet = VK_NULL_HANDLE;

  m_vkResCreated = false;
  m_vkZeroCopy = false;
  m_vkFullPipeline = false;
}

// -----------------------------------------------------------------------
// VulkanDispatchInterpolate: ZERO-COPY path
//   D3D11 CopyResource (GPU→GPU) into shared textures → Vulkan reads directly.
//   Output: Vulkan writes shared output → D3D11 CopyResource to m_outputTexture.
//   NO staging buffers. NO CPU copies. NO memcpy. All GPU.
// -----------------------------------------------------------------------
bool Interpolator::VulkanDispatchInterpolate(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    float alpha) {
  if (!m_vkResCreated || !m_renderDevice || !m_vkInterpolateSet) return false;
  if (m_vkInterpolatePipeline == VK_NULL_HANDLE) return false;
  if (!m_vkZeroCopy) return false;

  VkDevice dev = m_renderDevice->GetVkDevice();
  VkQueue queue = m_renderDevice->GetVkComputeQueue();
  VkCommandBuffer cmd = m_renderDevice->GetVkCommandBuffer();
  uint32_t oW = (uint32_t)m_outputWidth, oH = (uint32_t)m_outputHeight;
  uint32_t iW = (uint32_t)m_inputWidth, iH = (uint32_t)m_inputHeight;
  uint32_t hW = (uint32_t)m_lumaWidth, hH = (uint32_t)m_lumaHeight;

  // One-time dispatch log
  static bool loggedOnce = false;
  if (!loggedOnce) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "VulkanDispatchInterpolate: ENTER (zero-copy) input=" << iW << "x" << iH
        << " output=" << oW << "x" << oH << " luma=" << hW << "x" << hH << "\n";
    log.flush();
    loggedOnce = true;
  }

  // Select best motion/confidence from D3D11 pipeline
  ID3D11ShaderResourceView* motSrv =
      m_motionSmoothSrv ? m_motionSmoothSrv.Get() : m_motionSrv.Get();
  ID3D11ShaderResourceView* cfSrv =
      m_confidenceSmoothSrv ? m_confidenceSmoothSrv.Get() : m_confidenceSrv.Get();

  // ---- D3D11 side: GPU-to-GPU copy into shared textures ----
  // These are fast internal GPU copies (no CPU, no staging).
  auto d3dCopyFromSrv = [&](ID3D11ShaderResourceView* srv, ID3D11Texture2D* dst) {
    if (!srv || !dst) return;
    Microsoft::WRL::ComPtr<ID3D11Resource> res;
    srv->GetResource(&res);
    m_context->CopyResource(dst, res.Get());
  };

  d3dCopyFromSrv(prev,                m_sharedPrev.d3dTex.Get());
  d3dCopyFromSrv(curr,                m_sharedCurr.d3dTex.Get());
  d3dCopyFromSrv(motSrv,              m_sharedMotion.d3dTex.Get());
  d3dCopyFromSrv(cfSrv,               m_sharedConf.d3dTex.Get());
  d3dCopyFromSrv(m_prevLumaSrv.Get(), m_sharedFeatPrev.d3dTex.Get());
  d3dCopyFromSrv(m_currLumaSrv.Get(), m_sharedFeatCurr.d3dTex.Get());

  // Flush D3D11 to ensure all copies are committed to GPU before Vulkan reads
  m_context->Flush();

  // ---- Vulkan side: single command buffer ----
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo cbi = {};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &cbi);

  // Barrier: transition shared input images UNDEFINED → SHADER_READ_ONLY
  //          and output UNDEFINED → GENERAL
  VkImageMemoryBarrier barriers[7] = {};
  VkImage inputImgs[] = {
    m_sharedPrev.vkImage, m_sharedCurr.vkImage,
    m_sharedMotion.vkImage, m_sharedConf.vkImage,
    m_sharedFeatPrev.vkImage, m_sharedFeatCurr.vkImage,
  };
  for (int i = 0; i < 6; i++) {
    barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[i].image = inputImgs[i];
    barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  }
  barriers[6].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barriers[6].srcQueueFamilyIndex = barriers[6].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barriers[6].image = m_sharedOutput.vkImage;
  barriers[6].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  barriers[6].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barriers[6].newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barriers[6].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 7, barriers);

  // Update descriptor set with shared image views
  {
    VkDescriptorImageInfo si[10];
    si[0] = {m_vkLinearSampler, m_sharedPrev.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[1] = {m_vkLinearSampler, m_sharedCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[2] = {m_vkLinearSampler, m_sharedMotion.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[3] = {m_vkLinearSampler, m_sharedMotion.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[4] = {m_vkLinearSampler, m_sharedConf.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[5] = {m_vkLinearSampler, m_sharedConf.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[6] = {m_vkLinearSampler, m_sharedFeatPrev.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[7] = {m_vkLinearSampler, m_sharedFeatCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[8] = {m_vkLinearSampler, m_sharedCurr.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    si[9] = {m_vkPointSampler, m_sharedMotion.vkView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkDescriptorImageInfo oi = {};
    oi.imageView = m_sharedOutput.vkView;
    oi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet ws[11] = {};
    for (int i = 0; i < 10; i++) {
      ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      ws[i].dstSet = m_vkInterpolateSet;
      ws[i].dstBinding = i;
      ws[i].descriptorCount = 1;
      ws[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      ws[i].pImageInfo = &si[i];
    }
    ws[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    ws[10].dstSet = m_vkInterpolateSet;
    ws[10].dstBinding = 10;
    ws[10].descriptorCount = 1;
    ws[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ws[10].pImageInfo = &oi;
    vkUpdateDescriptorSets(dev, 11, ws, 0, nullptr);
  }

  // Bind pipeline + push constants
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkInterpolatePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      m_vkInterpolateLayout, 0, 1, &m_vkInterpolateSet, 0, nullptr);

  struct {
    float frameSize[4];
    float motionSize[4];
    float alpha;
    float confPower;
    float diffScale;
    int useBidirectional;
    int useOcclusion;
    int qualityMode;
    int padding;
  } pc;
  pc.frameSize[0] = (float)oW; pc.frameSize[1] = (float)oH;
  pc.frameSize[2] = 1.0f / (float)oW; pc.frameSize[3] = 1.0f / (float)oH;
  pc.motionSize[0] = (float)hW; pc.motionSize[1] = (float)hH;
  pc.motionSize[2] = 1.0f / (float)hW; pc.motionSize[3] = 1.0f / (float)hH;
  pc.alpha = alpha;
  pc.confPower = std::clamp(m_confPower, 0.25f, 4.0f);
  pc.diffScale = 2.0f;
  pc.useBidirectional = 0;
  pc.useOcclusion = (m_qualityMode >= 1) ? 1 : 0;
  pc.qualityMode = m_qualityMode;
  pc.padding = 0;
  vkCmdPushConstants(cmd, m_vkInterpolateLayout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, 60, &pc);

  // Dispatch
  vkCmdDispatch(cmd, (oW + 15) / 16, (oH + 15) / 16, 1);

  // Barrier: output GENERAL → shader complete (ensure writes visible)
  {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_sharedOutput.vkImage;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  // Submit and wait
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si = {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, m_vkComputeFence);
  vkWaitForFences(dev, 1, &m_vkComputeFence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &m_vkComputeFence);

  // ---- D3D11 readback: shared output → m_outputTexture (GPU copy, no CPU) ----
  m_context->CopyResource(m_outputTexture.Get(), m_sharedOutput.d3dTex.Get());

  return true;
}

// -----------------------------------------------------------------------
// VulkanReWarp: FAST re-warp path for InterpolateOnly
//   Data already present from first Execute() call.  Only alpha changes.
//   Handles both full-VK path (intermediates) and hybrid path (shared textures).
//   NO D3D11 CopyResource.  NO Flush.  Just barriers + dispatch + fence.
// -----------------------------------------------------------------------
bool Interpolator::VulkanReWarp(float alpha) {
  if (!m_vkResCreated || !m_renderDevice) return false;
  if (m_vkInterpolatePipeline == VK_NULL_HANDLE) return false;
  if (!m_vkZeroCopy) return false;

  // Choose descriptor set: full-VK path uses VK intermediates for motion,
  // hybrid path uses shared D3D11 textures.
  bool fullPath = m_vkFullPipeline && m_vkInterpolateFullSet;
  VkDescriptorSet interpSet = fullPath ? m_vkInterpolateFullSet : m_vkInterpolateSet;
  if (!interpSet) return false;

  VkDevice dev = m_renderDevice->GetVkDevice();
  VkQueue queue = m_renderDevice->GetVkComputeQueue();
  VkCommandBuffer cmd = m_renderDevice->GetVkCommandBuffer();
  uint32_t oW = (uint32_t)m_outputWidth, oH = (uint32_t)m_outputHeight;
  uint32_t hW = (uint32_t)m_lumaWidth, hH = (uint32_t)m_lumaHeight;

  // ---- Vulkan command buffer ----
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo cbi = {};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &cbi);

  if (fullPath) {
    // Full-VK path: VK intermediates hold motion data from VulkanFullDispatch
    VkImageMemoryBarrier barriers[7] = {};
    int n = 0;
    auto addBar = [&](VkImage img, VkImageLayout newLay, VkAccessFlags access) {
      barriers[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[n].srcQueueFamilyIndex = barriers[n].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[n].image = img;
      barriers[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      barriers[n].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[n].newLayout = newLay;
      barriers[n].dstAccessMask = access;
      n++;
    };
    addBar(m_sharedPrev.vkImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_sharedCurr.vkImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_vkFlowOut.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_vkConfOut.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_vkFeatPrev.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_vkFeatCurr.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_sharedOutput.vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, n, barriers);
  } else {
    // Hybrid path: shared D3D11 textures hold motion data
    VkImageMemoryBarrier barriers[7] = {};
    VkImage imgs[] = {
      m_sharedPrev.vkImage, m_sharedCurr.vkImage,
      m_sharedMotion.vkImage, m_sharedConf.vkImage,
      m_sharedFeatPrev.vkImage, m_sharedFeatCurr.vkImage,
    };
    for (int i = 0; i < 6; i++) {
      barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barriers[i].srcQueueFamilyIndex = barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barriers[i].image = imgs[i];
      barriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      barriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    barriers[6].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[6].srcQueueFamilyIndex = barriers[6].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[6].image = m_sharedOutput.vkImage;
    barriers[6].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barriers[6].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[6].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[6].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 7, barriers);
  }

  // Bind pipeline + descriptor set
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkInterpolatePipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
      m_vkInterpolateLayout, 0, 1, &interpSet, 0, nullptr);

  // Push constants — only alpha differs from the first dispatch
  struct {
    float frameSize[4];
    float motionSize[4];
    float alpha;
    float confPower;
    float diffScale;
    int useBidirectional;
    int useOcclusion;
    int qualityMode;
    int padding;
  } pc;
  pc.frameSize[0] = (float)oW; pc.frameSize[1] = (float)oH;
  pc.frameSize[2] = 1.0f / (float)oW; pc.frameSize[3] = 1.0f / (float)oH;
  pc.motionSize[0] = (float)hW; pc.motionSize[1] = (float)hH;
  pc.motionSize[2] = 1.0f / (float)hW; pc.motionSize[3] = 1.0f / (float)hH;
  pc.alpha = alpha;
  pc.confPower = std::clamp(m_confPower, 0.25f, 4.0f);
  pc.diffScale = 2.0f;
  pc.useBidirectional = 0;
  pc.useOcclusion = (m_qualityMode >= 1) ? 1 : 0;
  pc.qualityMode = m_qualityMode;
  pc.padding = 0;
  vkCmdPushConstants(cmd, m_vkInterpolateLayout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, 60, &pc);

  // Dispatch
  vkCmdDispatch(cmd, (oW + 15) / 16, (oH + 15) / 16, 1);

  // Barrier: output writes complete before D3D11 reads
  {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_sharedOutput.vkImage;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  // Submit and wait
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si = {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, m_vkComputeFence);
  vkWaitForFences(dev, 1, &m_vkComputeFence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &m_vkComputeFence);

  // D3D11: shared output → presentation texture (GPU copy, no CPU)
  m_context->CopyResource(m_outputTexture.Get(), m_sharedOutput.d3dTex.Get());

  return true;
}

// -----------------------------------------------------------------------
// VulkanFullDispatch: Complete Vulkan pipeline — D3D11 copies ONLY prev/curr
//   D3D11 copy prev+curr → shared textures → Flush
//   Vulkan: downsample(prev) → downsample(curr) → cost_volume → flow_decoder → interpolate
//   D3D11 copy output back for presentation
//   Motion estimation + interpolation entirely in Vulkan.
// -----------------------------------------------------------------------
bool Interpolator::VulkanFullDispatch(
    ID3D11ShaderResourceView* prev,
    ID3D11ShaderResourceView* curr,
    float alpha) {
  if (!m_vkResCreated || !m_renderDevice || !m_vkFullPipeline) return false;
  if (!m_vkZeroCopy) return false;
  if (m_vkDownsamplePipeline == VK_NULL_HANDLE ||
      m_vkCostVolumePipeline == VK_NULL_HANDLE ||
      m_vkFlowDecoderPipeline == VK_NULL_HANDLE ||
      m_vkInterpolatePipeline == VK_NULL_HANDLE) return false;

  VkDevice dev = m_renderDevice->GetVkDevice();
  VkQueue queue = m_renderDevice->GetVkComputeQueue();
  VkCommandBuffer cmd = m_renderDevice->GetVkCommandBuffer();
  uint32_t oW = (uint32_t)m_outputWidth, oH = (uint32_t)m_outputHeight;
  uint32_t iW = (uint32_t)m_inputWidth, iH = (uint32_t)m_inputHeight;
  uint32_t hW = (uint32_t)m_lumaWidth, hH = (uint32_t)m_lumaHeight;

  static bool loggedOnce = false;
  if (!loggedOnce) {
    std::ofstream log("vulkan_debug.txt", std::ios::app);
    log << "VulkanFullDispatch: ENTER input=" << iW << "x" << iH
        << " output=" << oW << "x" << oH << " luma=" << hW << "x" << hH << "\n";
    log.flush();
    loggedOnce = true;
  }

  // ---- D3D11: copy ONLY prev + curr into shared textures ----
  auto d3dCopyFromSrv = [&](ID3D11ShaderResourceView* srv, ID3D11Texture2D* dst) {
    if (!srv || !dst) return;
    Microsoft::WRL::ComPtr<ID3D11Resource> res;
    srv->GetResource(&res);
    m_context->CopyResource(dst, res.Get());
  };
  d3dCopyFromSrv(prev, m_sharedPrev.d3dTex.Get());
  d3dCopyFromSrv(curr, m_sharedCurr.d3dTex.Get());
  m_context->Flush();

  // ---- Vulkan: single command buffer with all stages ----
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo cbi = {};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &cbi);

  // === Stage 0: Initial barriers ===
  // Inputs: sharedPrev, sharedCurr → SHADER_READ_ONLY
  // Intermediates: featPrev, featCurr, costVol, flowOut, confOut, dummy → GENERAL
  // Output: sharedOutput → GENERAL
  {
    VkImageMemoryBarrier bars[10] = {};
    int n = 0;
    auto addBar = [&](VkImage img, VkImageLayout newLay, VkAccessFlags access) {
      bars[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bars[n].srcQueueFamilyIndex = bars[n].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bars[n].image = img;
      bars[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      bars[n].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      bars[n].newLayout = newLay;
      bars[n].dstAccessMask = access;
      n++;
    };
    addBar(m_sharedPrev.vkImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_sharedCurr.vkImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_SHADER_READ_BIT);
    addBar(m_vkFeatPrev.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_vkFeatCurr.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_vkCostVol.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_vkFlowOut.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_vkConfOut.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_vkDummy.image, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    addBar(m_sharedOutput.vkImage, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, n, bars);
  }

  // === Stage 1: Downsample prev → featPrev ===
  {
    struct {
      float inputSize[4];
      float outputSize0[4];
      float outputSize1[4];
      float outputSize2[4];
      float outputSize3[4];
      int numLevels;
      int padding;
    } pc;
    pc.inputSize[0] = (float)iW; pc.inputSize[1] = (float)iH;
    pc.inputSize[2] = 1.0f / (float)iW; pc.inputSize[3] = 1.0f / (float)iH;
    pc.outputSize0[0] = (float)hW; pc.outputSize0[1] = (float)hH;
    pc.outputSize0[2] = 1.0f / (float)hW; pc.outputSize0[3] = 1.0f / (float)hH;
    // levels 1-3 unused, set to 1×1
    for (int i = 0; i < 4; i++) {
      pc.outputSize1[i] = (i < 2) ? 1.0f : 1.0f;
      pc.outputSize2[i] = (i < 2) ? 1.0f : 1.0f;
      pc.outputSize3[i] = (i < 2) ? 1.0f : 1.0f;
    }
    pc.numLevels = 1;
    pc.padding = 0;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkDownsamplePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_vkDownsampleLayout, 0, 1, &m_vkDownsamplePrevSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_vkDownsampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 88, &pc);
    vkCmdDispatch(cmd, (hW + 15) / 16, (hH + 15) / 16, 1); // z=1 → only level 0
  }

  // === Stage 2: Downsample curr → featCurr ===
  {
    struct {
      float inputSize[4]; float outputSize0[4]; float outputSize1[4];
      float outputSize2[4]; float outputSize3[4]; int numLevels; int padding;
    } pc;
    pc.inputSize[0] = (float)iW; pc.inputSize[1] = (float)iH;
    pc.inputSize[2] = 1.0f / (float)iW; pc.inputSize[3] = 1.0f / (float)iH;
    pc.outputSize0[0] = (float)hW; pc.outputSize0[1] = (float)hH;
    pc.outputSize0[2] = 1.0f / (float)hW; pc.outputSize0[3] = 1.0f / (float)hH;
    for (int i = 0; i < 4; i++) {
      pc.outputSize1[i] = (i < 2) ? 1.0f : 1.0f;
      pc.outputSize2[i] = (i < 2) ? 1.0f : 1.0f;
      pc.outputSize3[i] = (i < 2) ? 1.0f : 1.0f;
    }
    pc.numLevels = 1; pc.padding = 0;

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_vkDownsampleLayout, 0, 1, &m_vkDownsampleCurrSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_vkDownsampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 88, &pc);
    vkCmdDispatch(cmd, (hW + 15) / 16, (hH + 15) / 16, 1);
  }

  // Barrier: featPrev + featCurr GENERAL → SHADER_READ_ONLY (downsample writes must complete)
  {
    VkImageMemoryBarrier bars[3] = {};
    for (int i = 0; i < 3; i++) {
      bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bars[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bars[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bars[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bars[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    bars[0].image = m_vkFeatPrev.image;
    bars[0].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[1].image = m_vkFeatCurr.image;
    bars[1].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bars[2].image = m_vkDummy.image;
    bars[2].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 3, bars);
  }

  // === Stage 3: Cost Volume ===
  {
    struct {
      float srcSize[4];
      float costVolumeSize[4];
      int level;
      int searchRange;
      int isBackward;
      int usePrevFlow;
      float flowScale;
      float padding;
    } pc;
    pc.srcSize[0] = (float)iW; pc.srcSize[1] = (float)iH;
    pc.srcSize[2] = 1.0f / (float)iW; pc.srcSize[3] = 1.0f / (float)iH;
    pc.costVolumeSize[0] = (float)hW; pc.costVolumeSize[1] = (float)hH;
    pc.costVolumeSize[2] = 1.0f / (float)hW; pc.costVolumeSize[3] = 1.0f / (float)hH;
    pc.level = 0;
    pc.searchRange = 4;
    pc.isBackward = 0;
    pc.usePrevFlow = 0;
    pc.flowScale = 1.0f;
    pc.padding = 0.0f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkCostVolumePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_vkCostVolumeLayout, 0, 1, &m_vkCostVolumeFullSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_vkCostVolumeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 56, &pc);
    vkCmdDispatch(cmd, (hW + 15) / 16, (hH + 15) / 16, 1);
  }

  // Barrier: costVol GENERAL → SHADER_READ_ONLY
  {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_vkCostVol.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  // === Stage 4: Flow Decoder → motion + confidence ===
  {
    struct {
      float costVolumeSize[4];
      float srcSize[4];
      int level;
      int searchRange;
      int isLastLevel;
      int iteration;
      float flowScale;
      float confidenceScale;
    } pc;
    pc.costVolumeSize[0] = (float)hW; pc.costVolumeSize[1] = (float)hH;
    pc.costVolumeSize[2] = 1.0f / (float)hW; pc.costVolumeSize[3] = 1.0f / (float)hH;
    pc.srcSize[0] = (float)iW; pc.srcSize[1] = (float)iH;
    pc.srcSize[2] = 1.0f / (float)iW; pc.srcSize[3] = 1.0f / (float)iH;
    pc.level = 0;
    pc.searchRange = 4;
    pc.isLastLevel = 1;
    pc.iteration = 0;
    pc.flowScale = 1.0f;
    pc.confidenceScale = 1.0f;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkFlowDecoderPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_vkFlowDecoderLayout, 0, 1, &m_vkFlowDecoderFullSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_vkFlowDecoderLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 56, &pc);
    vkCmdDispatch(cmd, (hW + 15) / 16, (hH + 15) / 16, 1);
  }

  // Barrier: flowOut + confOut GENERAL → SHADER_READ_ONLY
  {
    VkImageMemoryBarrier bars[2] = {};
    for (int i = 0; i < 2; i++) {
      bars[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      bars[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      bars[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      bars[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
      bars[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      bars[i].srcQueueFamilyIndex = bars[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bars[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    bars[0].image = m_vkFlowOut.image;
    bars[1].image = m_vkConfOut.image;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, bars);
  }

  // === Stage 5: Interpolate ===
  {
    struct {
      float frameSize[4];
      float motionSize[4];
      float alpha;
      float confPower;
      float diffScale;
      int useBidirectional;
      int useOcclusion;
      int qualityMode;
      int padding;
    } pc;
    pc.frameSize[0] = (float)oW; pc.frameSize[1] = (float)oH;
    pc.frameSize[2] = 1.0f / (float)oW; pc.frameSize[3] = 1.0f / (float)oH;
    pc.motionSize[0] = (float)hW; pc.motionSize[1] = (float)hH;
    pc.motionSize[2] = 1.0f / (float)hW; pc.motionSize[3] = 1.0f / (float)hH;
    pc.alpha = alpha;
    pc.confPower = std::clamp(m_confPower, 0.25f, 4.0f);
    pc.diffScale = 2.0f;
    pc.useBidirectional = 0;
    pc.useOcclusion = (m_qualityMode >= 1) ? 1 : 0;
    pc.qualityMode = m_qualityMode;
    pc.padding = 0;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkInterpolatePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_vkInterpolateLayout, 0, 1, &m_vkInterpolateFullSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_vkInterpolateLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, 60, &pc);
    vkCmdDispatch(cmd, (oW + 15) / 16, (oH + 15) / 16, 1);
  }

  // Barrier: output writes complete
  {
    VkImageMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_sharedOutput.vkImage;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
  }

  // Submit and wait
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si = {};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, m_vkComputeFence);
  vkWaitForFences(dev, 1, &m_vkComputeFence, VK_TRUE, UINT64_MAX);
  vkResetFences(dev, 1, &m_vkComputeFence);

  // D3D11: shared output → presentation texture
  m_context->CopyResource(m_outputTexture.Get(), m_sharedOutput.d3dTex.Get());

  return true;
}

#endif // USE_VULKAN