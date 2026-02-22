#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <string>

// ============================================================================
// Interpolator v2 - Rewritten motion estimation & interpolation pipeline
//
// Pipeline overview:
//   1. Downsample: Full-color -> half-res luma -> quarter-res luma -> eighth-res luma
//   2. Motion estimation at tiny (1/8) resolution using hexagonal search + ZNCC
//   3. Backward motion at tiny for consistency checking
//   4. (Full pipeline only) Refine motion at quarter and half resolution via
//      Lucas-Kanade gradient descent
//   5. (Full pipeline only) Joint bilateral spatial smoothing
//   6. (Full pipeline only) Motion-compensated temporal accumulation with
//      AABB neighborhood clamping
//   7. Bidirectional pure motion-compensated warp interpolation
// ============================================================================
class Interpolator {
public:
  enum class DebugViewMode {
    None = 0,
    MotionFlow = 1,
    ConfidenceHeatmap = 2,
    MotionNeedles = 3,
    ResidualError = 4,
    SplitScreen = 5,
    Occlusion = 6,
    AiDisocclusion = 7,
    StructureGradient = 8
  };

  bool Initialize(ID3D11Device* device, ID3D11DeviceContext* context);
  bool Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight);

  // --- Configuration ---
  void SetMotionModel(int model) { m_motionModel = model; }
  void SetMotionSmoothing(float edgeScale, float confPower) {
    m_smoothEdgeScale = edgeScale;
    m_smoothConfPower = confPower;
  }
  void SetQualityMode(int qualityMode) { m_qualityMode = qualityMode; }
  void SetMinimalMotionPipeline(bool enabled) { m_useMinimalMotionPipeline = enabled; }

  // --- Execution ---
  void Execute(
      ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr,
      float alpha,
      ID3D11ShaderResourceView* prevDepth = nullptr,
      ID3D11ShaderResourceView* currDepth = nullptr);
  // Re-warp with new alpha using cached motion field (skips motion estimation)
  void InterpolateOnly(
      ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr,
      float alpha);
  void Blit(ID3D11ShaderResourceView* src);
  void Debug(
      ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr,
      DebugViewMode mode,
      float motionScale,
      float diffScale);

  // --- Output ---
  ID3D11Texture2D* OutputTexture() const { return m_outputTexture.Get(); }
  ID3D11ShaderResourceView* OutputSrv() const { return m_outputSrv.Get(); }

private:
  bool LoadShaders();
  void CreateResources();
  bool ComputeMotion(
      ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr);
  std::wstring ShaderPath(const wchar_t* filename) const;

  // Helpers to dispatch and clear CS state
  void Dispatch(int w, int h);
  void ClearCS(int srvCount, int uavCount);

  Microsoft::WRL::ComPtr<ID3D11Device> m_device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;

  // Compute shaders
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_downsampleCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_downsampleLumaCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionRefineCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionSmoothCs;

  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_interpolateCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_copyCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_debugCs;

  // Luma pyramid
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevLuma;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currLuma;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevLumaSmall;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currLumaSmall;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevLumaTiny;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currLumaTiny;

  // Feature2 pyramid (Channels 5-8)
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature2;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature2;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature2Small;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature2Small;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature2Tiny;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature2Tiny;

  // Feature3 pyramid (Channels 9-12)
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature3;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature3;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature3Small;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature3Small;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevFeature3Tiny;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currFeature3Tiny;

  // Motion fields
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motion;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidence;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionCoarse;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionTiny;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionTinyBackward;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceTiny;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceTinyBackward;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceCoarse;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionSmooth;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceSmooth;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_outputTexture;

  // SRV / UAV views (luma pyramid)
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevLumaSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currLumaSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevLumaSmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currLumaSmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevLumaTinySrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currLumaTinySrv;

  // SRV / UAV views (Feature2 pyramid)
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature2Srv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature2Srv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature2SmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature2SmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature2TinySrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature2TinySrv;

  // SRV / UAV views (Feature3 pyramid)
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature3Srv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature3Srv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature3SmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature3SmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevFeature3TinySrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currFeature3TinySrv;

  // SRV / UAV views (motion)
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionCoarseSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionTinySrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionTinyBackwardSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceTinySrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceTinyBackwardSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceCoarseSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionSmoothSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceSmoothSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_outputSrv;

  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevLumaUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currLumaUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevLumaSmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currLumaSmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevLumaTinyUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currLumaTinyUav;

  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature2Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature2Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature2SmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature2SmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature2TinyUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature2TinyUav;

  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature3Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature3Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature3SmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature3SmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevFeature3TinyUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currFeature3TinyUav;

  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionCoarseUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionTinyUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionTinyBackwardUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceTinyUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceTinyBackwardUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceCoarseUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionSmoothUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceSmoothUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_outputUav;

  // Constant buffers
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_motionConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_refineConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_smoothConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_interpConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_debugConstants;
  Microsoft::WRL::ComPtr<ID3D11SamplerState> m_linearSampler;

  // Configuration state
  int m_inputWidth = 0;
  int m_inputHeight = 0;
  int m_outputWidth = 0;
  int m_outputHeight = 0;
  int m_lumaWidth = 0;
  int m_lumaHeight = 0;
  int m_smallWidth = 0;
  int m_smallHeight = 0;
  int m_tinyWidth = 0;
  int m_tinyHeight = 0;
  int m_qualityMode = 0;
  int m_motionModel = 1;
  float m_confPower = 1.0f;
  float m_smoothEdgeScale = 6.0f;
  float m_smoothConfPower = 1.0f;
  bool m_useMinimalMotionPipeline = true;
};
