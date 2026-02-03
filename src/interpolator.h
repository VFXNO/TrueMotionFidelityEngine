#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <string>

class Interpolator {
public:
  enum class DebugViewMode {
    None = 0,
    MotionFlow = 1,
    ConfidenceHeatmap = 2,
    MotionNeedles = 3,
    ResidualError = 4,
    SplitScreen = 5,
    Occlusion = 6
  };

  bool Initialize(ID3D11Device* device, ID3D11DeviceContext* context);
  bool Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight);
  void SetRadius(int radius) { m_radius = radius; }
  void SetMotionSmoothing(float edgeScale, float confPower) {
    m_smoothEdgeScale = edgeScale;
    m_smoothConfPower = confPower;
  }
  void SetRefineRadius(int radius) { m_refineRadius = radius; }
  void SetTemporalStabilization(bool enabled, float historyWeight, float confInfluence, int neighborhoodSize) {
    m_temporalEnabled = enabled;
    m_temporalHistoryWeight = historyWeight;
    m_temporalConfInfluence = confInfluence;
    m_temporalNeighborhoodSize = neighborhoodSize;
  }
  void SetMotionVectorPrediction(bool enabled) { m_useMotionPrediction = enabled; }
  void ResetTemporal() {
    m_temporalValid = false;
    m_temporalIndex = 0;
    m_historyValid = false;
  }

  void Execute(
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

  ID3D11Texture2D* OutputTexture() const { return m_outputTexture.Get(); }
  ID3D11ShaderResourceView* OutputSrv() const { return m_outputSrv.Get(); }

private:
  bool LoadShaders();
  void CreateResources();
  bool ComputeMotion(
      ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr);
  std::wstring ShaderPath(const wchar_t* filename) const;

  Microsoft::WRL::ComPtr<ID3D11Device> m_device;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_context;

  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_downsampleCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_downsampleLumaCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionRefineCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionSmoothCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_motionTemporalCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_interpolateCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_copyCs;
  Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_debugCs;

  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevLuma;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currLuma;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevLumaSmall;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_currLumaSmall;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motion;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidence;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionCoarse;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_prevMotionCoarse;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceCoarse;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionSmooth;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceSmooth;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_motionTemporal[2];
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_confidenceTemporal[2];
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_outputTexture;

  static const int kHistorySize = 4;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_historyLuma[kHistorySize];
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_historyColor[kHistorySize];
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_historyLumaSrv[kHistorySize];
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_historyColorSrv[kHistorySize];
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_historyLumaUav[kHistorySize];
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_historyColorUav[kHistorySize];
  int m_historyIndex = 0;
  bool m_historyValid = false;

  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevLumaSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currLumaSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevLumaSmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_currLumaSmallSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionCoarseSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_prevMotionCoarseSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceCoarseSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionSmoothSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceSmoothSrv;
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_motionTemporalSrv[2];
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_confidenceTemporalSrv[2];
  Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_outputSrv;

  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevLumaUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currLumaUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevLumaSmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_currLumaSmallUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionCoarseUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_prevMotionCoarseUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceCoarseUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionSmoothUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceSmoothUav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_motionTemporalUav[2];
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_confidenceTemporalUav[2];
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_outputUav;

  Microsoft::WRL::ComPtr<ID3D11Buffer> m_motionConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_refineConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_smoothConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_temporalConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_interpConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_debugConstants;
  Microsoft::WRL::ComPtr<ID3D11SamplerState> m_linearSampler;

  bool m_useMotionPrediction = false;
  bool m_prevMotionCoarseValid = false;

  int m_inputWidth = 0;
  int m_inputHeight = 0;
  int m_outputWidth = 0;
  int m_outputHeight = 0;
  int m_lumaWidth = 0;
  int m_lumaHeight = 0;
  int m_smallWidth = 0;
  int m_smallHeight = 0;
  int m_radius = 3;
  float m_confPower = 1.0f;
  float m_smoothEdgeScale = 6.0f;
  float m_smoothConfPower = 1.0f;
  int m_refineRadius = 2;
  bool m_temporalEnabled = true;
  bool m_temporalValid = false;
  int m_temporalIndex = 0;
  float m_temporalHistoryWeight = 0.2f;
  float m_temporalConfInfluence = 0.6f;
  int m_temporalNeighborhoodSize = 2;
};
