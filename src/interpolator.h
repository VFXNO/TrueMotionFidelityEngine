#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <string>

#ifdef USE_VULKAN
#include "render_device.h"
#endif

// ============================================================================
// Interpolator v2 - Rewritten motion estimation & interpolation pipeline
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

#ifdef USE_VULKAN
  // Set the Vulkan render device for compute pipeline (call before Initialize)
  void SetRenderDevice(tfe::RenderDevice* rd) { m_renderDevice = rd; }
#endif

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

  // --- Backend info ---
  bool IsVulkan() const { return m_useVulkan; }
  const char* GetBackendName() const { return m_useVulkan ? "Vulkan" : "D3D11"; }

  // --- Weight export/import ---
  bool LoadAttentionWeights(const wchar_t* path);
  bool SaveAttentionWeights(const wchar_t* path) const;
  bool ExportTrainedWeights(const wchar_t* path);  // Export EMA-trained weights from GPU
  void SetUseCustomWeights(bool use) { m_useCustomWeights = use; }
  bool GetUseCustomWeights() const { return m_useCustomWeights; }

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

#ifdef USE_VULKAN
  bool LoadVulkanShaders();
#endif

  // Backend state
  bool m_useVulkan = false;

#ifdef USE_VULKAN
  // Vulkan resources (if available)
  tfe::RenderDevice* m_renderDevice = nullptr;
  
  // Vulkan pipelines
  VkPipeline m_vkFeaturePyramidPipeline = VK_NULL_HANDLE;
  VkPipeline m_vkCostVolumePipeline = VK_NULL_HANDLE;
  VkPipeline m_vkFlowDecoderPipeline = VK_NULL_HANDLE;
  VkPipeline m_vkInterpolatePipeline = VK_NULL_HANDLE;
  VkPipeline m_vkDownsamplePipeline = VK_NULL_HANDLE;
  
  VkPipelineLayout m_vkFeaturePyramidLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_vkCostVolumeLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_vkFlowDecoderLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_vkInterpolateLayout = VK_NULL_HANDLE;
  VkPipelineLayout m_vkDownsampleLayout = VK_NULL_HANDLE;
  
  VkDescriptorSet m_vkFeaturePyramidSet = VK_NULL_HANDLE;
  VkDescriptorSet m_vkCostVolumeSet = VK_NULL_HANDLE;
  VkDescriptorSet m_vkFlowDecoderSet = VK_NULL_HANDLE;
  VkDescriptorSet m_vkInterpolateSet = VK_NULL_HANDLE;
  VkDescriptorSet m_vkDownsampleSet = VK_NULL_HANDLE;

  // Descriptor set layouts (needed for descriptor set allocation)
  VkDescriptorSetLayout m_vkFeaturePyramidSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_vkCostVolumeSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_vkFlowDecoderSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_vkInterpolateSetLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_vkDownsampleSetLayout = VK_NULL_HANDLE;

  // Vulkan image wrapper for compute dispatch
  struct VkImg {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  };

  // D3D11<->Vulkan zero-copy shared texture (same GPU memory)
  struct SharedImg {
    // D3D11 side
    Microsoft::WRL::ComPtr<ID3D11Texture2D> d3dTex;
    HANDLE ntHandle = nullptr;
    // Vulkan side (imported from D3D11 shared handle)
    VkImage vkImage = VK_NULL_HANDLE;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    VkImageView vkView = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
  };

  // Shared textures for zero-copy D3D11<->Vulkan
  SharedImg m_sharedPrev, m_sharedCurr;         // Input frame copies (BGRA8)
  SharedImg m_sharedMotion;                      // Motion field (RG16F)
  SharedImg m_sharedConf;                        // Confidence (R16F)
  SharedImg m_sharedFeatPrev, m_sharedFeatCurr;  // Features (RGBA16F)
  SharedImg m_sharedOutput;                      // Output (B8G8R8A8)

  // Non-shared output VkImage (GENERAL layout for storage writes) — not needed in zero-copy
  VkImg m_vkImgOutput;                     // Output (RGBA8, write-only)

  // Vulkan-only intermediate images for full Vulkan pipeline
  VkImg m_vkFeatPrev;    // Half-res features for prev frame (RGBA16F, hW×hH)
  VkImg m_vkFeatCurr;    // Half-res features for curr frame (RGBA16F, hW×hH)
  VkImg m_vkCostVol;     // Cost volume output (RGBA16F, hW×hH)
  VkImg m_vkFlowOut;     // Motion vectors output (RG16F, hW×hH)
  VkImg m_vkConfOut;     // Confidence output (R16F, hW×hH)
  VkImg m_vkDummy;       // 1×1 dummy for unused descriptor bindings

  // Descriptor sets for full Vulkan pipeline stages
  VkDescriptorSet m_vkDownsamplePrevSet  = VK_NULL_HANDLE;
  VkDescriptorSet m_vkDownsampleCurrSet  = VK_NULL_HANDLE;
  VkDescriptorSet m_vkCostVolumeFullSet  = VK_NULL_HANDLE;
  VkDescriptorSet m_vkFlowDecoderFullSet = VK_NULL_HANDLE;
  VkDescriptorSet m_vkInterpolateFullSet = VK_NULL_HANDLE;

  // Vulkan samplers and sync
  VkSampler m_vkLinearSampler = VK_NULL_HANDLE;
  VkSampler m_vkPointSampler = VK_NULL_HANDLE;
  VkFence m_vkComputeFence = VK_NULL_HANDLE;
  bool m_vkResCreated = false;
  bool m_vkZeroCopy = false;  // True if external memory import succeeded
  bool m_vkFullPipeline = false;  // True if full VK pipeline intermediates ready

  // Vulkan compute dispatch helpers
  bool CreateVkImg(VkImg& img, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage);
  void DestroyVkImg(VkImg& img);
  bool CreateSharedImg(SharedImg& s, uint32_t w, uint32_t h, DXGI_FORMAT d3dFmt,
                       VkFormat vkFmt, VkImageUsageFlags vkUsage);
  void DestroySharedImg(SharedImg& s);
  void CreateVulkanResources();
  void DestroyVulkanResources();
  bool VulkanDispatchInterpolate(ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr, float alpha);
  bool VulkanReWarp(float alpha);  // Fast re-warp: same data, new alpha only
  bool VulkanFullDispatch(ID3D11ShaderResourceView* prev,
      ID3D11ShaderResourceView* curr, float alpha);
#endif

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

  // Temporal attention priors (dynamic online adaptation)
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnSmall1;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnSmall2;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnSmall3;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnFull1;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnFull2;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> m_attnFull3;

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
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnSmall1Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnSmall2Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnSmall3Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnFull1Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnFull2Uav;
  Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_attnFull3Uav;

  // Constant buffers
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_motionConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_refineConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_smoothConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_interpConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_debugConstants;
  Microsoft::WRL::ComPtr<ID3D11Buffer> m_attentionWeights;
  Microsoft::WRL::ComPtr<ID3D11SamplerState> m_linearSampler;

  // Configuration state
  bool m_useCustomWeights = false;
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
