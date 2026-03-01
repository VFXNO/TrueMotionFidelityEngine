// ============================================================================
// Interpolator v3 - PWC-Net Motion Estimation + Vulkan/GPU Abstraction
//
// Features:
//   - PWC-Net style optical flow with pyramidal feature extraction
//   - Cost volume computation for accurate motion estimation
//   - Bidirectional flow with occlusion detection
//   - Confidence-based blending for artifact reduction
//   - Quality modes: Performance, Balanced, Quality
// ============================================================================

#pragma once

#include "render_device.h"

#include <vector>
#include <memory>
#include <string>

namespace tfe {

// ============================================================================
// PWC-Net Configuration
// ============================================================================

struct PWCNetConfig {
    // Pyramid levels: 0=full, 1=half, 2=quarter, 3=eighth
    int pyramidLevels = 4;
    
    // Search range at each level (in pixels at that level's resolution)
    int searchRangeTiny = 16;    // Level 3 (1/8 resolution)
    int searchRangeSmall = 12;   // Level 2 (1/4 resolution)
    int searchRangeFull = 8;     // Level 1 (1/2 resolution)
    
    // Refinement iterations
    int refineIterations = 3;
    
    // Quality presets
    enum class Quality {
        Performance,  // Fast, lower quality
        Balanced,     // Balance speed and quality
        Quality       // Maximum quality, slower
    };
    Quality quality = Quality::Balanced;
    
    // Enable bidirectional flow for better occlusion handling
    bool useBidirectional = true;
    
    // Enable occlusion detection
    bool useOcclusion = true;
    
    // Confidence threshold for blending
    float confidenceThreshold = 0.5f;
};

// ============================================================================
// Interpolator Class (v3 - PWC-Net with Vulkan support)
// ============================================================================

// Rename to avoid conflict with old Interpolator
class InterpolatorV3 {
    // Pyramid levels: 0=full, 1=half, 2=quarter, 3=eighth
    int pyramidLevels = 4;
    
    // Search range at each level (in pixels at that level's resolution)
    int searchRangeTiny = 16;    // Level 3 (1/8 resolution)
    int searchRangeSmall = 12;   // Level 2 (1/4 resolution)
    int searchRangeFull = 8;     // Level 1 (1/2 resolution)
    
    // Refinement iterations
    int refineIterations = 3;
    
    // Quality presets
    enum class Quality {
        Performance,  // Fast, lower quality
        Balanced,     // Balance speed and quality
        Quality       // Maximum quality, slower
    };
    Quality quality = Quality::Balanced;
    
    // Enable bidirectional flow for better occlusion handling
    bool useBidirectional = true;
    
    // Enable occlusion detection
    bool useOcclusion = true;
    
    // Confidence threshold for blending
    float confidenceThreshold = 0.5f;
};

// ============================================================================
// Interpolator Class
// ============================================================================

class Interpolator {
public:
    Interpolator();
    ~Interpolator();

    // Initialization
    bool Initialize(RenderDevice* device);
    bool Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight);
    void Shutdown();

    // Configuration
    void SetQuality(PWCNetConfig::Quality quality);
    void SetMotionSmoothing(float edgeScale, float confPower);
    void SetConfidenceThreshold(float threshold);

    // Execute frame interpolation
    void Execute(
        void* prevFrame,   // Texture or ID3D11Texture2D*
        void* currFrame,   // Texture or ID3D11Texture2D*
        float alpha);      // 0.0 to 1.0 interpolation factor

    // Re-use cached motion field for different alpha
    void InterpolateOnly(void* prevFrame, void* currFrame, float alpha);

    // Debug visualization
    void Debug(int mode, float scale);

    // Output
    void* GetOutputTexture() const;
    void* GetOutputSRV() const;

    // Performance metrics
    struct Stats {
        float motionEstimationTime = 0;
        float interpolationTime = 0;
        float totalTime = 0;
    };
    const Stats& GetStats() const { return m_stats; }

private:
    // Pipeline stages
    bool CreatePipelines();
    bool CreateResources();
    bool LoadShaders();
    void ShutdownResources();
    
    void ComputeFeaturePyramid(void* frame, int frameIndex);
    void ComputeMotionField();
    void ComputeBidirectionalMotion();
    void RefineMotion();
    void InterpolateFrame(float alpha);

    // Helper methods
    uint32_t DivUp(uint32_t size, uint32_t blockSize) {
        return (size + blockSize - 1) / blockSize;
    }

    // Device
    RenderDevice* m_device = nullptr;
    bool m_initialized = false;

    // Configuration
    PWCNetConfig m_config;
    float m_smoothEdgeScale = 6.0f;
    float m_smoothConfPower = 1.0f;

    // Dimensions
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

    // Textures - Input frames
    std::unique_ptr<Texture> m_prevFrame;
    std::unique_ptr<Texture> m_currFrame;

    // Textures - Feature pyramids (both frames)
    std::unique_ptr<Texture> m_prevFeatures[4];  // Levels 0-3
    std::unique_ptr<Texture> m_currFeatures[4];

    // Textures - Motion fields
    std::unique_ptr<Texture> m_motionFwd;    // Forward motion
    std::unique_ptr<Texture> m_motionBwd;    // Backward motion
    std::unique_ptr<Texture> m_confidenceFwd;
    std::unique_ptr<Texture> m_confidenceBwd;
    std::unique_ptr<Texture> m_motionSmooth;
    std::unique_ptr<Texture> m_confidenceSmooth;

    // Textures - Output
    std::unique_ptr<Texture> m_outputTexture;

    // Buffers - Constant buffers
    std::unique_ptr<Buffer> m_featurePyramidCB;
    std::unique_ptr<Buffer> m_costVolumeCB;
    std::unique_ptr<Buffer> m_flowDecoderCB;
    std::unique_ptr<Buffer> m_interpolateCB;

    // Pipelines (D3D11)
    std::unique_ptr<ComputePipeline> m_featurePyramidPipeline;
    std::unique_ptr<ComputePipeline> m_costVolumePipeline;
    std::unique_ptr<ComputePipeline> m_flowDecoderPipeline;
    std::unique_ptr<ComputePipeline> m_interpolatePipeline;

#ifdef USE_VULKAN
    // Vulkan pipelines and descriptor sets
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
#endif

    // Samplers
    std::unique_ptr<Sampler> m_linearSampler;
    std::unique_ptr<Sampler> m_pointSampler;

    // Stats
    Stats m_stats;

    // Frame tracking
    int m_frameIndex = 0;
    bool m_hasValidMotion = false;
};

// ============================================================================
// Inline Implementation
// ============================================================================

inline Interpolator::Interpolator() {
}

inline Interpolator::~Interpolator() {
    Shutdown();
}

inline void Interpolator::SetQuality(PWCNetConfig::Quality quality) {
    m_config.quality = quality;
    
    switch (quality) {
    case PWCNetConfig::Quality::Performance:
        m_config.pyramidLevels = 3;
        m_config.searchRangeTiny = 12;
        m_config.searchRangeSmall = 8;
        m_config.refineIterations = 2;
        m_config.useBidirectional = true;
        break;
    case PWCNetConfig::Quality::Balanced:
        m_config.pyramidLevels = 4;
        m_config.searchRangeTiny = 16;
        m_config.searchRangeSmall = 12;
        m_config.refineIterations = 3;
        m_config.useBidirectional = true;
        break;
    case PWCNetConfig::Quality::Quality:
        m_config.pyramidLevels = 4;
        m_config.searchRangeTiny = 24;
        m_config.searchRangeSmall = 16;
        m_config.searchRangeFull = 10;
        m_config.refineIterations = 4;
        m_config.useBidirectional = true;
        m_config.useOcclusion = true;
        break;
    }
}

inline void Interpolator::SetMotionSmoothing(float edgeScale, float confPower) {
    m_smoothEdgeScale = edgeScale;
    m_smoothConfPower = confPower;
}

inline void Interpolator::SetConfidenceThreshold(float threshold) {
    m_config.confidenceThreshold = threshold;
}

} // namespace tfe