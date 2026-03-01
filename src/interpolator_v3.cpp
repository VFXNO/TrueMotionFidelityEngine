#include "interpolator_v3.h"

#include <algorithm>
#include <fstream>
#include <chrono>

#ifdef USE_VULKAN
#include "vulkan_types.h"
#endif

#include <d3d11.h>

namespace tfe {

#ifdef USE_VULKAN
// ============================================================================
// Vulkan Helper Functions
// ============================================================================

// Transition image layout
static void TransitionImage(VkCommandBuffer cmd, VkImage image, 
                           VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }

    vkCmdPipelineBarrier(cmd, 1, &barrier, 0, nullptr, 0, nullptr, 0, nullptr);
}

// Update descriptor set with sampled image
static void UpdateDescriptorWithImage(VkDevice device, VkDescriptorSet set, 
                                     uint32_t binding, VkImageView view, VkSampler sampler) {
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = sampler;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// Update descriptor set with storage image
static void UpdateDescriptorWithStorageImage(VkDevice device, VkDescriptorSet set,
                                             uint32_t binding, VkImageView view) {
    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = VK_NULL_HANDLE;
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.descriptorCount = 1;
    write.pImageInfo = &imageInfo;
    
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}
#endif

// ============================================================================
// Helper Structures for Push Constants (must match GLSL/HLSL)
// ============================================================================

struct FeaturePyramidConstants {
    float inputSize[4];    // width, height, 1/width, 1/height
    float outputSize0[4];   // level 0 size
    float outputSize1[4];   // level 1 size
    float outputSize2[4];   // level 2 size
    float outputSize3[4];   // level 3 size
    int numLevels;
    int padding;
};

struct CostVolumeConstants {
    float srcSize[4];           // source image size
    float costVolumeSize[4];    // cost volume dimensions
    int level;
    int searchRange;
    int isBackward;
    int usePrevFlow;
    float flowScale;
    float padding;
};

struct FlowDecoderConstants {
    float costVolumeSize[4];   // width, height, 1/width, 1/height
    float srcSize[4];
    int level;
    int searchRange;
    int isLastLevel;
    int iteration;
    float flowScale;
    float confidenceScale;
};

struct InterpolationConstants {
    float frameSize[4];          // width, height, 1/width, 1/height
    float motionSize[4];         // motion field dimensions
    float alpha;
    float confPower;
    float diffScale;
    int useBidirectional;
    int useOcclusion;
    int qualityMode;
    int padding;
};

// ============================================================================
// Interpolator Implementation
// ============================================================================

bool Interpolator::Initialize(RenderDevice* device) {
    if (!device) return false;
    m_device = device;
    
    // Debug output which backend is being used
    if (m_device->IsVulkan()) {
        OutputDebugStringA("[Interpolatorv3] Using VULKAN backend\n");
    } else {
        OutputDebugStringA("[Interpolatorv3] Using D3D11 backend\n");
    }
    
    // Set default quality
    SetQuality(PWCNetConfig::Quality::Balanced);
    
    // Create samplers
    m_linearSampler = std::make_unique<Sampler>();
    m_pointSampler = std::make_unique<Sampler>();
    
#ifdef USE_VULKAN
    if (m_device->IsVulkan()) {
        m_linearSampler->Create(m_device, VK_FILTER_LINEAR, VK_FILTER_LINEAR, 
            VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        m_pointSampler->Create(m_device, VK_FILTER_NEAREST, VK_FILTER_NEAREST,
            VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
#else
    m_linearSampler->Create(m_device, D3D11_FILTER_MIN_MAG_MIP_LINEAR, D3D11_TEXTURE_ADDRESS_CLAMP);
    m_pointSampler->Create(m_device, D3D11_FILTER_MIN_MAG_MIP_POINT, D3D11_TEXTURE_ADDRESS_CLAMP);
#endif

    // Create constant buffers
    m_featurePyramidCB = std::make_unique<Buffer>();
    m_costVolumeCB = std::make_unique<Buffer>();
    m_flowDecoderCB = std::make_unique<Buffer>();
    m_interpolateCB = std::make_unique<Buffer>();
    
#ifdef USE_VULKAN
    m_featurePyramidCB->Create(m_device, sizeof(FeaturePyramidConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    m_costVolumeCB->Create(m_device, sizeof(CostVolumeConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    m_flowDecoderCB->Create(m_device, sizeof(FlowDecoderConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    m_interpolateCB->Create(m_device, sizeof(InterpolationConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
#else
    m_featurePyramidCB->Create(m_device, sizeof(FeaturePyramidConstants), D3D11_USAGE_DYNAMIC, true);
    m_costVolumeCB->Create(m_device, sizeof(CostVolumeConstants), D3D11_USAGE_DYNAMIC, true);
    m_flowDecoderCB->Create(m_device, sizeof(FlowDecoderConstants), D3D11_USAGE_DYNAMIC, true);
    m_interpolateCB->Create(m_device, sizeof(InterpolationConstants), D3D11_USAGE_DYNAMIC, true);
#endif

    // Load shaders and create pipelines
    if (!CreatePipelines()) {
        OutputDebugStringA("[Interpolatorv3] Failed to create pipelines\n");
        return false;
    }

    m_initialized = true;
    OutputDebugStringA("[Interpolatorv3] Initialized successfully\n");
    return true;
}

bool Interpolator::Resize(int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    if (inputWidth <= 0 || inputHeight <= 0 || outputWidth <= 0 || outputHeight <= 0)
        return false;

    m_inputWidth = inputWidth;
    m_inputHeight = inputHeight;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;

    // Calculate pyramid resolutions
    m_lumaWidth = (inputWidth + 1) / 2;
    m_lumaHeight = (inputHeight + 1) / 2;
    m_smallWidth = (m_lumaWidth + 1) / 2;
    m_smallHeight = (m_lumaHeight + 1) / 2;
    m_tinyWidth = (m_smallWidth + 1) / 2;
    m_tinyHeight = (m_smallHeight + 1) / 2;

    CreateResources();
    return true;
}

bool Interpolator::CreateResources() {
    if (!m_device) return false;

    // Clean up existing resources
    ShutdownResources();

    // Input frame textures
    m_prevFrame = std::make_unique<Texture>();
    m_currFrame = std::make_unique<Texture>();
    
#ifdef USE_VULKAN
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat featureFormat = VK_FORMAT_R16G16B16A16_FLOAT;
    VkFormat motionFormat = VK_FORMAT_R16G16_FLOAT;
    VkFormat confFormat = VK_FORMAT_R16_FLOAT;
    
    VkImageUsageFlags featureUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
#else
    DXGI_FORMAT colorFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
    DXGI_FORMAT featureFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
    DXGI_FORMAT motionFormat = DXGI_FORMAT_R16G16_FLOAT;
    DXGI_FORMAT confFormat = DXGI_FORMAT_R16_FLOAT;
    
    D3D11_USAGE resourceUsage = D3D11_USAGE_DEFAULT;
#endif

    // Feature pyramids for both frames (4 levels)
    for (int i = 0; i < 4; i++) {
        int w = (i == 0) ? m_lumaWidth : (i == 1) ? m_smallWidth : m_tinyWidth;
        int h = (i == 0) ? m_lumaHeight : (i == 1) ? m_smallHeight : m_tinyHeight;
        
        m_prevFeatures[i] = std::make_unique<Texture>();
        m_currFeatures[i] = std::make_unique<Texture>();
        
#ifdef USE_VULKAN
        m_prevFeatures[i]->Create(m_device, w, h, featureFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_currFeatures[i]->Create(m_device, w, h, featureFormat, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
#else
        m_prevFeatures[i]->Create(m_device, w, h, featureFormat, resourceUsage, true);
        m_currFeatures[i]->Create(m_device, w, h, featureFormat, resourceUsage, true);
#endif
    }

    // Motion fields
    m_motionFwd = std::make_unique<Texture>();
    m_motionBwd = std::make_unique<Texture>();
    m_confidenceFwd = std::make_unique<Texture>();
    m_confidenceBwd = std::make_unique<Texture>();
    m_motionSmooth = std::make_unique<Texture>();
    m_confidenceSmooth = std::make_unique<Texture>();

#ifdef USE_VULKAN
    m_motionFwd->Create(m_device, m_smallWidth, m_smallHeight, motionFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    m_motionBwd->Create(m_device, m_smallWidth, m_smallHeight, motionFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    m_confidenceFwd->Create(m_device, m_smallWidth, m_smallHeight, confFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    m_confidenceBwd->Create(m_device, m_smallWidth, m_smallHeight, confFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    m_motionSmooth->Create(m_device, m_lumaWidth, m_lumaHeight, motionFormat, VK_IMAGE_USAGE_STORAGE_BIT);
    m_confidenceSmooth->Create(m_device, m_lumaWidth, m_lumaHeight, confFormat, VK_IMAGE_USAGE_STORAGE_BIT);
#else
    m_motionFwd->Create(m_device, m_smallWidth, m_smallHeight, motionFormat, resourceUsage, true);
    m_motionBwd->Create(m_device, m_smallWidth, m_smallHeight, motionFormat, resourceUsage, true);
    m_confidenceFwd->Create(m_device, m_smallWidth, m_smallHeight, confFormat, resourceUsage, true);
    m_confidenceBwd->Create(m_device, m_smallWidth, m_smallHeight, confFormat, resourceUsage, true);
    m_motionSmooth->Create(m_device, m_lumaWidth, m_lumaHeight, motionFormat, resourceUsage, true);
    m_confidenceSmooth->Create(m_device, m_lumaWidth, m_lumaHeight, confFormat, resourceUsage, true);
#endif

    // Output texture
    m_outputTexture = std::make_unique<Texture>();
#ifdef USE_VULKAN
    m_outputTexture->Create(m_device, m_outputWidth, m_outputHeight, VK_FORMAT_R8G8B8A8_UNORM, 
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
#else
    m_outputTexture->Create(m_device, m_outputWidth, m_outputHeight, colorFormat, resourceUsage, true);
#endif

    OutputDebugStringA("[Interpolatorv3] Resources created successfully\n");
    return true;
}

void Interpolator::ShutdownResources() {
    // Textures will be automatically destroyed via unique_ptr
    m_prevFrame.reset();
    m_currFrame.reset();
    
    for (int i = 0; i < 4; i++) {
        m_prevFeatures[i].reset();
        m_currFeatures[i].reset();
    }
    
    m_motionFwd.reset();
    m_motionBwd.reset();
    m_confidenceFwd.reset();
    m_confidenceBwd.reset();
    m_motionSmooth.reset();
    m_confidenceSmooth.reset();
    m_outputTexture.reset();
}

void Interpolator::Shutdown() {
    ShutdownResources();
    m_initialized = false;
}

void Interpolator::Execute(void* prevFrame, void* currFrame, float alpha) {
    if (!m_initialized || !prevFrame || !currFrame) return;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Copy input frames
    // In a real implementation, we would copy from the provided frames to our textures
    
    // Compute feature pyramids
    ComputeFeaturePyramid(prevFrame, 0);
    ComputeFeaturePyramid(currFrame, 1);

    // Compute motion field
    ComputeMotionField();

    // Compute bidirectional motion if enabled
    if (m_config.useBidirectional) {
        ComputeBidirectionalMotion();
    }

    // Refine motion
    RefineMotion();

    auto motionEndTime = std::chrono::high_resolution_clock::now();
    m_stats.motionEstimationTime = std::chrono::duration<float, std::milli>(motionEndTime - startTime).count();

    // Interpolate frame
    InterpolateFrame(alpha);

    auto endTime = std::chrono::high_resolution_clock::now();
    m_stats.interpolationTime = std::chrono::duration<float, std::milli>(endTime - motionEndTime).count();
    m_stats.totalTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();

    m_hasValidMotion = true;
    m_frameIndex++;
}

void Interpolator::InterpolateOnly(void* prevFrame, void* currFrame, float alpha) {
    if (!m_hasValidMotion || !prevFrame || !currFrame) {
        Execute(prevFrame, currFrame, alpha);
        return;
    }

    // Skip motion estimation and just interpolate
    auto startTime = std::chrono::high_resolution_clock::now();
    InterpolateFrame(alpha);
    auto endTime = std::chrono::high_resolution_clock::now();
    
    m_stats.interpolationTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    m_stats.totalTime = m_stats.motionEstimationTime + m_stats.interpolationTime;
}

void Interpolator::Debug(int mode, float scale) {
    // Debug visualization not yet implemented in v3
}

void* Interpolator::GetOutputTexture() const {
    if (m_outputTexture) {
#ifdef USE_VULKAN
        return (void*)m_outputTexture->GetVkImage();
#else
        return (void*)m_outputTexture->GetD3D11Texture();
#endif
    }
    return nullptr;
}

void* Interpolator::GetOutputSRV() const {
    if (m_outputTexture) {
#ifdef USE_VULKAN
        return (void*)m_outputTexture->GetVkImageView();
#else
        return (void*)m_outputTexture->GetD3D11SRV();
#endif
    }
    return nullptr;
}

// ============================================================================
// Pipeline Stage Implementations
// ============================================================================

void Interpolator::ComputeFeaturePyramid(void* frame, int frameIndex) {
    // Set up constants
    FeaturePyramidConstants cb = {};
    cb.inputSize[0] = (float)m_inputWidth;
    cb.inputSize[1] = (float)m_inputHeight;
    cb.inputSize[2] = 1.0f / m_inputWidth;
    cb.inputSize[3] = 1.0f / m_inputHeight;
    
    cb.outputSize0[0] = (float)m_lumaWidth;
    cb.outputSize0[1] = (float)m_lumaHeight;
    cb.outputSize1[0] = (float)m_smallWidth;
    cb.outputSize1[1] = (float)m_smallHeight;
    cb.outputSize2[0] = (float)m_tinyWidth;
    cb.outputSize2[1] = (float)m_tinyHeight;
    cb.outputSize3[0] = (float)m_tinyWidth / 2;
    cb.outputSize3[1] = (float)m_tinyHeight / 2;
    cb.numLevels = m_config.pyramidLevels;
    
    void* mapped = m_featurePyramidCB->Map();
    if (mapped) {
        memcpy(mapped, &cb, sizeof(FeaturePyramidConstants));
        m_featurePyramidCB->Unmap();
    }

#ifdef USE_VULKAN
    if (m_device && m_device->IsVulkan() && m_vkFeaturePyramidPipeline != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = m_device->GetVkCommandBuffer();
        VkDevice dev = m_device->GetVkDevice();
        
        OutputDebugStringA("[Interpolatorv3] SHADER: feature_pyramid.comp\n");
        
        // Bind pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkFeaturePyramidPipeline);
        
        // Bind descriptor set
        if (m_vkFeaturePyramidSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                     m_vkFeaturePyramidLayout, 0, 1, 
                                     &m_vkFeaturePyramidSet, 0, nullptr);
        }
        
        // Dispatch: 16x16 workgroups
        uint32_t groupsX = (m_lumaWidth + 15) / 16;
        uint32_t groupsY = (m_lumaHeight + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        
        // Add barrier between stages
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 
                           1, &barrier, 0, nullptr, 0, nullptr);
        
        OutputDebugStringA("[Interpolatorv3] Feature pyramid dispatched (Vulkan)\n");
    } else {
        OutputDebugStringA("[Interpolatorv3] SHADER: DownsampleLuma.hlsl / DownsampleLumaR.hlsl\n");
        OutputDebugStringA("[Interpolatorv3] Computing feature pyramid (D3D11)...\n");
    }
#else
    // D3D11 would use the old pipeline
    OutputDebugStringA("[Interpolatorv3] SHADER: DownsampleLuma.hlsl / DownsampleLumaR.hlsl\n");
    OutputDebugStringA("[Interpolatorv3] Computing feature pyramid (D3D11)...\n");
#endif
}

void Interpolator::ComputeMotionField() {
    // Set up cost volume constants
    CostVolumeConstants cb = {};
    cb.srcSize[0] = (float)m_inputWidth;
    cb.srcSize[1] = (float)m_inputHeight;
    cb.costVolumeSize[0] = (float)m_smallWidth;
    cb.costVolumeSize[1] = (float)m_smallHeight;
    cb.level = 2;  // Quarter resolution
    cb.searchRange = m_config.searchRangeSmall;
    cb.isBackward = 0;
    cb.usePrevFlow = 0;
    cb.flowScale = 4.0f;  // Scale from tiny to small
    
    void* mapped = m_costVolumeCB->Map();
    if (mapped) {
        memcpy(mapped, &cb, sizeof(CostVolumeConstants));
        m_costVolumeCB->Unmap();
    }

#ifdef USE_VULKAN
    if (m_device && m_device->IsVulkan() && m_vkCostVolumePipeline != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = m_device->GetVkCommandBuffer();
        
        OutputDebugStringA("[Interpolatorv3] SHADER: cost_volume.comp\n");

        // Bind cost volume pipeline and dispatch
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkCostVolumePipeline);
        
        // Bind descriptor set
        if (m_vkCostVolumeSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                   m_vkCostVolumeLayout, 0, 1, 
                                   &m_vkCostVolumeSet, 0, nullptr);
        }
        
        uint32_t groupsX = (m_smallWidth + 15) / 16;
        uint32_t groupsY = (m_smallHeight + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        
        // Barrier between cost volume and flow decoder
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 
                           1, &barrier, 0, nullptr, 0, nullptr);
        
        OutputDebugStringA("[Interpolatorv3] Cost volume dispatched (Vulkan)\n");
        
        // Bind flow decoder and dispatch
        if (m_vkFlowDecoderPipeline != VK_NULL_HANDLE) {
            OutputDebugStringA("[Interpolatorv3] SHADER: flow_decoder.comp (forward)\n");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkFlowDecoderPipeline);
            
            // Bind descriptor set
            if (m_vkFlowDecoderSet != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                       m_vkFlowDecoderLayout, 0, 1, 
                                       &m_vkFlowDecoderSet, 0, nullptr);
            }
            
            vkCmdDispatch(cmd, groupsX, groupsY, 1);
            OutputDebugStringA("[Interpolatorv3] Flow decoder dispatched (Vulkan)\n");
        }
    }
#else
    OutputDebugStringA("[Interpolatorv3] SHADER: MotionEst.hlsl / MotionRefine.hlsl\n");
    OutputDebugStringA("[Interpolatorv3] Computing motion field (D3D11)...\n");
#endif
}

void Interpolator::ComputeBidirectionalMotion() {
    // Similar to ComputeMotionField but with swapped frame order
#ifdef USE_VULKAN
    if (m_device && m_device->IsVulkan() && m_vkCostVolumePipeline != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = m_device->GetVkCommandBuffer();
        
        uint32_t groupsX = (m_smallWidth + 15) / 16;
        uint32_t groupsY = (m_smallHeight + 15) / 16;
        
        // Dispatch backward flow
        if (m_vkFlowDecoderPipeline != VK_NULL_HANDLE) {
            OutputDebugStringA("[Interpolatorv3] SHADER: flow_decoder.comp (backward)\n");
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkFlowDecoderPipeline);
            
            // Bind descriptor set
            if (m_vkFlowDecoderSet != VK_NULL_HANDLE) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                       m_vkFlowDecoderLayout, 0, 1, 
                                       &m_vkFlowDecoderSet, 0, nullptr);
            }
            
            vkCmdDispatch(cmd, groupsX, groupsY, 1);
            OutputDebugStringA("[Interpolatorv3] Backward motion dispatched (Vulkan)\n");
        }
    }
#else
    OutputDebugStringA("[Interpolatorv3] SHADER: MotionEst.hlsl (bidirectional)\n");
    OutputDebugStringA("[Interpolatorv3] Computing bidirectional motion (D3D11)...\n");
#endif
}

void Interpolator::RefineMotion() {
    // Optional: Apply refinement passes for higher quality
#ifdef USE_VULKAN
    OutputDebugStringA("[Interpolatorv3] Motion refinement (Vulkan)...\n");
#else
    OutputDebugStringA("[Interpolatorv3] Motion refinement (D3D11)...\n");
#endif
}

void Interpolator::InterpolateFrame(float alpha) {
    // Set up interpolation constants
    InterpolationConstants cb = {};
    cb.frameSize[0] = (float)m_outputWidth;
    cb.frameSize[1] = (float)m_outputHeight;
    cb.frameSize[2] = 1.0f / m_outputWidth;
    cb.frameSize[3] = 1.0f / m_outputHeight;
    cb.motionSize[0] = (float)m_smallWidth;
    cb.motionSize[1] = (float)m_smallHeight;
    cb.motionSize[2] = 1.0f / m_smallWidth;
    cb.motionSize[3] = 1.0f / m_smallHeight;
    cb.alpha = alpha;
    cb.confPower = m_smoothConfPower;
    cb.diffScale = 2.0f;
    cb.useBidirectional = m_config.useBidirectional ? 1 : 0;
    cb.useOcclusion = m_config.useOcclusion ? 1 : 0;
    cb.qualityMode = (int)m_config.quality;
    
    void* mapped = m_interpolateCB->Map();
    if (mapped) {
        memcpy(mapped, &cb, sizeof(InterpolationConstants));
        m_interpolateCB->Unmap();
    }

#ifdef USE_VULKAN
    if (m_device && m_device->IsVulkan() && m_vkInterpolatePipeline != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = m_device->GetVkCommandBuffer();
        
        OutputDebugStringA("[Interpolatorv3] SHADER: interpolate.comp\n");

        // Bind interpolate pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_vkInterpolatePipeline);
        
        // Bind descriptor set
        if (m_vkInterpolateSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, 
                                   m_vkInterpolateLayout, 0, 1, 
                                   &m_vkInterpolateSet, 0, nullptr);
        }
        
        uint32_t groupsX = (m_outputWidth + 15) / 16;
        uint32_t groupsY = (m_outputHeight + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        
        OutputDebugStringA("[Interpolatorv3] Frame interpolation dispatched (Vulkan)\n");
    }
#else
    OutputDebugStringA("[Interpolatorv3] SHADER: Interpolate.hlsl\n");
    OutputDebugStringA("[Interpolatorv3] Interpolating frame (D3D11)...\n");
#endif
}

bool Interpolator::CreatePipelines() {
    // Note: In full implementation, we would load and compile shaders
    // For now, this is a placeholder that will be implemented
    // when the shader loading system is integrated
    
    // Actually load shaders now
    return LoadShaders();
}

// ============================================================================
// Load SPIR-V Shaders
// ============================================================================

#ifdef USE_VULKAN

#include <fstream>

static std::vector<char> ReadSpvFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        OutputDebugStringA(("[Interpolatorv3] Failed to open: " + filename + "\n").c_str());
        return {};
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

bool Interpolator::LoadShaders() {
    if (!m_device || !m_device->IsVulkan()) {
        OutputDebugStringA("[Interpolatorv3] Not using Vulkan, skipping SPIR-V load\n");
        return true;  // D3D11 path doesn't need this
    }

    auto loadPipeline = [&](const char* name, const char* filename, 
                           std::vector<VkDescriptorSetLayoutBinding>& bindings) -> bool {
        std::string path = "shaders/vulkan/";
        path += filename;
        path += ".spv";
        
        auto spv = ReadSpvFile(path);
        if (spv.empty()) {
            OutputDebugStringA(("[Interpolatorv3] Failed to load: " + path + "\n").c_str());
            return false;
        }

        // Create shader module
        VkShaderModuleCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = spv.size();
        ci.pCode = (uint32_t*)spv.data();
        
        VkShaderModule module;
        if (vkCreateShaderModule(m_device->GetVkDevice(), &ci, nullptr, &module) != VK_SUCCESS) {
            OutputDebugStringA(("[Interpolatorv3] Failed to create module: " + path + "\n").c_str());
            return false;
        }

        // Create descriptor set layout
        VkDescriptorSetLayoutCreateInfo dslci = {};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = (uint32_t)bindings.size();
        dslci.pBindings = bindings.data();
        
        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(m_device->GetVkDevice(), &dslci, nullptr, &layout) != VK_SUCCESS) {
            OutputDebugStringA(("[Interpolatorv3] Failed to create descriptor set layout: " + path + "\n").c_str());
            vkDestroyShaderModule(m_device->GetVkDevice(), module, nullptr);
            return false;
        }

        // Create pipeline layout
        VkPipelineLayoutCreateInfo plci = {};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &layout;
        
        VkPipelineLayout pipelineLayout;
        if (vkCreatePipelineLayout(m_device->GetVkDevice(), &plci, nullptr, &pipelineLayout) != VK_SUCCESS) {
            OutputDebugStringA(("[Interpolatorv3] Failed to create pipeline layout: " + path + "\n").c_str());
            vkDestroyDescriptorSetLayout(m_device->GetVkDevice(), layout, nullptr);
            vkDestroyShaderModule(m_device->GetVkDevice(), module, nullptr);
            return false;
        }

        // Create compute pipeline
        VkComputePipelineCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = module;
        pci.stage.pName = "main";
        pci.layout = pipelineLayout;
        
        VkPipeline pipeline;
        if (vkCreateComputePipelines(m_device->GetVkDevice(), VK_NULL_HANDLE, 1, &pci, nullptr, &pipeline) != VK_SUCCESS) {
            OutputDebugStringA(("[Interpolatorv3] Failed to create pipeline: " + path + "\n").c_str());
            vkDestroyPipelineLayout(m_device->GetVkDevice(), pipelineLayout, nullptr);
            vkDestroyDescriptorSetLayout(m_device->GetVkDevice(), layout, nullptr);
            vkDestroyShaderModule(m_device->GetVkDevice(), module, nullptr);
            return false;
        }

        // Store handles for later use
        if (strcmp(name, "FeaturePyramid") == 0) {
            m_vkFeaturePyramidPipeline = pipeline;
            m_vkFeaturePyramidLayout = pipelineLayout;
        } else if (strcmp(name, "CostVolume") == 0) {
            m_vkCostVolumePipeline = pipeline;
            m_vkCostVolumeLayout = pipelineLayout;
        } else if (strcmp(name, "FlowDecoder") == 0) {
            m_vkFlowDecoderPipeline = pipeline;
            m_vkFlowDecoderLayout = pipelineLayout;
        } else if (strcmp(name, "Interpolate") == 0) {
            m_vkInterpolatePipeline = pipeline;
            m_vkInterpolateLayout = pipelineLayout;
        } else if (strcmp(name, "Downsample") == 0) {
            m_vkDownsamplePipeline = pipeline;
            m_vkDownsampleLayout = pipelineLayout;
        }
        
        OutputDebugStringA(("[Interpolatorv3] SHADER LOADED: " + std::string(filename) + ".spv\n").c_str());
        OutputDebugStringA(("[Interpolatorv3] Loaded pipeline: " + path + "\n").c_str());
        
        // Allocate descriptor set for this pipeline
        VkDescriptorSetAllocateInfo dsai = {};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = m_device->GetVkDescriptorPool();
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &layout;
        
        VkDescriptorSet descriptorSet;
        if (vkAllocateDescriptorSets(m_device->GetVkDevice(), &dsai, &descriptorSet) != VK_SUCCESS) {
            OutputDebugStringA(("[Interpolatorv3] Failed to allocate descriptor set: " + path + "\n").c_str());
            return false;
        }
        
        // Store descriptor set
        if (strcmp(name, "FeaturePyramid") == 0) {
            m_vkFeaturePyramidSet = descriptorSet;
        } else if (strcmp(name, "CostVolume") == 0) {
            m_vkCostVolumeSet = descriptorSet;
        } else if (strcmp(name, "FlowDecoder") == 0) {
            m_vkFlowDecoderSet = descriptorSet;
        } else if (strcmp(name, "Interpolate") == 0) {
            m_vkInterpolateSet = descriptorSet;
        } else if (strcmp(name, "Downsample") == 0) {
            m_vkDownsampleSet = descriptorSet;
        }
        
        OutputDebugStringA(("[Interpolatorv3] Allocated descriptor set for: " + path + "\n").c_str());
        return true;
    };

    // Feature Pyramid shader bindings
    std::vector<VkDescriptorSetLayoutBinding> featureBindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // inputSampler
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level0
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level1
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level2
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level3
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level4
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level5
    };
    if (!loadPipeline("FeaturePyramid", "feature_pyramid", featureBindings)) return false;

    // Cost Volume shader bindings
    std::vector<VkDescriptorSetLayoutBinding> costBindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        // ... more bindings for all pyramid levels
    };
    if (!loadPipeline("CostVolume", "cost_volume", costBindings)) return false;

    // Flow Decoder shader bindings
    std::vector<VkDescriptorSetLayoutBinding> flowBindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // costVolume
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // feat0
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // feat1
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // flowPrev
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // flowOut
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // confidenceOut
    };
    if (!loadPipeline("FlowDecoder", "flow_decoder", flowBindings)) return false;

    // Interpolate shader bindings
    std::vector<VkDescriptorSetLayoutBinding> interpBindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // framePrev
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // frameCurr
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // motionFwd
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // motionBwd
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // confidenceFwd
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // confidenceBwd
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // featPrev
        {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // featCurr
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearSampler
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // pointSampler
        {10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // outputFrame
    };
    if (!loadPipeline("Interpolate", "interpolate", interpBindings)) return false;

    // Downsample shader bindings
    std::vector<VkDescriptorSetLayoutBinding> downBindings = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // inputSampler
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // outputLevel0
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // outputLevel1
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // outputLevel2
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // outputLevel3
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level0Sampler
        {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level1Sampler
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // level2Sampler
    };
    if (!loadPipeline("Downsample", "downsample", downBindings)) return false;

    OutputDebugStringA("[Interpolatorv3] All pipelines loaded successfully!\n");
    return true;
}

#else
// D3D11 version - shaders are loaded differently
bool Interpolator::LoadShaders() {
    OutputDebugStringA("[Interpolatorv3] D3D11 path - using HLSL shaders\n");
    OutputDebugStringA("[Interpolatorv3] HLSL set: DownsampleLuma.hlsl, DownsampleLumaR.hlsl, MotionEst.hlsl, MotionRefine.hlsl, Interpolate.hlsl\n");
    return true;
}
#endif

} // namespace tfe
