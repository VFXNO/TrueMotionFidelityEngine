#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <functional>

#include <windows.h>

namespace tfe {

// ============================================================================
// Vulkan Utility Macros
// ============================================================================

inline bool vkCheckResult(VkResult result, const char* call) {
    if (result != VK_SUCCESS) {
        char buf[256];
        snprintf(buf, sizeof(buf), "Vulkan error: %s failed with %d\n", call, result);
        OutputDebugStringA(buf);
        return false;
    }
    return true;
}

#define VK_CHECK(call) \
    do { \
        if (!vkCheckResult(call, #call)) { \
            return false; \
        } \
    } while(0)

#define VK_CHECK_RET(call, ret) \
    do { \
        if (!vkCheckResult(call, #call)) { \
            return ret; \
        } \
    } while(0)

// ============================================================================
// Forward Declarations
// ============================================================================

class VulkanDevice;
class VulkanBuffer;
class VulkanImage;
class VulkanImageView;
class VulkanSampler;
class VulkanDescriptorSet;
class VulkanPipeline;
class VulkanShaderModule;

// ============================================================================
// Vulkan Exception for detailed errors
// ============================================================================

class VulkanException : public std::exception {
public:
    VulkanException(VkResult result, const char* message)
        : m_result(result), m_message(message) {}
    
    const char* what() const noexcept override {
        return m_message.c_str();
    }
    
    VkResult result() const { return m_result; }
    
private:
    VkResult m_result;
    std::string m_message;
};

// ============================================================================
// Queue Family Indices
// ============================================================================

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> transferFamily;
    
    bool isComplete() const {
        return computeFamily.has_value();
    }
    
    uint32_t getComputeFamily() const {
        return computeFamily.value_or(graphicsFamily.value_or(0));
    }
};

// ============================================================================
// Vulkan Memory Requirements
// ============================================================================

struct MemoryRequirements {
    VkDeviceSize size = 0;
    uint32_t alignment = 0;
    uint32_t memoryTypeBits = 0;
};

// ============================================================================
// Image Create Info Helper
// ============================================================================

struct ImageCreateInfo {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageLayout initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageType imageType = VK_IMAGE_TYPE_2D;
};

// ============================================================================
// Buffer Create Info Helper
// ============================================================================

struct BufferCreateInfo {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags memoryFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    bool mapped = false;
};

// ============================================================================
// Descriptor Set Layout Binding
// ============================================================================

struct DescriptorSetLayoutBinding {
    uint32_t binding = 0;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    uint32_t descriptorCount = 1;
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
};

// ============================================================================
// Descriptor Write Info
// ============================================================================

struct DescriptorImageInfo {
    VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageView imageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct DescriptorBufferInfo {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
};

struct WriteDescriptorSet {
    uint32_t dstBinding = 0;
    uint32_t dstArrayElement = 0;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    std::optional<DescriptorImageInfo> imageInfo;
    std::optional<DescriptorBufferInfo> bufferInfo;
};

// ============================================================================
// Compute Pipeline Create Info
// ============================================================================

struct ComputePipelineCreateInfo {
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    const char* entryPoint = "main";
    std::vector<VkSpecializationMapEntry> specializationEntries;
    std::vector<uint32_t> specializationData;
    std::vector<DescriptorSetLayoutBinding> descriptorSetBindings;
};

// ============================================================================
// Push Constant Range
// ============================================================================

struct PushConstantRange {
    VkShaderStageFlags stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    uint32_t offset = 0;
    uint32_t size = 0;
};

// ============================================================================
// Pipeline Layout Create Info
// ============================================================================

struct PipelineLayoutCreateInfo {
    std::vector<DescriptorSetLayoutBinding> descriptorSetBindings;
    std::vector<PushConstantRange> pushConstantRanges;
};

// ============================================================================
// Clear Value for Images
// ============================================================================

struct ClearValue {
    std::optional<VkClearColorValue> color;
    std::optional<VkClearDepthStencilValue> depthStencil;
    
    static ClearValue Color(float r, float g, float b, float a) {
        ClearValue cv;
        cv.color = VkClearColorValue{{r, g, b, a}};
        return cv;
    }
    
    static ClearValue DepthStencil(float depth, uint32_t stencil) {
        ClearValue cv;
        cv.depthStencil = VkClearDepthStencilValue{depth, stencil};
        return cv;
    }
};

} // namespace tfe
