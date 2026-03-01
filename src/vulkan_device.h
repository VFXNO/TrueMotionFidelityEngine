#pragma once

#include "vulkan_types.h"

#include <d3d11.h>
#include <wrl/client.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace tfe {

// ============================================================================
// Vulkan Device - Main wrapper class for Vulkan operations
// ============================================================================

class VulkanDevice {
public:
    VulkanDevice();
    ~VulkanDevice();

    // =========================================================================
    // Initialization
    // =========================================================================

    // Initialize Vulkan (can fall back to D3D11 if Vulkan unavailable)
    bool Initialize(bool preferVulkan = true);
    void Shutdown();

    // Check if using Vulkan
    bool IsVulkan() const { return m_useVulkan; }
    
    // Get the D3D11 device (for fallback)
    ID3D11Device* GetD3D11Device() const { return m_d3d11Device.Get(); }
    ID3D11DeviceContext* GetD3D11Context() const { return m_d3d11Context.Get(); }
    
    // Get Vulkan device
    VkDevice GetVkDevice() const { return m_device; }
    VkPhysicalDevice GetVkPhysicalDevice() const { return m_physicalDevice; }
    VkQueue GetVkComputeQueue() const { return m_computeQueue; }
    uint32_t GetVkComputeQueueFamily() const { return m_queueFamilyIndices.getComputeFamily(); }

    // =========================================================================
    // Instance & Debug
    // =========================================================================

    VkInstance GetInstance() const { return m_instance; }
    const std::string& GetDeviceName() const { return m_deviceName; }
    bool HasExternalDebug() const { return m_hasDebugExtension; }

    // =========================================================================
    // Memory Management
    // =========================================================================

    uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool AllocateMemory(VkMemoryRequirements* requirements, VkMemoryPropertyFlags properties, VkDeviceMemory* outMemory);
    void FreeMemory(VkDeviceMemory memory);

    // =========================================================================
    // Buffer Operations
    // =========================================================================

    bool CreateBuffer(const BufferCreateInfo& createInfo, VkBuffer* outBuffer, VkDeviceMemory* outMemory = nullptr);
    void DestroyBuffer(VkBuffer buffer, VkDeviceMemory memory = VK_NULL_HANDLE);
    
    void* MapMemory(VkDeviceMemory memory, VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void UnmapMemory(VkDeviceMemory memory);
    
    // Copy buffer data (host to device)
    bool UploadToBuffer(VkBuffer buffer, const void* data, size_t size, size_t offset = 0);

    // =========================================================================
    // Image Operations
    // =========================================================================

    bool CreateImage(const ImageCreateInfo& createInfo, VkImage* outImage, VkDeviceMemory* outMemory = nullptr);
    void DestroyImage(VkImage image, VkDeviceMemory memory = VK_NULL_HANDLE);
    
    bool CreateImageView(VkImage image, VkFormat format, VkImageViewType viewType, uint32_t baseMipLevel, uint32_t mipLevels,
                        VkImageAspectFlags aspectMask, VkImageView* outView);
    void DestroyImageView(VkImageView view);

    // =========================================================================
    // Sampler Operations
    // =========================================================================

    bool CreateSampler(VkFilter magFilter, VkFilter minFilter, VkSamplerMipmapMode mipmapMode,
                      VkSamplerAddressMode addressModeU, VkSamplerAddressMode addressModeV,
                      VkSamplerAddressMode addressModeW, VkSampler* outSampler);
    void DestroySampler(VkSampler sampler);

    // =========================================================================
    // Descriptor Set Layout & Pool
    // =========================================================================

    bool CreateDescriptorSetLayout(const std::vector<DescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout* outLayout);
    void DestroyDescriptorSetLayout(VkDescriptorSetLayout layout);
    
    bool CreateDescriptorPool(uint32_t maxSets, const std::vector<VkDescriptorPoolSize>& poolSizes, VkDescriptorPool* outPool);
    bool AllocateDescriptorSets(VkDescriptorPool pool, const std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorSet* outSets);
    void FreeDescriptorSets(VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets);
    void DestroyDescriptorPool(VkDescriptorPool pool);
    
    void UpdateDescriptorSets(const std::vector<WriteDescriptorSet>& writes, uint32_t descriptorWriteCount);

    // =========================================================================
    // Pipeline Operations
    // =========================================================================

    bool CreatePipelineLayout(const PipelineLayoutCreateInfo& createInfo, VkPipelineLayout* outLayout);
    void DestroyPipelineLayout(VkPipelineLayout layout);
    
    bool CreateComputePipeline(VkPipelineLayout layout, VkShaderModule shaderModule, const char* entryPoint,
                               VkPipeline* outPipeline);
    void DestroyComputePipeline(VkPipeline pipeline);

    // =========================================================================
    // Shader Module Operations
    // =========================================================================

    bool CreateShaderModule(const uint32_t* spirvData, size_t spirvSize, VkShaderModule* outModule);
    bool CreateShaderModuleFromFile(const char* filename, VkShaderModule* outModule);
    void DestroyShaderModule(VkShaderModule module);

    // =========================================================================
    // Command Buffer & Execution
    // =========================================================================

    bool CreateCommandPool(uint32_t queueFamily, VkCommandPool* outPool);
    void DestroyCommandPool(VkCommandPool pool);
    
    bool AllocateCommandBuffer(VkCommandPool pool, VkCommandBuffer* outBuffer);
    void FreeCommandBuffer(VkCommandPool pool, VkCommandBuffer buffer);
    
    bool BeginCommandBuffer(VkCommandBuffer buffer, VkCommandBufferUsageFlags flags = 0);
    bool EndCommandBuffer(VkCommandBuffer buffer);
    
    // Submit compute work
    bool QueueSubmit(VkCommandBuffer buffer, VkSemaphore waitSemaphore = VK_NULL_HANDLE, 
                    VkSemaphore signalSemaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE);
    bool QueueWaitIdle();

    // =========================================================================
    // Pipeline Barrier & Memory Operations
    // =========================================================================

    void CmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                           const std::vector<VkImageMemoryBarrier>& imageBarriers,
                           const std::vector<VkBufferMemoryBarrier>& bufferBarriers = {});
    
    void CmdCopyBuffer(VkCommandBuffer buffer, VkBuffer src, VkBuffer dst, size_t size);
    void CmdCopyImage(VkCommandBuffer buffer, VkImage src, VkImageLayout srcLayout, 
                     VkImage dst, VkImageLayout dstLayout, uint32_t width, uint32_t height);
    void CmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer src, VkImage dst, 
                             VkImageLayout layout, uint32_t width, uint32_t height);
    void CmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage src, VkImageLayout srcLayout,
                             VkBuffer dst, uint32_t width, uint32_t height);

    // =========================================================================
    // Image Layout Transitions
    // =========================================================================

    bool TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout,
                              uint32_t mipLevel = 0, uint32_t mipLevels = VK_REMAINING_MIP_LEVELS);

    // =========================================================================
    // Debug
    // =========================================================================

    void SetDebugName(VkObjectType objectType, uint64_t objectHandle, const char* name);

private:
    // =========================================================================
    // Private Helpers
    // =========================================================================

    bool CreateInstance();
    bool SetupDebugMessenger();
    bool EnumeratePhysicalDevices();
    bool CreateDevice();
    bool CreateCommandBuffers();
    bool CreateDescriptorPools();
    
    bool CreateD3D11Device();
    
    // Extension helpers
    bool CheckValidationLayerSupport();
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device);
    std::vector<const char*> GetRequiredExtensions();
    
    // D3D11 Fallback
    bool m_useVulkan = false;
    bool m_preferVulkan = true;
    bool m_hasDebugExtension = false;
    
    // D3D11 resources (fallback)
    Microsoft::WRL::ComPtr<ID3D11Device> m_d3d11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3d11Context;
    
    // Vulkan resources
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    
    QueueFamilyIndices m_queueFamilyIndices;
    
    std::string m_deviceName;
    std::vector<VkExtensionProperties> m_availableExtensions;
    std::vector<VkLayerProperties> m_availableLayers;
    
    // Memory tracking
    VkDeviceMemory m_stagingMemory = VK_NULL_HANDLE;
    VkBuffer m_stagingBuffer = VK_NULL_HANDLE;
    size_t m_stagingBufferSize = 0;
};

// ============================================================================
// Inline Helper Functions
// ============================================================================

inline uint32_t VulkanDevice::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    // Fallback: try any available
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i)) {
            return i;
        }
    }
    
    return 0;
}

} // namespace tfe
