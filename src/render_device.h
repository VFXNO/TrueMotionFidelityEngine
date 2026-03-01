#pragma once

#ifdef USE_VULKAN
#include "vulkan_types.h"
#endif

#include <d3d11.h>
#include <wrl/client.h>

#include <string>
#include <vector>

namespace tfe {

// ============================================================================
// Render Device Abstraction - Supports both Vulkan and D3D11
// ============================================================================

class RenderDevice {
public:
    RenderDevice();
    ~RenderDevice();

    // Initialization
    bool Initialize(void* windowHandle, bool preferVulkan = true);
    void Shutdown();

    // Backend info
    bool IsVulkan() const { return m_useVulkan; }
    const std::string& GetBackendName() const { return m_backendName; }

#ifdef USE_VULKAN
    // Vulkan getters
    VkDevice GetVkDevice() const { return m_vkDevice; }
    VkPhysicalDevice GetVkPhysicalDevice() const { return m_vkPhysicalDevice; }
    VkQueue GetVkComputeQueue() const { return m_vkComputeQueue; }
    uint32_t GetVkComputeQueueFamily() const { return m_vkQueueFamily; }
    VkCommandBuffer GetVkCommandBuffer() const { return m_vkCommandBuffer; }
    VkDescriptorPool GetVkDescriptorPool() const { return m_vkDescriptorPool; }
#endif

    // D3D11 getters (always available)
    ID3D11Device* GetD3D11Device() const { return m_d3d11Device.Get(); }
    ID3D11DeviceContext* GetD3D11Context() const { return m_d3d11Context.Get(); }

    // Device lost handling
    bool IsDeviceLost() const { return m_deviceLost; }
    void MarkDeviceLost() { m_deviceLost = true; }
    void ResetDevice();

    // Synchronization
    void Flush();
    void WaitIdle();

    // Helper methods
    uint32_t GetMaxComputeWorkGroupSize() const { return m_maxComputeWorkGroupSize; }
    uint32_t GetMaxImageSize() const { return m_maxImageSize; }

private:
    bool InitializeVulkan(void* windowHandle);
    bool InitializeD3D11(void* windowHandle);
    void CleanupVulkan();
    void CleanupD3D11();

    bool m_useVulkan = false;
    bool m_deviceLost = false;
    std::string m_backendName = "Unknown";

#ifdef USE_VULKAN
    // Vulkan resources
    VkInstance m_vkInstance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_vkDebugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_vkPhysicalDevice = VK_NULL_HANDLE;
    VkDevice m_vkDevice = VK_NULL_HANDLE;
    VkQueue m_vkComputeQueue = VK_NULL_HANDLE;
    VkCommandPool m_vkCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_vkCommandBuffer = VK_NULL_HANDLE;
    VkDescriptorPool m_vkDescriptorPool = VK_NULL_HANDLE;
    uint32_t m_vkQueueFamily = 0;
#endif

    // D3D11 resources (always needed for ImGui fallback)
    Microsoft::WRL::ComPtr<ID3D11Device> m_d3d11Device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> m_d3d11Context;

    // Device capabilities
    uint32_t m_maxComputeWorkGroupSize = 256;
    uint32_t m_maxImageSize = 4096;
};

// ============================================================================
// Texture/Image Abstraction
// ============================================================================

class Texture {
public:
    Texture();
    ~Texture();

#ifdef USE_VULKAN
    bool Create(RenderDevice* device, uint32_t width, uint32_t height, 
                VkFormat format, VkImageUsageFlags usage, bool allowUAV = true);
    VkImage GetVkImage() const { return m_vkImage; }
    VkImageView GetVkImageView() const { return m_vkImageView; }
    VkImageView GetVkUAV() const { return m_vkImageView; }
    void TransitionToRead(VkCommandBuffer cmd, VkImageLayout newLayout);
#else
    bool Create(RenderDevice* device, uint32_t width, uint32_t height,
                DXGI_FORMAT format, D3D11_USAGE usage, bool allowUAV = true);
    ID3D11Texture2D* GetD3D11Texture() const { return m_d3d11Texture.Get(); }
    ID3D11ShaderResourceView* GetD3D11SRV() const { return m_d3d11SRV.Get(); }
    ID3D11UnorderedAccessView* GetD3D11UAV() const { return m_d3d11UAV.Get(); }
#endif

    void Destroy();

    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }

protected:
    RenderDevice* m_device = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;

#ifdef USE_VULKAN
    VkImage m_vkImage = VK_NULL_HANDLE;
    VkDeviceMemory m_vkMemory = VK_NULL_HANDLE;
    VkImageView m_vkImageView = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
#else
    Microsoft::WRL::ComPtr<ID3D11Texture2D> m_d3d11Texture;
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> m_d3d11SRV;
    Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView> m_d3d11UAV;
    DXGI_FORMAT m_format = DXGI_FORMAT_UNKNOWN;
#endif
};

// ============================================================================
// Buffer Abstraction
// ============================================================================

class Buffer {
public:
    Buffer();
    ~Buffer();

#ifdef USE_VULKAN
    bool Create(RenderDevice* device, size_t size, VkBufferUsageFlags usage, 
                bool cpuAccessible = false);
    VkBuffer GetVkBuffer() const { return m_vkBuffer; }
    void* Map();
    void Unmap();
#else
    bool Create(RenderDevice* device, size_t size, D3D11_USAGE usage, bool cpuAccessible = false);
    ID3D11Buffer* GetD3D11Buffer() const { return m_d3d11Buffer.Get(); }
    void* Map();
    void Unmap();
#endif

    void Destroy();
    size_t GetSize() const { return m_size; }

protected:
    RenderDevice* m_device = nullptr;
    size_t m_size = 0;

#ifdef USE_VULKAN
    VkBuffer m_vkBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vkMemory = VK_NULL_HANDLE;
#else
    Microsoft::WRL::ComPtr<ID3D11Buffer> m_d3d11Buffer;
#endif
};

// ============================================================================
// Pipeline/Shader Abstraction
// ============================================================================

class ComputePipeline {
public:
    ComputePipeline();
    ~ComputePipeline();

#ifdef USE_VULKAN
    bool Create(RenderDevice* device, const char* shaderCode, size_t shaderSize,
                const std::vector<DescriptorSetLayoutBinding>& bindings);
    VkPipeline GetVkPipeline() const { return m_vkPipeline; }
    VkPipelineLayout GetVkPipelineLayout() const { return m_vkPipelineLayout; }
    VkDescriptorSetLayout GetVkDescriptorSetLayout() const { return m_vkDescriptorSetLayout; }
#else
    bool Create(RenderDevice* device, const void* shaderBytecode, size_t bytecodeSize);
    ID3D11ComputeShader* GetD3D11Shader() const { return m_d3d11Shader.Get(); }
#endif

    void Destroy();

protected:
    RenderDevice* m_device = nullptr;

#ifdef USE_VULKAN
    VkShaderModule m_vkShaderModule = VK_NULL_HANDLE;
    VkPipeline m_vkPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_vkPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_vkDescriptorSetLayout = VK_NULL_HANDLE;
#else
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> m_d3d11Shader;
#endif
};

// ============================================================================
// Sampler Abstraction
// ============================================================================

class Sampler {
public:
    Sampler();
    ~Sampler();

#ifdef USE_VULKAN
    bool Create(RenderDevice* device, VkFilter magFilter, VkFilter minFilter,
                VkSamplerMipmapMode mipmapMode, VkSamplerAddressMode addressMode);
    VkSampler GetVkSampler() const { return m_vkSampler; }
#else
    bool Create(RenderDevice* device, D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE addressMode);
    ID3D11SamplerState* GetD3D11Sampler() const { return m_d3d11Sampler.Get(); }
#endif

    void Destroy();

protected:
    RenderDevice* m_device = nullptr;

#ifdef USE_VULKAN
    VkSampler m_vkSampler = VK_NULL_HANDLE;
#else
    Microsoft::WRL::ComPtr<ID3D11SamplerState> m_d3d11Sampler;
#endif
};

} // namespace tfe
