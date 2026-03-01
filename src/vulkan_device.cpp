#include "vulkan_device.h"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

// Windows specific includes
#include <windows.h>
#include <dxgi1_2.h>
#include <d3d11.h>

// Try to include Vulkan (will fail gracefully if not available)
#ifdef VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#endif

namespace tfe {

// ============================================================================
// Constants
// ============================================================================

#ifdef _DEBUG
static const bool kEnableValidationLayers = true;
#else
static const bool kEnableValidationLayers = false;
#endif

static const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
static const char* kDebugExtensionName = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

// Required device extensions
static const std::vector<const char*> kRequiredDeviceExtensions = {
    VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME,
    VK_KHR_MAINTENANCE1_EXTENSION_NAME,
};

// Required instance extensions
static const std::vector<const char*> kRequiredInstanceExtensions = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
};

// ============================================================================
// VulkanDevice Constructor/Destructor
// ============================================================================

VulkanDevice::VulkanDevice()
{
}

VulkanDevice::~VulkanDevice() {
    Shutdown();
}

// ============================================================================
// Initialization
// ============================================================================

bool VulkanDevice::Initialize(bool preferVulkan) {
    m_preferVulkan = preferVulkan;
    
    // Try Vulkan first if preferred
    if (preferVulkan) {
        if (CreateInstance() && 
            SetupDebugMessenger() && 
            EnumeratePhysicalDevices() && 
            CreateDevice() &&
            CreateCommandBuffers()) {
            m_useVulkan = true;
            OutputDebugStringA("[VulkanDevice] Initialized Vulkan successfully\n");
            return true;
        }
        OutputDebugStringA("[VulkanDevice] Vulkan initialization failed, falling back to D3D11\n");
        Shutdown();
    }
    
    // Fallback to D3D11
    return CreateD3D11Device();
}

void VulkanDevice::Shutdown() {
    if (m_useVulkan) {
        // Wait for queue to finish
        if (m_computeQueue != VK_NULL_HANDLE) {
            vkQueueWaitIdle(m_computeQueue);
        }
        
        // Destroy Vulkan resources
        if (m_commandBuffer != VK_NULL_HANDLE && m_commandPool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_commandBuffer);
        }
        
        if (m_commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        }
        
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        }
        
        if (m_device != VK_NULL_HANDLE) {
            vkDestroyDevice(m_device, nullptr);
        }
        
        if (m_debugMessenger != VK_NULL_HANDLE) {
            auto vkDestroyDebugUtilsMessenger = (PFN_vkDestroyDebugUtilsMessengerEXT)
                vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
            if (vkDestroyDebugUtilsMessenger) {
                vkDestroyDebugUtilsMessenger(m_instance, m_debugMessenger, nullptr);
            }
        }
        
        if (m_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_instance, nullptr);
        }
        
        m_device = VK_NULL_HANDLE;
        m_physicalDevice = VK_NULL_HANDLE;
        m_instance = VK_NULL_HANDLE;
    } else {
        // Release D3D11 resources
        m_d3d11Device.Reset();
        m_d3d11Context.Reset();
    }
    
    m_useVulkan = false;
}

// ============================================================================
// Instance Creation
// ============================================================================

bool VulkanDevice::CreateInstance() {
    // Check if Vulkan is available
    PFN_vkEnumerateInstanceExtensionProperties enumerateExt = 
        (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties");
    if (!enumerateExt) {
        OutputDebugStringA("[VulkanDevice] Vulkan not available on this system\n");
        return false;
    }
    
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "TrueMotionFidelityEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "TrueMotionEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;
    
    auto extensions = GetRequiredExtensions();
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    // Enable validation layers in debug
    if (kEnableValidationLayers && CheckValidationLayerSupport()) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayerName;
        m_hasDebugExtension = true;
    } else {
        createInfo.enabledLayerCount = 0;
    }
    
    VK_CHECK_RET(vkCreateInstance(&createInfo, nullptr, &m_instance), false);
    
    OutputDebugStringA("[VulkanDevice] Vulkan instance created\n");
    return true;
}

bool VulkanDevice::SetupDebugMessenger() {
    if (!m_hasDebugExtension) return true;
    
    auto vkCreateDebugUtilsMessenger = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!vkCreateDebugUtilsMessenger) return true; // Not fatal
    
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                    void* pUserData) -> VkBool32 {
        OutputDebugStringA("[Vulkan] ");
        OutputDebugStringA(pCallbackData->pMessage);
        OutputDebugStringA("\n");
        return VK_FALSE;
    };
    
    VK_CHECK_RET(vkCreateDebugUtilsMessenger(m_instance, &createInfo, nullptr, &m_debugMessenger), false);
    
    OutputDebugStringA("[VulkanDevice] Debug messenger created\n");
    return true;
}

bool VulkanDevice::EnumeratePhysicalDevices() {
    uint32_t deviceCount = 0;
    VK_CHECK_RET(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr), false);
    
    if (deviceCount == 0) {
        OutputDebugStringA("[VulkanDevice] No Vulkan devices found\n");
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK_RET(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()), false);
    
    // Select first discrete GPU, prefer NVIDIA/AMD
    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_physicalDevice = dev;
            m_deviceName = props.deviceName;
            break;
        }
    }
    
    // Fallback to first device
    if (m_physicalDevice == VK_NULL_HANDLE) {
        m_physicalDevice = devices[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
        m_deviceName = props.deviceName;
    }
    
    OutputDebugStringA(("[VulkanDevice] Selected GPU: " + m_deviceName + "\n").c_str());
    
    // Query queue families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            m_queueFamilyIndices.graphicsFamily = i;
        }
        if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_queueFamilyIndices.computeFamily = i;
        }
        if (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT) {
            m_queueFamilyIndices.transferFamily = i;
        }
        i++;
    }
    
    if (!m_queueFamilyIndices.isComplete()) {
        // Use graphics as compute fallback
        if (!m_queueFamilyIndices.computeFamily.has_value() && m_queueFamilyIndices.graphicsFamily.has_value()) {
            m_queueFamilyIndices.computeFamily = m_queueFamilyIndices.graphicsFamily;
        }
    }
    
    // Check extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
    m_availableExtensions.resize(extCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, m_availableExtensions.data());
    
    if (!CheckDeviceExtensionSupport(m_physicalDevice)) {
        OutputDebugStringA("[VulkanDevice] Required device extensions not supported\n");
        return false;
    }
    
    return true;
}

bool VulkanDevice::CreateDevice() {
    float queuePriority = 1.0f;
    
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_queueFamilyIndices.getComputeFamily();
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    VkPhysicalDeviceFeatures2 deviceFeatures = {};
    deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    
    // Enable required features
    VkPhysicalDeviceVulkan12Features vulkan12Features = {};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    vulkan12Features.storageBuffer8BitAccess = VK_TRUE;
    
    deviceFeatures.pNext = &vulkan12Features;
    
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = &deviceFeatures;
    
    // Extensions
    auto extensions = kRequiredDeviceExtensions;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    // Layers
    if (kEnableValidationLayers && CheckValidationLayerSupport()) {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayerName;
    } else {
        createInfo.enabledLayerCount = 0;
    }
    
    VK_CHECK_RET(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device), false);
    
    vkGetDeviceQueue(m_device, m_queueFamilyIndices.getComputeFamily(), 0, &m_computeQueue);
    
    OutputDebugStringA("[VulkanDevice] Logical device created\n");
    return true;
}

bool VulkanDevice::CreateCommandBuffers() {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_queueFamilyIndices.getComputeFamily();
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    VK_CHECK_RET(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool), false);
    
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    VK_CHECK_RET(vkAllocateCommandBuffers(m_device, &allocInfo, &m_commandBuffer), false);
    
    OutputDebugStringA("[VulkanDevice] Command buffer created\n");
    return true;
}

bool VulkanDevice::CreateDescriptorPools() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 128 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128 },
    };
    
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 512;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    
    VK_CHECK_RET(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool), false);
    
    OutputDebugStringA("[VulkanDevice] Descriptor pool created\n");
    return true;
}

// ============================================================================
// D3D11 Fallback
// ============================================================================

bool VulkanDevice::CreateD3D11Device() {
    OutputDebugStringA("[VulkanDevice] Creating D3D11 fallback device\n");
    
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_DEBUG,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        &m_d3d11Device,
        nullptr,
        &m_d3d11Context
    );
    
    if (FAILED(hr)) {
        // Try WARP
        hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            0,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            &m_d3d11Device,
            nullptr,
            &m_d3d11Context
        );
    }
    
    if (FAILED(hr)) {
        OutputDebugStringA("[VulkanDevice] Failed to create D3D11 device\n");
        return false;
    }
    
    m_useVulkan = false;
    OutputDebugStringA("[VulkanDevice] D3D11 device created successfully\n");
    return true;
}

// ============================================================================
// Extension Helpers
// ============================================================================

bool VulkanDevice::CheckValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    
    for (const char* layerName : { kValidationLayerName }) {
        bool layerFound = false;
        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }
        if (!layerFound) return false;
    }
    return true;
}

bool VulkanDevice::CheckDeviceExtensionSupport(VkPhysicalDevice device) {
    for (const char* extName : kRequiredDeviceExtensions) {
        bool found = false;
        for (const auto& ext : m_availableExtensions) {
            if (strcmp(extName, ext.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

std::vector<const char*> VulkanDevice::GetRequiredExtensions() {
    std::vector<const char*> extensions = kRequiredInstanceExtensions;
    
    // Check for debug extension availability
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, availableExts.data());
    
    bool debugExtFound = false;
    for (const auto& ext : availableExts) {
        if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            debugExtFound = true;
            break;
        }
    }
    
    if (!debugExtFound) {
        extensions.erase(
            std::remove(extensions.begin(), extensions.end(), VK_EXT_DEBUG_UTILS_EXTENSION_NAME),
            extensions.end()
        );
    }
    
    return extensions;
}

// ============================================================================
// Memory Management
// ============================================================================

bool VulkanDevice::AllocateMemory(VkMemoryRequirements* requirements, VkMemoryPropertyFlags properties, VkDeviceMemory* outMemory) {
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = requirements->size;
    allocInfo.memoryTypeIndex = FindMemoryType(requirements->memoryTypeBits, properties);
    
    VK_CHECK_RET(vkAllocateMemory(m_device, &allocInfo, nullptr, outMemory), false);
    return true;
}

void VulkanDevice::FreeMemory(VkDeviceMemory memory) {
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, memory, nullptr);
    }
}

// ============================================================================
// Buffer Operations
// ============================================================================

bool VulkanDevice::CreateBuffer(const BufferCreateInfo& createInfo, VkBuffer* outBuffer, VkDeviceMemory* outMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = createInfo.size;
    bufferInfo.usage = createInfo.usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK_RET(vkCreateBuffer(m_device, &bufferInfo, nullptr, outBuffer), false);
    
    if (outMemory) {
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(m_device, *outBuffer, &memRequirements);
        
        if (!AllocateMemory(&memRequirements, createInfo.memoryFlags, outMemory)) {
            vkDestroyBuffer(m_device, *outBuffer, nullptr);
            *outBuffer = VK_NULL_HANDLE;
            return false;
        }
        
        VK_CHECK_RET(vkBindBufferMemory(m_device, *outBuffer, *outMemory, 0), false);
    }
    
    return true;
}

void VulkanDevice::DestroyBuffer(VkBuffer buffer, VkDeviceMemory memory) {
    if (buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, memory, nullptr);
    }
}

void* VulkanDevice::MapMemory(VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size) {
    void* data = nullptr;
    vkMapMemory(m_device, memory, offset, size, 0, &data);
    return data;
}

void VulkanDevice::UnmapMemory(VkDeviceMemory memory) {
    vkUnmapMemory(m_device, memory);
}

bool VulkanDevice::UploadToBuffer(VkBuffer buffer, const void* data, size_t size, size_t offset) {
    // For small uploads, use staging buffer
    if (size > m_stagingBufferSize) {
        // Recreate staging buffer
        if (m_stagingBuffer != VK_NULL_HANDLE) {
            DestroyBuffer(m_stagingBuffer, m_stagingMemory);
        }
        
        BufferCreateInfo info = {};
        info.size = size * 2; // Double to avoid frequent reallocations
        info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        info.memoryFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        
        if (!CreateBuffer(info, &m_stagingBuffer, &m_stagingMemory)) {
            return false;
        }
        m_stagingBufferSize = info.size;
    }
    
    // Copy to staging
    void* mapped = MapMemory(m_stagingMemory);
    if (!mapped) return false;
    
    memcpy((uint8_t*)mapped + offset, data, size);
    UnmapMemory(m_stagingMemory);
    
    // Copy to device buffer
    if (!BeginCommandBuffer(m_commandBuffer)) return false;
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = offset;
    copyRegion.dstOffset = offset;
    copyRegion.size = size;
    vkCmdCopyBuffer(m_commandBuffer, m_stagingBuffer, buffer, 1, &copyRegion);
    if (!EndCommandBuffer(m_commandBuffer)) return false;
    
    return QueueWaitIdle();
}

// ============================================================================
// Image Operations
// ============================================================================

bool VulkanDevice::CreateImage(const ImageCreateInfo& createInfo, VkImage* outImage, VkDeviceMemory* outMemory) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = createInfo.imageType;
    imageInfo.extent.width = createInfo.width;
    imageInfo.extent.height = createInfo.height;
    imageInfo.extent.depth = createInfo.depth;
    imageInfo.mipLevels = createInfo.mipLevels;
    imageInfo.arrayLayers = createInfo.arrayLayers;
    imageInfo.format = createInfo.format;
    imageInfo.tiling = createInfo.tiling;
    imageInfo.initialLayout = createInfo.initialLayout;
    imageInfo.usage = createInfo.usage;
    imageInfo.samples = createInfo.samples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VK_CHECK_RET(vkCreateImage(m_device, &imageInfo, nullptr, outImage), false);
    
    if (outMemory) {
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(m_device, *outImage, &memRequirements);
        
        if (!AllocateMemory(&memRequirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outMemory)) {
            vkDestroyImage(m_device, *outImage, nullptr);
            *outImage = VK_NULL_HANDLE;
            return false;
        }
        
        VK_CHECK_RET(vkBindImageMemory(m_device, *outImage, *outMemory, 0), false);
    }
    
    return true;
}

void VulkanDevice::DestroyImage(VkImage image, VkDeviceMemory memory) {
    if (image != VK_NULL_HANDLE) {
        vkDestroyImage(m_device, image, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, memory, nullptr);
    }
}

bool VulkanDevice::CreateImageView(VkImage image, VkFormat format, VkImageViewType viewType, 
                                   uint32_t baseMipLevel, uint32_t mipLevels, VkImageAspectFlags aspectMask,
                                   VkImageView* outView) {
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectMask;
    viewInfo.subresourceRange.baseMipLevel = baseMipLevel;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    
    VK_CHECK_RET(vkCreateImageView(m_device, &viewInfo, nullptr, outView), false);
    return true;
}

void VulkanDevice::DestroyImageView(VkImageView view) {
    if (view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, view, nullptr);
    }
}

// ============================================================================
// Sampler Operations
// ============================================================================

bool VulkanDevice::CreateSampler(VkFilter magFilter, VkFilter minFilter, VkSamplerMipmapMode mipmapMode,
                                 VkSamplerAddressMode addressModeU, VkSamplerAddressMode addressModeV,
                                 VkSamplerAddressMode addressModeW, VkSampler* outSampler) {
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = magFilter;
    samplerInfo.minFilter = minFilter;
    samplerInfo.mipmapMode = mipmapMode;
    samplerInfo.addressModeU = addressModeU;
    samplerInfo.addressModeV = addressModeV;
    samplerInfo.addressModeW = addressModeW;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    
    VK_CHECK_RET(vkCreateSampler(m_device, &samplerInfo, nullptr, outSampler), false);
    return true;
}

void VulkanDevice::DestroySampler(VkSampler sampler) {
    if (sampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, sampler, nullptr);
    }
}

// ============================================================================
// Descriptor Set Operations
// ============================================================================

bool VulkanDevice::CreateDescriptorSetLayout(const std::vector<DescriptorSetLayoutBinding>& bindings, VkDescriptorSetLayout* outLayout) {
    std::vector<VkDescriptorSetLayoutBinding> vkBindings(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++) {
        vkBindings[i].binding = bindings[i].binding;
        vkBindings[i].descriptorType = bindings[i].descriptorType;
        vkBindings[i].descriptorCount = bindings[i].descriptorCount;
        vkBindings[i].stageFlags = bindings[i].stageFlags;
    }
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(vkBindings.size());
    layoutInfo.pBindings = vkBindings.data();
    
    VK_CHECK_RET(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, outLayout), false);
    return true;
}

void VulkanDevice::DestroyDescriptorSetLayout(VkDescriptorSetLayout layout) {
    if (layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
    }
}

bool VulkanDevice::CreateDescriptorPool(uint32_t maxSets, const std::vector<VkDescriptorPoolSize>& poolSizes, VkDescriptorPool* outPool) {
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    
    VK_CHECK_RET(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, outPool), false);
    return true;
}

bool VulkanDevice::AllocateDescriptorSets(VkDescriptorPool pool, const std::vector<VkDescriptorSetLayout>& layouts, VkDescriptorSet* outSets) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();
    
    VK_CHECK_RET(vkAllocateDescriptorSets(m_device, &allocInfo, outSets), false);
    return true;
}

void VulkanDevice::FreeDescriptorSets(VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets) {
    vkFreeDescriptorSets(m_device, pool, count, sets);
}

void VulkanDevice::DestroyDescriptorPool(VkDescriptorPool pool) {
    if (pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, pool, nullptr);
    }
}

void VulkanDevice::UpdateDescriptorSets(const std::vector<WriteDescriptorSet>& writes, uint32_t descriptorWriteCount) {
    std::vector<VkWriteDescriptorSet> vkWrites(writes.size());
    std::vector<VkDescriptorImageInfo> imageInfos;
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    
    for (size_t i = 0; i < writes.size(); i++) {
        vkWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vkWrites[i].dstBinding = writes[i].dstBinding;
        vkWrites[i].dstArrayElement = writes[i].dstArrayElement;
        vkWrites[i].descriptorType = writes[i].descriptorType;
        
        if (writes[i].imageInfo) {
            VkDescriptorImageInfo info = {};
            info.imageView = writes[i].imageInfo->imageView;
            info.sampler = writes[i].imageInfo->sampler;
            info.imageLayout = writes[i].imageInfo->layout;
            imageInfos.push_back(info);
            vkWrites[i].pImageInfo = &imageInfos.back();
            vkWrites[i].descriptorCount = 1;
        }
        
        if (writes[i].bufferInfo) {
            bufferInfos.push_back({
                writes[i].bufferInfo->buffer,
                writes[i].bufferInfo->offset,
                writes[i].bufferInfo->range
            });
            vkWrites[i].pBufferInfo = &bufferInfos.back();
            vkWrites[i].descriptorCount = 1;
        }
    }
    
    vkUpdateDescriptorSets(m_device, descriptorWriteCount, vkWrites.data(), 0, nullptr);
}

// ============================================================================
// Pipeline Operations
// ============================================================================

bool VulkanDevice::CreatePipelineLayout(const PipelineLayoutCreateInfo& createInfo, VkPipelineLayout* outLayout) {
    // Create descriptor set layouts
    std::vector<VkDescriptorSetLayout> setLayouts;
    if (!createInfo.descriptorSetBindings.empty()) {
        for (const auto& binding : createInfo.descriptorSetBindings) {
            VkDescriptorSetLayout layout;
            if (!CreateDescriptorSetLayout({binding}, &layout)) {
                return false;
            }
            setLayouts.push_back(layout);
        }
    }
    
    // Convert push constants
    std::vector<VkPushConstantRange> pushConstants;
    for (const auto& pc : createInfo.pushConstantRanges) {
        pushConstants.push_back({
            pc.stageFlags,
            pc.offset,
            pc.size
        });
    }
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstants.size());
    pipelineLayoutInfo.pPushConstantRanges = pushConstants.data();
    
    VK_CHECK_RET(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, outLayout), false);
    
    return true;
}

void VulkanDevice::DestroyPipelineLayout(VkPipelineLayout layout) {
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, layout, nullptr);
    }
}

bool VulkanDevice::CreateComputePipeline(VkPipelineLayout layout, VkShaderModule shaderModule, const char* entryPoint,
                                        VkPipeline* outPipeline) {
    VkSpecializationInfo specInfo = {};
    
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = layout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = entryPoint;
    pipelineInfo.stage.pSpecializationInfo = &specInfo;
    
    VK_CHECK_RET(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, outPipeline), false);
    return true;
}

void VulkanDevice::DestroyComputePipeline(VkPipeline pipeline) {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, pipeline, nullptr);
    }
}

// ============================================================================
// Shader Module Operations
// ============================================================================

bool VulkanDevice::CreateShaderModule(const uint32_t* spirvData, size_t spirvSize, VkShaderModule* outModule) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvSize;
    createInfo.pCode = spirvData;
    
    VK_CHECK_RET(vkCreateShaderModule(m_device, &createInfo, nullptr, outModule), false);
    return true;
}

bool VulkanDevice::CreateShaderModuleFromFile(const char* filename, VkShaderModule* outModule) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        OutputDebugStringA(("[VulkanDevice] Failed to open shader file: " + std::string(filename) + "\n").c_str());
        return false;
    }
    
    size_t fileSize = file.tellg();
    std::vector<uint32_t> spirv(fileSize / 4);
    
    file.seekg(0);
    file.read((char*)spirv.data(), fileSize);
    file.close();
    
    return CreateShaderModule(spirv.data(), fileSize, outModule);
}

void VulkanDevice::DestroyShaderModule(VkShaderModule module) {
    if (module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, module, nullptr);
    }
}

// ============================================================================
// Command Buffer Operations
// ============================================================================

bool VulkanDevice::CreateCommandPool(uint32_t queueFamily, VkCommandPool* outPool) {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    VK_CHECK_RET(vkCreateCommandPool(m_device, &poolInfo, nullptr, outPool), false);
    return true;
}

void VulkanDevice::DestroyCommandPool(VkCommandPool pool) {
    if (pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, pool, nullptr);
    }
}

bool VulkanDevice::AllocateCommandBuffer(VkCommandPool pool, VkCommandBuffer* outBuffer) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    VK_CHECK_RET(vkAllocateCommandBuffers(m_device, &allocInfo, outBuffer), false);
    return true;
}

void VulkanDevice::FreeCommandBuffer(VkCommandPool pool, VkCommandBuffer buffer) {
    if (buffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_device, pool, 1, &buffer);
    }
}

bool VulkanDevice::BeginCommandBuffer(VkCommandBuffer buffer, VkCommandBufferUsageFlags flags) {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = flags;
    
    VK_CHECK_RET(vkBeginCommandBuffer(buffer, &beginInfo), false);
    return true;
}

bool VulkanDevice::EndCommandBuffer(VkCommandBuffer buffer) {
    VK_CHECK_RET(vkEndCommandBuffer(buffer), false);
    return true;
}

bool VulkanDevice::QueueSubmit(VkCommandBuffer buffer, VkSemaphore waitSemaphore, 
                               VkSemaphore signalSemaphore, VkFence fence) {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &buffer;
    
    if (waitSemaphore != VK_NULL_HANDLE) {
        VkSemaphoreWaitInfo waitInfo = {};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &waitSemaphore;
        waitInfo.pValues = nullptr;
        // Note: This is simplified - real implementation would handle properly
    }
    
    if (signalSemaphore != VK_NULL_HANDLE || fence != VK_NULL_HANDLE) {
        if (signalSemaphore != VK_NULL_HANDLE) {
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &signalSemaphore;
        }
    }
    
    VK_CHECK_RET(vkQueueSubmit(m_computeQueue, 1, &submitInfo, fence), false);
    return true;
}

bool VulkanDevice::QueueWaitIdle() {
    VK_CHECK_RET(vkQueueWaitIdle(m_computeQueue), false);
    return true;
}

// ============================================================================
// Pipeline Barriers
// ============================================================================

void VulkanDevice::CmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                     const std::vector<VkImageMemoryBarrier>& imageBarriers,
                                     const std::vector<VkBufferMemoryBarrier>& bufferBarriers) {
    // Handle buffer barriers
    if (!bufferBarriers.empty()) {
        vkCmdPipelineBarrier(
            buffer, srcStage, dstStage,
            0,
            0, nullptr,
            static_cast<uint32_t>(bufferBarriers.size()), bufferBarriers.data(),
            0, nullptr
        );
    }
    
    // Handle image barriers  
    if (!imageBarriers.empty()) {
        vkCmdPipelineBarrier(
            buffer, srcStage, dstStage,
            0,
            0, nullptr,
            0, nullptr,
            static_cast<uint32_t>(imageBarriers.size()), imageBarriers.data()
        );
    }
}

void VulkanDevice::CmdCopyBuffer(VkCommandBuffer buffer, VkBuffer src, VkBuffer dst, size_t size) {
    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(buffer, src, dst, 1, &copyRegion);
}

void VulkanDevice::CmdCopyImage(VkCommandBuffer buffer, VkImage src, VkImageLayout srcLayout,
                                VkImage dst, VkImageLayout dstLayout, uint32_t width, uint32_t height) {
    VkImageCopy copyRegion = {};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.extent.width = width;
    copyRegion.extent.height = height;
    copyRegion.extent.depth = 1;
    
    vkCmdCopyImage(buffer, src, srcLayout, dst, dstLayout, 1, &copyRegion);
}

void VulkanDevice::CmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer src, VkImage dst,
                                       VkImageLayout layout, uint32_t width, uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyBufferToImage(buffer, src, dst, layout, 1, &region);
}

void VulkanDevice::CmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage src, VkImageLayout srcLayout,
                                       VkBuffer dst, uint32_t width, uint32_t height) {
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyImageToBuffer(buffer, src, srcLayout, dst, 1, &region);
}

// ============================================================================
// Image Layout Transitions
// ============================================================================

bool VulkanDevice::TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout,
                                        VkImageLayout newLayout, uint32_t mipLevel, uint32_t mipLevels) {
    if (!BeginCommandBuffer(m_commandBuffer)) return false;
    
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = mipLevel;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    } else {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    
    CmdPipelineBarrier(m_commandBuffer, srcStage, dstStage, {barrier});
    
    if (!EndCommandBuffer(m_commandBuffer)) return false;
    return QueueWaitIdle();
}

// ============================================================================
// Debug
// ============================================================================

void VulkanDevice::SetDebugName(VkObjectType objectType, uint64_t objectHandle, const char* name) {
    if (!m_hasDebugExtension || m_device == VK_NULL_HANDLE) return;
    
    auto vkSetDebugUtilsObjectName = (PFN_vkSetDebugUtilsObjectNameEXT)
        vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT");
    
    if (!vkSetDebugUtilsObjectName) return;
    
    VkDebugUtilsObjectNameInfoEXT nameInfo = {};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = name;
    
    vkSetDebugUtilsObjectName(m_device, &nameInfo);
}

} // namespace tfe
