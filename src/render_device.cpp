#include "render_device.h"

#include <algorithm>
#include <fstream>
#include <cstring>

#ifdef USE_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <windows.h>
#include <dxgi1_2.h>
#include <d3d11.h>

namespace tfe {

// ============================================================================
// RenderDevice Implementation
// ============================================================================

RenderDevice::RenderDevice() {
}

RenderDevice::~RenderDevice() {
    Shutdown();
}

bool RenderDevice::Initialize(void* windowHandle, bool preferVulkan) {
    {
        std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
        debugFile << "RenderDevice::Initialize start, preferVulkan=" << preferVulkan << "\n";
        debugFile.flush();
    }

    // Always initialize D3D11 first (needed for swap chain, ImGui, textures)
    if (!InitializeD3D11(windowHandle)) {
        OutputDebugStringA("[RenderDevice] Failed to initialize D3D11 backend\n");
        return false;
    }

#ifdef USE_VULKAN
    // Try Vulkan as compute-only backend alongside D3D11
    if (preferVulkan && InitializeVulkan(windowHandle)) {
        m_useVulkan = true;
        m_backendName = "Vulkan+D3D11";
        {
            std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
            debugFile << "RenderDevice::Initialize Vulkan succeeded, m_useVulkan=true\n";
            debugFile.flush();
        }
        return true;
    }
#endif

    m_useVulkan = false;
    m_backendName = "D3D11";
    return true;
}

void RenderDevice::Shutdown() {
#ifdef USE_VULKAN
    if (m_useVulkan) {
        CleanupVulkan();
    } else {
#endif
        CleanupD3D11();
#ifdef USE_VULKAN
    }
#endif
    m_backendName = "None";
}

void RenderDevice::ResetDevice() {
    m_deviceLost = false;
    // Recreate resources if needed
}

void RenderDevice::Flush() {
#ifdef USE_VULKAN
    if (m_useVulkan && m_vkCommandBuffer != VK_NULL_HANDLE) {
        vkEndCommandBuffer(m_vkCommandBuffer);
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_vkCommandBuffer;
        vkQueueSubmit(m_vkComputeQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_vkComputeQueue);
        
        // Reset command buffer for reuse
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_vkCommandBuffer, &beginInfo);
    }
#endif
}

void RenderDevice::WaitIdle() {
#ifdef USE_VULKAN
    if (m_useVulkan && m_vkComputeQueue != VK_NULL_HANDLE) {
        vkQueueWaitIdle(m_vkComputeQueue);
    }
#endif
}

// ============================================================================
// Vulkan Implementation
// ============================================================================

#ifdef USE_VULKAN

bool RenderDevice::InitializeVulkan(void* windowHandle) {
    {
        std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
        debugFile << "InitializeVulkan ENTER\n";
        debugFile.flush();
    }
    // Check for Vulkan
    PFN_vkEnumerateInstanceExtensionProperties extFn = 
        (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties");
    if (!extFn) {
        {
            std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
            debugFile << "InitializeVulkan FAIL: vkGetInstanceProcAddr failed\n";
            debugFile.flush();
        }
        OutputDebugStringA("[RenderDevice] Vulkan not available\n");
        return false;
    }

    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "TrueMotionFidelityEngine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "TrueMotionEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    // Check for debug utils extension
    uint32_t extCount = 0;
    extFn(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    extFn(nullptr, &extCount, availableExts.data());
    
    bool hasDebugExt = false;
    for (const auto& ext : availableExts) {
        if (strcmp(ext.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
            hasDebugExt = true;
            break;
        }
    }

    std::vector<const char*> enabledExts = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    };
    if (hasDebugExt)
        enabledExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = (uint32_t)enabledExts.size();
    createInfo.ppEnabledExtensionNames = enabledExts.data();

    if (vkCreateInstance(&createInfo, nullptr, &m_vkInstance) != VK_SUCCESS) {
        {
            std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
            debugFile << "InitializeVulkan FAIL: vkCreateInstance failed\n";
            debugFile << "  numExts=" << enabledExts.size() << "\n";
            for (size_t i = 0; i < enabledExts.size(); i++) {
                debugFile << "  ext[" << i << "]=" << enabledExts[i] << "\n";
            }
            debugFile.flush();
        }
        OutputDebugStringA("[RenderDevice] Failed to create Vulkan instance\n");
        return false;
    }

    // Enumerate physical devices
    uint32_t deviceCount = 0;
    if (vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, nullptr) != VK_SUCCESS || deviceCount == 0) {
        {
            std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
            debugFile << "InitializeVulkan FAIL: No devices found\n";
            debugFile.flush();
        }
        OutputDebugStringA("[RenderDevice] No Vulkan devices found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_vkInstance, &deviceCount, devices.data());

    // Select first discrete GPU
    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            m_vkPhysicalDevice = dev;
            break;
        }
    }
    if (m_vkPhysicalDevice == VK_NULL_HANDLE) {
        m_vkPhysicalDevice = devices[0];
    }

    // Get queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_vkPhysicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_vkQueueFamily = i;
            break;
        }
    }
    if (m_vkQueueFamily == 0 && queueFamilyCount > 0) {
        // Fallback to first queue with graphics
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                m_vkQueueFamily = i;
                break;
            }
        }
    }

    // Create device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_vkQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Check which device extensions are available
    uint32_t devExtCount = 0;
    vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &devExtCount, nullptr);
    std::vector<VkExtensionProperties> availableDevExts(devExtCount);
    vkEnumerateDeviceExtensionProperties(m_vkPhysicalDevice, nullptr, &devExtCount, availableDevExts.data());

    auto hasDevExt = [&](const char* name) {
      for (const auto& e : availableDevExts)
        if (strcmp(e.extensionName, name) == 0) return true;
      return false;
    };

    std::vector<const char*> deviceExtVec;
    deviceExtVec.push_back(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);

    // External memory extensions for D3D11<->Vulkan zero-copy sharing
    bool hasExtMem = hasDevExt(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME)
                  && hasDevExt(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    if (hasExtMem) {
      deviceExtVec.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
      deviceExtVec.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    // Dedicated allocation (improves import compatibility)
    if (hasDevExt(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME)) {
      deviceExtVec.push_back(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
    }
    if (hasDevExt(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)) {
      deviceExtVec.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    }

    {
      std::ofstream debugFile("vulkan_debug.txt", std::ios::app);
      debugFile << "VkDevice exts: hasExtMem=" << hasExtMem << "\n";
      for (const auto* e : deviceExtVec) debugFile << "  " << e << "\n";
      debugFile.flush();
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtVec.size();
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtVec.data();

    if (vkCreateDevice(m_vkPhysicalDevice, &deviceCreateInfo, nullptr, &m_vkDevice) != VK_SUCCESS) {
        OutputDebugStringA("[RenderDevice] Failed to create Vulkan device\n");
        return false;
    }

    vkGetDeviceQueue(m_vkDevice, m_vkQueueFamily, 0, &m_vkComputeQueue);

    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_vkQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(m_vkDevice, &poolInfo, nullptr, &m_vkCommandPool);

    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_vkDevice, &allocInfo, &m_vkCommandBuffer);

    // Begin command buffer
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(m_vkCommandBuffer, &beginInfo);

    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 },
    };
    VkDescriptorPoolCreateInfo descPoolInfo = {};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.maxSets = 256;
    descPoolInfo.poolSizeCount = poolSizes.size();
    descPoolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(m_vkDevice, &descPoolInfo, nullptr, &m_vkDescriptorPool);

    // Query device capabilities
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_vkPhysicalDevice, &props);
    m_maxComputeWorkGroupSize = props.limits.maxComputeWorkGroupSize[0];
    m_maxImageSize = props.limits.maxImageDimension2D;

    OutputDebugStringA(("[RenderDevice] Vulkan initialized: " + std::string(props.deviceName) + "\n").c_str());
    return true;
}

void RenderDevice::CleanupVulkan() {
    if (m_vkDevice != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_vkDevice);

        if (m_vkCommandBuffer != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(m_vkDevice, m_vkCommandPool, 1, &m_vkCommandBuffer);
        }
        if (m_vkCommandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_vkDevice, m_vkCommandPool, nullptr);
        }
        if (m_vkDescriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_vkDevice, m_vkDescriptorPool, nullptr);
        }
        vkDestroyDevice(m_vkDevice, nullptr);
    }
    if (m_vkInstance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_vkInstance, nullptr);
    }

    m_vkDevice = VK_NULL_HANDLE;
    m_vkInstance = VK_NULL_HANDLE;
}

#endif // USE_VULKAN

// ============================================================================
// D3D11 Implementation
// ============================================================================

bool RenderDevice::InitializeD3D11(void* windowHandle) {
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
        OutputDebugStringA("[RenderDevice] Failed to create D3D11 device\n");
        return false;
    }

    m_maxComputeWorkGroupSize = 256;
    m_maxImageSize = 16384;

    OutputDebugStringA("[RenderDevice] D3D11 initialized\n");
    return true;
}

void RenderDevice::CleanupD3D11() {
    if (m_d3d11Context) {
        m_d3d11Context->Flush();
        m_d3d11Context.Reset();
    }
    m_d3d11Device.Reset();
}

// ============================================================================
// Texture Implementation
// ============================================================================

Texture::Texture() {
}

Texture::~Texture() {
    Destroy();
}

#ifdef USE_VULKAN

bool Texture::Create(RenderDevice* device, uint32_t width, uint32_t height,
                     VkFormat format, VkImageUsageFlags usage, bool allowUAV) {
    m_device = device;
    m_width = width;
    m_height = height;
    m_format = format;

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device->GetVkDevice(), &imageInfo, nullptr, &m_vkImage) != VK_SUCCESS) {
        return false;
    }

    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device->GetVkDevice(), m_vkImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.allocationSize = memRequirements.size;
    
    // Find memory type
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device->GetVkPhysicalDevice(), &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (memRequirements.memoryTypeBits & (1 << i)) {
            if (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                allocInfo.memoryTypeIndex = i;
                break;
            }
        }
    }

    vkAllocateMemory(device->GetVkDevice(), &allocInfo, nullptr, &m_vkMemory);
    vkBindImageMemory(device->GetVkDevice(), m_vkImage, m_vkMemory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_vkImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device->GetVkDevice(), &viewInfo, nullptr, &m_vkImageView);

    return true;
}

void Texture::TransitionToRead(VkCommandBuffer cmd, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_vkImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(cmd, 
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

#else // D3D11

bool Texture::Create(RenderDevice* device, uint32_t width, uint32_t height,
                     DXGI_FORMAT format, D3D11_USAGE usage, bool allowUAV) {
    m_device = device;
    m_width = width;
    m_height = height;
    m_format = format;

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = usage;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    if (allowUAV) {
        desc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
    }

    if (FAILED(device->GetD3D11Device()->CreateTexture2D(&desc, nullptr, &m_d3d11Texture))) {
        return false;
    }

    // Create SRV
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = format;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    device->GetD3D11Device()->CreateShaderResourceView(m_d3d11Texture.Get(), &srvDesc, &m_d3d11SRV);

    // Create UAV if requested
    if (allowUAV) {
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = format;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        device->GetD3D11Device()->CreateUnorderedAccessView(m_d3d11Texture.Get(), &uavDesc, &m_d3d11UAV);
    }

    return true;
}

#endif

void Texture::Destroy() {
#ifdef USE_VULKAN
    if (m_device && m_vkImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device->GetVkDevice(), m_vkImageView, nullptr);
    }
    if (m_device && m_vkImage != VK_NULL_HANDLE) {
        vkDestroyImage(m_device->GetVkDevice(), m_vkImage, nullptr);
    }
    if (m_device && m_vkMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device->GetVkDevice(), m_vkMemory, nullptr);
    }
    m_vkImageView = VK_NULL_HANDLE;
    m_vkImage = VK_NULL_HANDLE;
    m_vkMemory = VK_NULL_HANDLE;
#else
    m_d3d11SRV.Reset();
    m_d3d11UAV.Reset();
    m_d3d11Texture.Reset();
#endif
    m_width = 0;
    m_height = 0;
}

// ============================================================================
// Buffer Implementation
// ============================================================================

Buffer::Buffer() {
}

Buffer::~Buffer() {
    Destroy();
}

#ifdef USE_VULKAN

bool Buffer::Create(RenderDevice* device, size_t size, VkBufferUsageFlags usage, bool cpuAccessible) {
    m_device = device;
    m_size = size;

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device->GetVkDevice(), &bufferInfo, nullptr, &m_vkBuffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device->GetVkDevice(), m_vkBuffer, &memRequirements);

    VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (cpuAccessible) {
        memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.allocationSize = memRequirements.size;
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device->GetVkPhysicalDevice(), &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (memRequirements.memoryTypeBits & (1 << i)) {
            if (memProperties.memoryTypes[i].propertyFlags & memFlags) {
                allocInfo.memoryTypeIndex = i;
                break;
            }
        }
    }

    vkAllocateMemory(device->GetVkDevice(), &allocInfo, nullptr, &m_vkMemory);
    vkBindBufferMemory(device->GetVkDevice(), m_vkBuffer, m_vkMemory, 0);

    return true;
}

void* Buffer::Map() {
    if (m_vkMemory == VK_NULL_HANDLE) return nullptr;
    void* data = nullptr;
    vkMapMemory(m_device->GetVkDevice(), m_vkMemory, 0, m_size, 0, &data);
    return data;
}

void Buffer::Unmap() {
    if (m_vkMemory != VK_NULL_HANDLE) {
        vkUnmapMemory(m_device->GetVkDevice(), m_vkMemory);
    }
}

#else // D3D11

bool Buffer::Create(RenderDevice* device, size_t size, D3D11_USAGE usage, bool cpuAccessible) {
    m_device = device;
    m_size = size;

    D3D11_BUFFER_DESC desc = {};
    desc.ByteWidth = static_cast<UINT>(size);
    desc.Usage = usage;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    if (cpuAccessible) {
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    }

    if (FAILED(device->GetD3D11Device()->CreateBuffer(&desc, nullptr, &m_d3d11Buffer))) {
        return false;
    }
    return true;
}

void* Buffer::Map() {
    if (!m_d3d11Buffer) return nullptr;
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    m_device->GetD3D11Context()->Map(m_d3d11Buffer.Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    return mapped.pData;
}

void Buffer::Unmap() {
    if (m_d3d11Buffer) {
        m_device->GetD3D11Context()->Unmap(m_d3d11Buffer.Get(), 0);
    }
}

#endif

void Buffer::Destroy() {
#ifdef USE_VULKAN
    if (m_device && m_vkBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device->GetVkDevice(), m_vkBuffer, nullptr);
    }
    if (m_device && m_vkMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device->GetVkDevice(), m_vkMemory, nullptr);
    }
    m_vkBuffer = VK_NULL_HANDLE;
    m_vkMemory = VK_NULL_HANDLE;
#else
    m_d3d11Buffer.Reset();
#endif
    m_size = 0;
}

// ============================================================================
// ComputePipeline Implementation
// ============================================================================

ComputePipeline::ComputePipeline() {
}

ComputePipeline::~ComputePipeline() {
    Destroy();
}

#ifdef USE_VULKAN

bool ComputePipeline::Create(RenderDevice* device, const char* shaderCode, size_t shaderSize,
                            const std::vector<DescriptorSetLayoutBinding>& bindings) {
    m_device = device;

    // Create shader module
    VkShaderModuleCreateInfo moduleInfo = {};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = shaderSize;
    moduleInfo.pCode = (const uint32_t*)shaderCode;

    if (vkCreateShaderModule(device->GetVkDevice(), &moduleInfo, nullptr, &m_vkShaderModule) != VK_SUCCESS) {
        return false;
    }

    // Create descriptor set layout
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

    vkCreateDescriptorSetLayout(device->GetVkDevice(), &layoutInfo, nullptr, &m_vkDescriptorSetLayout);

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_vkDescriptorSetLayout;

    vkCreatePipelineLayout(device->GetVkDevice(), &pipelineLayoutInfo, nullptr, &m_vkPipelineLayout);

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_vkPipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = m_vkShaderModule;
    pipelineInfo.stage.pName = "main";

    vkCreateComputePipelines(device->GetVkDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_vkPipeline);

    return true;
}

#else // D3D11

bool ComputePipeline::Create(RenderDevice* device, const void* shaderBytecode, size_t bytecodeSize) {
    m_device = device;

    if (FAILED(device->GetD3D11Device()->CreateComputeShader(shaderBytecode, bytecodeSize, nullptr, &m_d3d11Shader))) {
        return false;
    }
    return true;
}

#endif

void ComputePipeline::Destroy() {
#ifdef USE_VULKAN
    if (m_device) {
        if (m_vkPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device->GetVkDevice(), m_vkPipeline, nullptr);
        }
        if (m_vkPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device->GetVkDevice(), m_vkPipelineLayout, nullptr);
        }
        if (m_vkDescriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device->GetVkDevice(), m_vkDescriptorSetLayout, nullptr);
        }
        if (m_vkShaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device->GetVkDevice(), m_vkShaderModule, nullptr);
        }
    }
    m_vkPipeline = VK_NULL_HANDLE;
    m_vkPipelineLayout = VK_NULL_HANDLE;
    m_vkDescriptorSetLayout = VK_NULL_HANDLE;
    m_vkShaderModule = VK_NULL_HANDLE;
#else
    m_d3d11Shader.Reset();
#endif
}

// ============================================================================
// Sampler Implementation
// ============================================================================

Sampler::Sampler() {
}

Sampler::~Sampler() {
    Destroy();
}

#ifdef USE_VULKAN

bool Sampler::Create(RenderDevice* device, VkFilter magFilter, VkFilter minFilter,
                     VkSamplerMipmapMode mipmapMode, VkSamplerAddressMode addressMode) {
    m_device = device;

    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = magFilter;
    samplerInfo.minFilter = minFilter;
    samplerInfo.mipmapMode = mipmapMode;
    samplerInfo.addressModeU = addressMode;
    samplerInfo.addressModeV = addressMode;
    samplerInfo.addressModeW = addressMode;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    return vkCreateSampler(device->GetVkDevice(), &samplerInfo, nullptr, &m_vkSampler) == VK_SUCCESS;
}

#else // D3D11

bool Sampler::Create(RenderDevice* device, D3D11_FILTER filter, D3D11_TEXTURE_ADDRESS_MODE addressMode) {
    m_device = device;

    D3D11_SAMPLER_DESC desc = {};
    desc.Filter = filter;
    desc.AddressU = addressMode;
    desc.AddressV = addressMode;
    desc.AddressW = addressMode;
    desc.MinLOD = 0;
    desc.MaxLOD = D3D11_FLOAT32_MAX;

    return SUCCEEDED(device->GetD3D11Device()->CreateSamplerState(&desc, &m_d3d11Sampler));
}

#endif

void Sampler::Destroy() {
#ifdef USE_VULKAN
    if (m_device && m_vkSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device->GetVkDevice(), m_vkSampler, nullptr);
    }
    m_vkSampler = VK_NULL_HANDLE;
#else
    m_d3d11Sampler.Reset();
#endif
}

} // namespace tfe
