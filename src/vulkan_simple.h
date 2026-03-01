#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace tfe {

// Simple Vulkan device wrapper
class VulkanDevice {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    uint32_t computeQueueFamily = 0;
    std::string deviceName;

    bool Init() {
        // App info
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "TrueMotion";
        appInfo.apiVersion = VK_API_VERSION_1_2;

        // Extensions
        const char* exts[] = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        };

        VkInstanceCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ci.pApplicationInfo = &appInfo;
        ci.enabledExtensionCount = 2;
        ci.ppEnabledExtensionNames = exts;

        if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) return false;

        // Physical device
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) return false;
        std::vector<VkPhysicalDevice> devs(count);
        vkEnumeratePhysicalDevices(instance, &count, devs.data());
        physicalDevice = devs[0];

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physicalDevice, &props);
        deviceName = props.deviceName;

        // Queue family
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qcount, qprops.data());
        for (uint32_t i = 0; i < qcount; i++) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily = i;
                break;
            }
        }

        // Device
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci = {};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = computeQueueFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        const char* dext = VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME;
        VkDeviceCreateInfo dci = {};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = &dext;

        if (vkCreateDevice(physicalDevice, &dci, nullptr, &device) != VK_SUCCESS) return false;
        vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

        // Command pool
        VkCommandPoolCreateInfo cpci = {};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = computeQueueFamily;
        vkCreateCommandPool(device, &cpci, nullptr, &commandPool);

        VkCommandBufferAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        vkAllocateCommandBuffers(device, &ai, &commandBuffer);

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &bi);

        // Descriptor pool
        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64},
        };
        VkDescriptorPoolCreateInfo dpci = {};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 256;
        dpci.poolSizeCount = 3;
        dpci.pPoolSizes = sizes;
        vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool);

        return true;
    }

    void Shutdown() {
        if (device) vkDeviceWaitIdle(device);
        if (commandBuffer && commandPool) vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
        device = VK_NULL_HANDLE;
        instance = VK_NULL_HANDLE;
    }

    void Submit() {
        vkEndCommandBuffer(commandBuffer);
        VkSubmitInfo si = {};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &commandBuffer;
        vkQueueSubmit(computeQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(computeQueue);

        VkCommandBufferBeginInfo bi = {};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(commandBuffer, &bi);
    }
};

// Image wrapper
class VulkanImage {
public:
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    VulkanDevice* dev = nullptr;

    bool Create(VulkanDevice* d, uint32_t w, uint32_t h, VkFormat fmt, bool storage = true) {
        dev = d;
        width = w; height = h; format = fmt;

        VkImageCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.format = fmt;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (storage) ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;

        vkCreateImage(d->device, &ci, nullptr, &image);

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(d->device, image, &mr);

        VkMemoryAllocateInfo ai = {};
        ai.allocationSize = mr.size;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(d->physicalDevice, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
            if (mr.memoryTypeBits & (1 << i) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                ai.memoryTypeIndex = i;
                break;
            }
        }
        vkAllocateMemory(d->device, &ai, nullptr, &memory);
        vkBindImageMemory(d->device, image, memory, 0);

        VkImageViewCreateInfo vi = {};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(d->device, &vi, nullptr, &view);

        return true;
    }

    void Destroy() {
        if (dev && image) vkDestroyImage(dev->device, image, nullptr);
        if (dev && memory) vkFreeMemory(dev->device, memory, nullptr);
        if (dev && view) vkDestroyImageView(dev->device, view, nullptr);
        image = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        view = VK_NULL_HANDLE;
    }

    void Transition(VkImageLayout newLayout) {
        VkImageMemoryBarrier b = {};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = newLayout;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(dev->commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
};

// Shader module wrapper
class VulkanShader {
public:
    VkShaderModule mod = VK_NULL_HANDLE;
    VulkanDevice* dev = nullptr;

    bool Create(VulkanDevice* d, const uint32_t* spv, size_t size) {
        dev = d;
        VkShaderModuleCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = size;
        ci.pCode = spv;
        return vkCreateShaderModule(d->device, &ci, nullptr, &mod) == VK_SUCCESS;
    }

    void Destroy() {
        if (dev && mod) vkDestroyShaderModule(dev->device, mod, nullptr);
        mod = VK_NULL_HANDLE;
    }
};

// Pipeline wrapper
class VulkanPipeline {
public:
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VulkanDevice* dev = nullptr;

    bool Create(VulkanDevice* d, VkShaderModule mod, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        dev = d;

        VkDescriptorSetLayoutCreateInfo lci = {};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = static_cast<uint32_t>(bindings.size());
        lci.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(d->device, &lci, nullptr, &dsLayout);

        VkPipelineLayoutCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pci.setLayoutCount = 1;
        pci.pSetLayouts = &dsLayout;
        vkCreatePipelineLayout(d->device, &pci, nullptr, &layout);

        VkComputePipelineCreateInfo ci = {};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.layout = layout;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        return vkCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline) == VK_SUCCESS;
    }

    void Destroy() {
        if (dev) {
            if (pipeline) vkDestroyPipeline(dev->device, pipeline, nullptr);
            if (layout) vkDestroyPipelineLayout(dev->device, layout, nullptr);
            if (dsLayout) vkDestroyDescriptorSetLayout(dev->device, dsLayout, nullptr);
        }
        pipeline = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        dsLayout = VK_NULL_HANDLE;
    }
};

} // namespace
