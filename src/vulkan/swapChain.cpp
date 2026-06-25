#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <optional>
#include <set>

#include <cstdint> 
#include <limits>
#include <algorithm> 

#include <fstream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp> //for matrices
#include <glm/gtc/matrix_transform.hpp>




#include <array>


#include "initVulkan.h"
#include "device.h"
#include "utils.h"
#include "offscreen.h"
#include "buffer.h"   



#pragma region swapChain

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, appState& state) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        glfwGetFramebufferSize(state.window, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

void createSwapChain(appState& state) {
    if (USE_OFF_SCREEN_RENDERING) {
        state.swapChainExtent = {(uint32_t)state.WIDTH, (uint32_t)state.HEIGHT};
        state.swapChainImageFormat = VK_FORMAT_B8G8R8A8_SRGB;
        return;
    }
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(state);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities, state);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = state.surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(state);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = nullptr; // Optional
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform; 
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode; 
    createInfo.clipped = VK_TRUE; 
    
    createInfo.oldSwapchain = VK_NULL_HANDLE; 


    if (vkCreateSwapchainKHR(state.device, &createInfo, nullptr, &state.swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(state.device, state.swapChain, &imageCount, nullptr);
    state.swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(state.device, state.swapChain, &imageCount, state.swapChainImages.data());

    state.swapChainImageFormat = surfaceFormat.format;
    state.swapChainExtent = extent;

}

void cleanupSwapChain(appState& state) {
    
    for (size_t i = 0; i < state.depthImageViews.size(); i++) {
        vkDestroyImageView(state.device, state.depthImageViews[i], nullptr);
        vkDestroyImage(state.device, state.depthImages[i], nullptr);
        vkFreeMemory(state.device, state.depthImageMemories[i], nullptr);
    }
    state.depthImageViews.clear();
    state.depthImages.clear();
    state.depthImageMemories.clear();

    for (auto framebuffer : state.swapChainFramebuffers) {
        vkDestroyFramebuffer(state.device, framebuffer, nullptr);
    }

    if (USE_OFF_SCREEN_RENDERING) return;

    for (auto imageView : state.swapChainImageViews) {
        vkDestroyImageView(state.device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(state.device, state.swapChain, nullptr);
}

void createImageViews(appState& state)
{
    if (USE_OFF_SCREEN_RENDERING) return;
    
    state.swapChainImageViews.resize(state.swapChainImages.size());

    for (size_t i = 0; i < state.swapChainImages.size(); i++) {

        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = state.swapChainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = state.swapChainImageFormat;

        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(state.device, &createInfo, nullptr, &state.swapChainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

void createDepthResources(appState& state)
{
    state.depthFormat = VK_FORMAT_D32_SFLOAT;

    size_t count = USE_OFF_SCREEN_RENDERING
                       ? (size_t)state.MAX_FRAMES_IN_FLIGHT
                       : state.swapChainImageViews.size();
    state.depthImages.resize(count);
    state.depthImageMemories.resize(count);
    state.depthImageViews.resize(count);

    for (size_t i = 0; i < count; i++)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType     = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width  = state.swapChainExtent.width;
        imageInfo.extent.height = state.swapChainExtent.height;
        imageInfo.extent.depth  = 1;
        imageInfo.mipLevels     = 1;
        imageInfo.arrayLayers   = 1;
        imageInfo.format        = state.depthFormat;
        imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage(state.device, &imageInfo, nullptr, &state.depthImages[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create depth image!");

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(state.device, state.depthImages[i], &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, state);

        if (vkAllocateMemory(state.device, &allocInfo, nullptr, &state.depthImageMemories[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate depth image memory!");

        vkBindImageMemory(state.device, state.depthImages[i], state.depthImageMemories[i], 0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = state.depthImages[i];
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = state.depthFormat;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(state.device, &viewInfo, nullptr, &state.depthImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create depth image view!");
    }
}

void createFramebuffers(appState& state)
{
    size_t count = USE_OFF_SCREEN_RENDERING
                       ? (size_t)state.MAX_FRAMES_IN_FLIGHT
                       : state.swapChainImageViews.size();
    state.swapChainFramebuffers.resize(count);

    for (size_t i = 0; i < count; i++)
    {
        VkImageView attachments[] = {
            USE_OFF_SCREEN_RENDERING ? state.offscreenImageViews[i]
                                     : state.swapChainImageViews[i],
            state.depthImageViews[i]
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = state.renderPass;
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = state.swapChainExtent.width;
        framebufferInfo.height = state.swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(state.device, &framebufferInfo, nullptr, &state.swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void recreateSwapChain(appState& state) {
    vkDeviceWaitIdle(state.device);

    cleanupSwapChain(state);

    createSwapChain(state);
    createImageViews(state);
    createDepthResources(state);
    createFramebuffers(state);
}


#pragma endregion swapChain
