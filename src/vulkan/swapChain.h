#ifndef SWAPCHAIN_H
#define SWAPCHAIN_H

#include "utils.h"
#include <vulkan/vulkan.h>
#include <vector>

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,appState& state);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
void createSwapChain(appState& state);
void cleanupSwapChain(appState& state);
void createImageViews(appState& state);
void createDepthResources(appState& state);
void createFramebuffers(appState& state);
void recreateSwapChain(appState& state);
#endif
