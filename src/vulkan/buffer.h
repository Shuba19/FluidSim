#pragma once

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
#include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp>

#include "utils.h"



uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, appState & state);
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, appState & state);
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, appState & state);
void createIndexBuffer(std::vector<uint16_t> indices, appState & state);
void createVertexBuffer(std::vector<Vertex> vertices, appState & state);
void createInstanceBuffer(std::vector<glm::vec3> positions, appState& state);
void createMCVertexBuffer(appState& state);
void updateInstanceBuffer(uint32_t currentImage, appState & state); 
void createCommandPool(appState & state);
void createCommandBuffer(appState & state);
void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, appState & state) ;
void createGridBuffers(appState& state);
void createUniformBuffers(appState & state);
void updateUniformBuffer(uint32_t currentImage, appState & state);