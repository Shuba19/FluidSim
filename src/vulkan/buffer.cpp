

#include "buffer.h"
#include "utils.h"
#include "device.h"
#include <chrono>
#include <cstring>
#include <cmath>
#include <array>

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, appState &state)
{
    
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(state.physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        
        if ((typeFilter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory, appState &state)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(state.device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(state.device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties, state);

    if (vkAllocateMemory(state.device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(state.device, buffer, bufferMemory, 0);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size, appState &state)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = state.commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VkResult r = vkAllocateCommandBuffers(state.device, &allocInfo, &commandBuffer);
    if (r != VK_SUCCESS)
        throw std::runtime_error("alloc failed");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult r2 = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (r2 != VK_SUCCESS)
        throw std::runtime_error("begin failed");

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;

    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    VkResult r3 = vkEndCommandBuffer(commandBuffer);
    if (r3 != VK_SUCCESS)
        throw std::runtime_error("end failed");

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(state.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(state.graphicsQueue);

    vkFreeCommandBuffers(state.device, state.commandPool, 1, &commandBuffer);
}

void createIndexBuffer(std::vector<uint16_t> indices, appState &state)
{
    VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, state);

    void *data;
    vkMapMemory(state.device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, indices.data(), (size_t)bufferSize);
    vkUnmapMemory(state.device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, state.indexBuffer, state.indexBufferMemory, state);

    copyBuffer(stagingBuffer, state.indexBuffer, bufferSize, state);

    vkDestroyBuffer(state.device, stagingBuffer, nullptr);
    vkFreeMemory(state.device, stagingBufferMemory, nullptr);
}

void createVertexBuffer(std::vector<Vertex> vertices, appState &state)
{
    VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
   
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory, state);

    void *data;
    vkMapMemory(state.device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, vertices.data(), (size_t)bufferSize);
    vkUnmapMemory(state.device, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, state.vertexBuffer, state.vertexBufferMemory, state);

    copyBuffer(stagingBuffer, state.vertexBuffer, bufferSize, state);

    vkDestroyBuffer(state.device, stagingBuffer, nullptr);
    vkFreeMemory(state.device, stagingBufferMemory, nullptr);
}


void createInstanceBuffer(std::vector<glm::vec3> positions, appState &state)
{
    VkDeviceSize bufferSize = sizeof(SFInstance) * positions.size();

    state.instanceBuffers.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.instanceBuffersMemory.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.instanceBuffersMapped.resize(state.MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        createBuffer(bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, state.instanceBuffers[i], state.instanceBuffersMemory[i], state);
        vkMapMemory(state.device, state.instanceBuffersMemory[i], 0, bufferSize, 0, &state.instanceBuffersMapped[i]);
    }
}

static void createInteropVertexBuffer(appState &state, VkDeviceSize size,
                                      VkBuffer &buffer, VkDeviceMemory &memory,
                                      cudaExternalMemory_t &extMem, void *&cudaPtr)
{
    // 1) Buffer flagged exportable.
    VkExternalMemoryBufferCreateInfo extBufInfo{};
    extBufInfo.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.pNext       = &extBufInfo;
    bufInfo.size        = size;
    bufInfo.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(state.device, &bufInfo, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create interop vertex buffer!");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(state.device, buffer, &memReq);

    VkMemoryDedicatedAllocateInfo dedInfo{};
    dedInfo.sType  = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedInfo.buffer = buffer;

    VkExportMemoryAllocateInfo expInfo{};
    expInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    expInfo.pNext       = &dedInfo;
    expInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext           = &expInfo;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, state);
    if (vkAllocateMemory(state.device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate interop vertex buffer memory!");
    vkBindBufferMemory(state.device, buffer, memory, 0);

    auto pfnGetMemoryFdKHR =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(state.device, "vkGetMemoryFdKHR");
    if (!pfnGetMemoryFdKHR)
        throw std::runtime_error("vkGetMemoryFdKHR unavailable (VK_KHR_external_memory_fd)!");

    VkMemoryGetFdInfoKHR fdInfo{};
    fdInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    fdInfo.memory     = memory;
    fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd = -1;
    if (pfnGetMemoryFdKHR(state.device, &fdInfo, &fd) != VK_SUCCESS)
        throw std::runtime_error("vkGetMemoryFdKHR failed!");

    cudaExternalMemoryHandleDesc memDesc{};
    memDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    memDesc.handle.fd = fd;
    memDesc.size      = memReq.size;
    std::cout << "createInteropVertexBuffer: size = " << size << std::endl;
    std::cout << "createInteropVertexBuffer: memReq.size = " << memReq.size << std::endl;
    std::cout << "createInteropVertexBuffer: memDesc.size = " << memDesc.size << std::endl;
    std::cout << "createInteropVertexBuffer: fd = " << fd << std::endl;
    std::cout << "createInteropVertexBuffer: extMem = " << extMem << std::endl;
    if (cudaImportExternalMemory(&extMem, &memDesc) != cudaSuccess)
        throw std::runtime_error("cudaImportExternalMemory failed!");

    cudaExternalMemoryBufferDesc bufDesc{};
    bufDesc.offset = 0;
    bufDesc.size   = size;
    bufDesc.flags  = 0;
    if (cudaExternalMemoryGetMappedBuffer(&cudaPtr, extMem, &bufDesc) != cudaSuccess)
        throw std::runtime_error("cudaExternalMemoryGetMappedBuffer failed!");
}

void createMCVertexBuffer(appState &state)
{
    VkDeviceSize bufferSize = sizeof(MCVertex) * state.MC_MAX_VERTS;

    state.mcVertexBuffers.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.mcVertexBuffersMemory.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.mcVertexCudaPtr.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.cudaExtMem.resize(state.MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        createInteropVertexBuffer(state, bufferSize,
                                  state.mcVertexBuffers[i], state.mcVertexBuffersMemory[i],
                                  state.cudaExtMem[i], state.mcVertexCudaPtr[i]);
    }
}


void updateInstanceBuffer(uint32_t currentImage, appState &state)
{
    static auto startTime = std::chrono::high_resolution_clock::now();
    float time = USE_OFF_SCREEN_RENDERING
        ? static_cast<float>(state.offscreenFrameIndex) / state.videoFPS
        : std::chrono::duration<float, std::chrono::seconds::period>(
              std::chrono::high_resolution_clock::now() - startTime).count();

  
    float t = (sin(time) + 1.0f) / 2.0f;
}
void createCommandPool(appState &state)
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(state);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // usato per buffer più duraturi e persistenti
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(state.device, &poolInfo, nullptr, &state.commandPool) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create command pool!");
    }
}

void createCommandBuffer(appState &state)
{
    state.commandBuffers.resize(state.MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = state.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)state.commandBuffers.size();

    if (vkAllocateCommandBuffers(state.device, &allocInfo, state.commandBuffers.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, appState &state)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0;                  // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional usefull for secondary buffer to inherit stuff

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = state.renderPass;
    renderPassInfo.framebuffer = USE_OFF_SCREEN_RENDERING
                                     ? state.swapChainFramebuffers[state.currentFrame]
                                     : state.swapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = state.swapChainExtent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data(); 

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE); 

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(state.swapChainExtent.width);
    viewport.height = static_cast<float>(state.swapChainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = state.swapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // bind della graphic pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, state.graphicsPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipelineLayout, 0, 1, &state.descriptorSets[state.currentFrame], 0, nullptr);

    if (OBJ_INSTANCING)
    {
        VkBuffer vertexBuffers[] = {state.vertexBuffer, state.instanceBuffers[state.currentFrame]};
        VkDeviceSize offsets[] = {0, 0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, state.indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        uint32_t instanceCount = DEBUG_SHOW_SCALAR_FIELD ? state.sfInstanceCount : state.particleCount;
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), instanceCount, 0, 0, 0);
    }
    else
    {
        if (state.mcVertexCount > 0)
        {
            VkBuffer vertexBuffers[] = {state.mcVertexBuffers[state.currentFrame]};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdDraw(commandBuffer, state.mcVertexCount, 1, 0, 0);
        }
    }

    if (SHOW_GRID_BORDERS) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, state.gridPipeline);
        VkBuffer gridBuf[] = {state.gridVertexBuffer};
        VkDeviceSize gridOff[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, gridBuf, gridOff);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, state.pipelineLayout, 0, 1, &state.descriptorSets[state.currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffer, state.gridVertexCount, 1, 0, 0);
    }

    vkCmdEndRenderPass(commandBuffer);

    if (USE_OFF_SCREEN_RENDERING && state.readbackThisFrame)
    {
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.bufferRowLength   = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {state.swapChainExtent.width, state.swapChainExtent.height, 1};

        vkCmdCopyImageToBuffer(commandBuffer,
            state.offscreenImages[state.currentFrame],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            state.readbackBuffers[state.currentFrame],
            1, &region);
    }

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void createGridBuffers(appState& state) {
    float s = state.gridWorldSize;
    
    float verts[] = {
        0,0,0, s,0,0,   s,0,0, s,s,0,   s,s,0, 0,s,0,   0,s,0, 0,0,0, // bottom face
        0,0,s, s,0,s,   s,0,s, s,s,s,   s,s,s, 0,s,s,   0,s,s, 0,0,s, // top face
        0,0,0, 0,0,s,   s,0,0, s,0,s,   s,s,0, s,s,s,   0,s,0, 0,s,s  // vertical edges
    };
    state.gridVertexCount = 24;
    VkDeviceSize size = sizeof(verts);
    createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 state.gridVertexBuffer, state.gridVertexBufferMemory, state);
    void* data;
    vkMapMemory(state.device, state.gridVertexBufferMemory, 0, size, 0, &data);
    memcpy(data, verts, size);
    vkUnmapMemory(state.device, state.gridVertexBufferMemory);
}

void createUniformBuffers(appState &state)
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    state.uniformBuffers.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.uniformBuffersMemory.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.uniformBuffersMapped.resize(state.MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, state.uniformBuffers[i], state.uniformBuffersMemory[i], state);

        vkMapMemory(state.device, state.uniformBuffersMemory[i], 0, bufferSize, 0, &state.uniformBuffersMapped[i]);
    }
}

void updateUniformBuffer(uint32_t currentImage, appState& state) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    float time = USE_OFF_SCREEN_RENDERING
        ? static_cast<float>(state.offscreenFrameIndex) / state.videoFPS
        : std::chrono::duration<float, std::chrono::seconds::period>(
            std::chrono::high_resolution_clock::now() - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);

    float w    = state.gridWorldSize;
    float half = w * 0.5f;
    glm::vec3 gridCenter(half, half, half);

    float autoRadius = w * 2.0f;

    float theta = glm::radians(state.camTheta);
    float phi   = glm::radians(state.camPhi);

    glm::vec3 eye = gridCenter + autoRadius * glm::vec3(
        std::sin(phi) * std::sin(theta),
        std::cos(phi),
        std::sin(phi) * std::cos(theta));

    ubo.view = glm::lookAt(eye, gridCenter, glm::vec3(0.0f, 1.0f, 0.0f));

    ubo.proj = glm::perspective(
        glm::radians(60.0f),
        state.swapChainExtent.width / (float)state.swapChainExtent.height,
        w * 0.001f,   
        w * 30.0f     
    );
    ubo.proj[1][1] *= -1;

    memcpy(state.uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}