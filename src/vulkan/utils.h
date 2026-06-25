#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <array>
#include <optional>
#include <string>
#include <cstdio>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>



#include "../simulation/cudaCommons.h"
#include "../parser/parser_json.h"

#pragma region Config

extern bool OBJ_INSTANCING;

extern bool USE_OFF_SCREEN_RENDERING;

extern bool SHOW_GRID_BORDERS;

extern bool DEBUG_STATIC_PARTICLES;

extern std::vector<glm::vec3> DEBUG_PARTICLE_POSITIONS;

extern bool DEBUG_SHOW_SCALAR_FIELD;

#pragma endregion Config

#pragma region structs

struct SFInstance
{
    float px, py, pz;
    float r, g, b, a;
};

struct MCVertex
{
    float px, py, pz;
    float nx, ny, nz;
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getVertexBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static VkVertexInputBindingDescription getInstancingBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 1;
        bindingDescription.stride = sizeof(SFInstance); 
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getInstanceAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 1;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(SFInstance, px);

        attributeDescriptions[3].binding = 1;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(SFInstance, r);

        return attributeDescriptions;
    }
};

struct ParticleInstance
{
    glm::vec3 position;
};

extern std::vector<Vertex> vertices;
extern std::vector<uint16_t> indices;
extern std::vector<glm::vec3> particleInitialPositions;
extern std::vector<glm::vec3> particleBasePositions;

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct appState
{

    
    int sphereStacks = 8;
    int sphereSectors = 16;
    float sphereRadius = 0.08f;

    
    float maxDuration = 5.0f;

    
    std::vector<VkImage> offscreenImages;
    std::vector<VkDeviceMemory> offscreenImageMemories;
    std::vector<VkImageView> offscreenImageViews;

    std::vector<VkBuffer> readbackBuffers;
    std::vector<VkDeviceMemory> readbackBufferMemories;
    std::vector<void *> readbackBuffersMapped;
    bool readbackThisFrame = false;
    uint32_t lastRenderedSlot = 0;
    std::vector<bool> slotHasPendingReadback;

    uint32_t offscreenFrameIndex = 0;
    FILE *ffmpegPipe = nullptr;

    uint32_t videoFPS = 60;

    gridSize simGrid{100, 100, 100, 0.03f}; 
    int reconResolution =100;            
   
    gridSize reconGrid;
    float gridWorldSize = 0.0f; // derivato da simGri
    float reconBaseRadius = 0.14f;

    cudaStream_t reconStream = 0;

   
    float camTheta = 45.0f; 
    float camPhi = 60.0f;   
    float camRadius = 3.0f; 
    bool camDragging = false;
    double camLastX = 0.0, camLastY = 0.0;
    float camOrbitDegsPerSec = 20.0f;

    // app info
    bool framebufferResized = false;

    const int WIDTH = 1200;
    const int HEIGHT = 1000;

    int MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t currentFrame = 0;

    GLFWwindow *window;
    VkInstance instance;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkBuffer> instanceBuffers;
    std::vector<VkDeviceMemory> instanceBuffersMemory;
    std::vector<void *> instanceBuffersMapped;

    uint32_t particleCount;

    VkCommandPool commandPool;

    static constexpr uint32_t SF_MAX_INSTANCES = 5000000;
    uint32_t sfInstanceCount = 0;

    static constexpr uint32_t MC_MAX_VERTS = 10000000;
    std::vector<VkBuffer> mcVertexBuffers;
    std::vector<VkDeviceMemory> mcVertexBuffersMemory;
    std::vector<void *> mcVertexCudaPtr;
    uint32_t mcVertexCount = 0;

    VkPipeline graphicsPipeline;
    VkPipeline gridPipeline;
    VkBuffer gridVertexBuffer;
    VkDeviceMemory gridVertexBufferMemory;
    uint32_t gridVertexCount = 0;
    VkPipeline fluidPipeline;
    VkPipeline particlePipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkRenderPass renderPass;

    std::vector<VkImage> depthImages;
    std::vector<VkDeviceMemory> depthImageMemories;
    std::vector<VkImageView> depthImageViews;
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkExtent2D swapChainExtent;
    VkFormat swapChainImageFormat;
    std::vector<VkImageView> swapChainImageViews;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    // CUDA INTEROP
    std::vector<cudaExternalMemory_t> cudaExtMem;
    std::vector<void *> cudaMappedPtr;
};


void initAppState(appState &state, SimulationConfig &config);
void printAppState(const appState &state);
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        if (USE_OFF_SCREEN_RENDERING)
            return graphicsFamily.has_value();
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
#pragma endregion structs

#pragma region functions

std::vector<char> readFile(const std::string &filename);

void buildSphereMesh(appState &state);

#pragma endregion functions

#endif