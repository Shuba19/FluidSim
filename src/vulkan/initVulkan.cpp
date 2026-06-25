#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <optional>
#include <set>

#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp

#include <fstream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp> //for matrices
#include <glm/gtc/matrix_transform.hpp>




#include <array>

#include "initVulkan.h"
#include "utils.h"

static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto state = reinterpret_cast<appState*>(glfwGetWindowUserPointer(window));
    if (state) state->framebufferResized = true;
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    auto state = reinterpret_cast<appState*>(glfwGetWindowUserPointer(window));
    if (!state) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            state->camDragging = true;
            glfwGetCursorPos(window, &state->camLastX, &state->camLastY);
        } else {
            state->camDragging = false;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto state = reinterpret_cast<appState*>(glfwGetWindowUserPointer(window));
    if (!state || !state->camDragging) return;
    double dx = xpos - state->camLastX;
    double dy = ypos - state->camLastY;
    state->camLastX = xpos;
    state->camLastY = ypos;
    state->camTheta -= (float)dx * 0.3f;
    state->camPhi   -= (float)dy * 0.3f;
    state->camPhi = glm::clamp(state->camPhi, 1.0f, 179.0f);
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/, double yoffset) {
    auto state = reinterpret_cast<appState*>(glfwGetWindowUserPointer(window));
    if (!state) return;
    state->camRadius -= (float)yoffset * 0.2f;
    state->camRadius = glm::max(state->camRadius, 0.5f);
}

void initWindow(appState & state) {
    if (USE_OFF_SCREEN_RENDERING) return;
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    state.window = glfwCreateWindow(state.WIDTH, state.HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(state.window, &state);
    glfwSetFramebufferSizeCallback(state.window, framebufferResizeCallback);
    glfwSetMouseButtonCallback(state.window, mouseButtonCallback);
    glfwSetCursorPosCallback(state.window, cursorPosCallback);
    glfwSetScrollCallback(state.window, scrollCallback);
}

void createInstance(appState & state) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    if (!USE_OFF_SCREEN_RENDERING) {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
    } else {
        createInfo.enabledExtensionCount = 0;
        createInfo.ppEnabledExtensionNames = nullptr;
    }
    createInfo.enabledLayerCount = 0;

    if (vkCreateInstance(&createInfo, nullptr, &state.instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void createSurface(appState & state)
{
    if (USE_OFF_SCREEN_RENDERING) {
        state.surface = VK_NULL_HANDLE;
        return;
    }
    if (glfwCreateWindowSurface(state.instance, state.window, nullptr, &state.surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

