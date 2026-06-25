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
#include <cmath>
#include <random>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp> 
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include "utils.h"

#include "utils.h"

#pragma region Config
bool OBJ_INSTANCING = false; 
bool USE_OFF_SCREEN_RENDERING = true;
bool SHOW_GRID_BORDERS = true;
bool DEBUG_STATIC_PARTICLES = false;
bool DEBUG_SHOW_SCALAR_FIELD = false;
#pragma endregion Config

void initAppState(appState &state, SimulationConfig &config)
{
    state.sphereStacks = config.sphereStacks;
    state.sphereSectors = config.sphereSectors;
    state.simGrid = config.grid_size;
    state.reconResolution = config.reconResolution;
    state.sphereRadius = config.grid_size.cellSize / 0.5f;
    state.particleCount = config.numParticles;
    state.maxDuration = config.duration;
    state.videoFPS = config.videoFPS;
    state.MAX_FRAMES_IN_FLIGHT = config.maxFramesInFlight;
    OBJ_INSTANCING = config.objInstancing;
    state.reconGrid = gridSize::matchingDomain(state.simGrid, state.reconResolution);
    state.gridWorldSize = state.simGrid.worldSizeX(); // derivato da simGrid
    state.camTheta = 45.0f;
    state.camPhi = 60.0f;
    state.camRadius = 3.0f;
    state.camDragging = false;
    state.camLastX = 0.0;
    state.camLastY = 0.0;
    state.camOrbitDegsPerSec = 20.0f;
    printAppState(state); 
}

void printAppState(const appState &state)
{
    std::cout << "App State:" << std::endl;
    std::cout << "Sphere Stacks: " << state.sphereStacks << std::endl;
    std::cout << "Sphere Sectors: " << state.sphereSectors << std::endl;
    std::cout << "Simulation Grid: (" << state.simGrid.x << ", " << state.simGrid.y << ", " << state.simGrid.z << ") with cell size " << state.simGrid.cellSize << std::endl;
    std::cout << "Reconstruction Resolution: " << state.reconResolution << std::endl;
    std::cout << "Sphere Radius: " << state.sphereRadius << std::endl;
    std::cout << "Particle Count: " << state.particleCount << std::endl;
    std::cout << "Max Duration: " << state.maxDuration << " seconds" << std::endl;
    std::cout << "Video FPS: " << state.videoFPS << std::endl;
}

static std::vector<glm::vec3> generateDebugSplash()
{
    const float spacing = 0.07f;
    const glm::vec2 center(1.5f, 1.5f);
    std::vector<glm::vec3> pts;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> jitter(-spacing * 0.3f, spacing * 0.3f);

    for (float x = 0.15f; x <= 2.85f; x += spacing)
        for (float z = 0.15f; z <= 2.85f; z += spacing)
        {
            float dx = x - center.x, dz = z - center.y;
            float r = std::sqrt(dx * dx + dz * dz);

            float h = 0.25f;
            h -= 0.18f * std::exp(-std::pow(r / 0.25f, 2.f));   
            h += 0.04f * std::sin(x * 9.f) * std::cos(z * 8.3f); 
            h += 0.025f * std::sin(x * 17.f + z * 13.f);
            h = std::max(h, 0.06f);

            for (float y = 0.06f; y <= h; y += spacing)
                pts.push_back({x + jitter(rng), y + jitter(rng), z + jitter(rng)});
        }

    std::uniform_real_distribution<float> rx(0.3f, 2.7f), ry(0.28f, 0.55f), rz(0.3f, 2.7f);
    for (int i = 0; i < 20; ++i)
        pts.push_back({rx(rng), ry(rng), rz(rng)});

    return pts;
}

std::vector<glm::vec3> DEBUG_PARTICLE_POSITIONS = generateDebugSplash();



std::vector<glm::vec3> particleInitialPositions = {
    {0.0f, 0.0f, 0.0f},
    {3.050902f, 7.100272f, 4.852314f},
    {0.0f, 1.0f, 2.0f},
    {1.0f, 1.0f, 3.0f},
};

std::vector<glm::vec3> particleBasePositions = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 2.0f},
    {1.0f, 1.0f, 3.0f},
};

static std::vector<Vertex> makeSphereVertices(float r, int stacks, int sectors)
{
    std::vector<Vertex> verts;
    for (int i = 0; i <= stacks; i++)
    {
        float theta = (float)i / stacks * M_PI; // 0 .. PI
        for (int j = 0; j <= sectors; j++)
        {
            float phi = (float)j / sectors * 2.0f * M_PI; // 0 .. 2PI
            glm::vec3 n = {
                sinf(theta) * cosf(phi),
                sinf(theta) * sinf(phi),
                cosf(theta)};
            Vertex v;
            v.pos = n * r;
            v.color = (n + glm::vec3(1.0f)) * 0.5f; // normal → [0,1] color
            verts.push_back(v);
        }
    }
    return verts;
}

static std::vector<uint16_t> makeSphereIndices(int stacks, int sectors)
{
    std::vector<uint16_t> idx;
    for (int i = 0; i < stacks; i++)
    {
        for (int j = 0; j < sectors; j++)
        {
            uint16_t v0 = (uint16_t)(i * (sectors + 1) + j);
            uint16_t v1 = (uint16_t)(i * (sectors + 1) + j + 1);
            uint16_t v2 = (uint16_t)((i + 1) * (sectors + 1) + j);
            uint16_t v3 = (uint16_t)((i + 1) * (sectors + 1) + j + 1);
            idx.insert(idx.end(), {v0, v2, v1, v1, v2, v3});
        }
    }
    return idx;
}

std::vector<Vertex> vertices;
std::vector<uint16_t> indices;

void buildSphereMesh(appState &state)
{
    vertices = makeSphereVertices(state.sphereRadius, state.sphereStacks, state.sphereSectors);
    indices = makeSphereIndices(state.sphereStacks, state.sphereSectors);
}

std::vector<char> readFile(const std::string &filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}
