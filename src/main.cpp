#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <optional>
#include <set>

#include <cstdint>   // Necessary for uint32_t
#include <limits>    // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp

#include <fstream>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp> //for matrices
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstdlib>    // system()
#include <sys/stat.h> // mkdir

// project modules
#include "vulkan/utils.h"
#include "vulkan/initVulkan.h"
#include "vulkan/buffer.h"
#include "vulkan/pipeline.h"
#include "vulkan/device.h"
#include "vulkan/swapChain.h"
#include "vulkan/offscreen.h"
#include <cstring>

// basciu
#include "simulation/ChronoCuda/ChronoCuda.h"
#include "simulation/cudaParticleSimulator/cudaParticleSimulator.h"
#include "simulation/cudaCommons.h"
#include "MeshReconstruction/scalarField.cuh"
#include "MeshReconstruction/marchingCubes.cuh"
#include "MeshReconstruction/neighborIndex.cuh"
#include "parser/parser_json.h"
#include "simulation/ChronoCuda/ChronoCuda.h"
#include "stats/stats.h"

// LIB PER FRAME GEN
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vulkan/stb_image_write.h"

SimulationConfig config = defaultConfig();
stats_fs_cuda stats = stats_fs_cuda();
appState state;
cudaParticleSimulator cPS;
ScalarFieldBuilder sfBuilder;
NeighborIndex nbrIndex; // acceleration grid, rebuilt each frame, feeds the WPCA passes
MarchingCubesExtractor mcExtractor;
std::vector<VkSemaphore> imageAvailableSemaphores;
std::vector<VkSemaphore> renderFinishedSemaphores;
std::vector<VkFence> inFlightFences;

FrameWriter frameWriter;

void createSyncObjects()
{
    imageAvailableSemaphores.resize(state.MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(state.MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(state.MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // crea una fence gia "segnalata" per superare il wait alla prima iterazione

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(state.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(state.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(state.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
        {

            throw std::runtime_error("failed to create synchronization objects for a frame!");
        }
    }
}

static bool RUN_NEIGHBOR_INDEX_SMOKE_TEST = true;

static void smokeTestNeighborIndex()
{
    const std::vector<glm::vec3> &src = DEBUG_PARTICLE_POSITIONS;
    int n = (int)src.size();
    if (n == 0)
    {
        printf("[NeighborIndex smoke] DEBUG_PARTICLE_POSITIONS empty — skipping\n");
        return;
    }

    // glm::vec3 -> float3 on the host, then upload. build() only cares that it
    // receives a device float3*, exactly as it would from a real solver buffer.
    std::vector<float3> host(n);
    glm::vec3 lo = src[0], hi = src[0];
    for (int i = 0; i < n; ++i)
    {
        host[i] = make_float3(src[i].x, src[i].y, src[i].z);
        lo = glm::min(lo, src[i]);
        hi = glm::max(hi, src[i]);
    }

    float3 *d_pos = nullptr;
    cudaMalloc(&d_pos, (size_t)n * sizeof(float3));
    cudaMemcpy(d_pos, host.data(), (size_t)n * sizeof(float3), cudaMemcpyHostToDevice);

    // Grid sized on a hypothetical reconstruction radius, independent of the sim grid.
    const float radius = 0.1f; // future Zhu-Bridson influence radius
    glm::vec3 pad(radius);     // one ring of slack around the AABB
    glm::vec3 origin = lo - pad;
    glm::vec3 extent = (hi + pad) - origin;
    int nx = (int)(extent.x / radius) + 2;
    int ny = (int)(extent.y / radius) + 2;
    int nz = (int)(extent.z / radius) + 2;

    NeighborIndex nbr(make_float3(origin.x, origin.y, origin.z),
                      make_int3(nx, ny, nz), radius, n);
    nbr.build(d_pos, n);
    nbr.debugValidate();
    printf("[NeighborIndex smoke] %d particles, grid %dx%dx%d, cell %.3f\n",
           n, nx, ny, nz, radius);

    cudaFree(d_pos);
}

void initCuda()
{
    state.simGrid = config.grid_size;
    state.reconGrid = gridSize::matchingDomain(state.simGrid, state.reconResolution);
    state.gridWorldSize = state.simGrid.worldSizeX();
    state.particleCount = config.numParticles;
    
    cudaStreamCreate(&state.reconStream);

    state.reconBaseRadius = config.mReconParams.reconRadiusInSpacings * state.simGrid.cellSize * config.spacing;
    cPS = cudaParticleSimulator(config, state.reconStream);
    cPS.setAppState(&state);
    cPS.setSphereRadius(state.sphereRadius);
    cPS.initParticles();
    if (DEBUG_STATIC_PARTICLES)
        cPS.loadStaticParticles(DEBUG_PARTICLE_POSITIONS);
    sfBuilder = ScalarFieldBuilder(state.reconGrid, state.reconBaseRadius, config.numParticles);

   
    {
        sfBuilder.setParams(config.mReconParams.reconParams);
        sfBuilder.setStream(state.reconStream);
        // Collega la maschera cellType della sim: il puntatore è stabile per tutta la
        // vita del simulatore e punta sempre ai dati aggiornati dall'ultimo p2g().
        // Abilita l'early-exit nel gatherFieldKernel per le celle sicuramente vuote.
        sfBuilder.setSimCellMask(cPS.getCellTypes(), cPS.getSimGrid());
    }

    
    {
        const float h = state.reconBaseRadius;
        const ReconParams rp = sfBuilder.getParams();
        // Supporto del kernel: baseline isotropico = h; il fallback per vicinati radi
        // lo allunga a h/kn (kn≤1), e scatta solo con l'anisotropico attivo.
        float support = h;
        if (rp.anisotropic && rp.kn > 0.f && rp.kn < 1.f)
            support = h / rp.kn;
        // Lo smoothing sposta una particella fino a λ·h dalla sua posizione di binning,
        // quindi il walk deve arrivare altrettanto più lontano per trovarla comunque.
        float disp = rp.smoothing ? rp.lambda * h : 0.f;
        float cell = (support + disp) * config.mReconParams.reconSupportScale; // cella ≥ reach → 3×3×3 la copre

        // +1 cella di margine così il dominio è coperto interamente (floor intero + ring).
        int nx = (int)(state.simGrid.worldSizeX() / cell) + 1;
        int ny = (int)(state.simGrid.worldSizeY() / cell) + 1;
        int nz = (int)(state.simGrid.worldSizeZ() / cell) + 1;
        nbrIndex = NeighborIndex(make_float3(0.f, 0.f, 0.f),
                                 make_int3(nx, ny, nz), cell, config.numParticles);
        nbrIndex.setStream(state.reconStream);
        printf("[NeighborIndex] domain grid %dx%dx%d, cell %.3f (h=%.3f, support=%.3f, disp=%.3f)\n",
               nx, ny, nz, cell, h, support, disp);
    }

    if (!OBJ_INSTANCING)
    {
        mcExtractor = MarchingCubesExtractor(appState::MC_MAX_VERTS, state.reconGrid.gridCells);
        mcExtractor.setStream(state.reconStream);
    }

    if (RUN_NEIGHBOR_INDEX_SMOKE_TEST)
        smokeTestNeighborIndex();
}
void initVulkan()
{
    printf("ciao\n");
    createInstance(state);
    createSurface(state);
    pickPhysicalDevice(state);
    createLogicalDevice(state);

    createSwapChain(state);
    createImageViews(state);

    createRenderPass(state);
    createDescriptorSetLayout(state);

    // createGraphicsPipeline(state);
    createInstancingPipeline(state);
    if (SHOW_GRID_BORDERS)
        createGridPipeline(state);

    // Off-screen resources must exist before createFramebuffers so the
    // framebuffers can reference the off-screen image views.
    if (USE_OFF_SCREEN_RENDERING)
    {
        createOffscreenResources(state);
    }

    createDepthResources(state);
    createFramebuffers(state);
    createCommandPool(state);
    createVertexBuffer(vertices, state);
    // questo da moficare
    std::vector<glm::vec3> cudaInitPos(config.numParticles, glm::vec3(0.0f));
    createInstanceBuffer(cudaInitPos, state);
    // init particle(state.InstanceBuffer)

    if (!OBJ_INSTANCING)
        createMCVertexBuffer(state);

    createIndexBuffer(indices, state);

    if (SHOW_GRID_BORDERS)
        createGridBuffers(state);

    createUniformBuffers(state);
    createDescriptorPool(state);
    createDescriptorSets(state);

    createCommandBuffer(state);

    createSyncObjects();
}

void drawFrame()
{
    static int counter = 0;
    counter++;
    if (USE_OFF_SCREEN_RENDERING)
    {
        

        vkWaitForFences(state.device, 1, &inFlightFences[state.currentFrame], VK_TRUE, UINT64_MAX);
        
        if (state.ffmpegPipe != nullptr && state.slotHasPendingReadback[state.currentFrame])
        {
            frameWriter.enqueue(state.readbackBuffersMapped[state.currentFrame]);
            state.slotHasPendingReadback[state.currentFrame] = false;
        }

        static chrono_cuda *sim = new chrono_cuda("sim", state.reconStream);
        static chrono_cuda *nbr = new chrono_cuda("nbr", state.reconStream);
        static chrono_cuda *sf = new chrono_cuda("sf", state.reconStream);
        static chrono_cuda *mc = new chrono_cuda("mc", state.reconStream);

        sim->cc_start();

        if (!DEBUG_STATIC_PARTICLES)
        {
            cPS.updateSystem();
        }
        sim->cc_stop_async();
        updateUniformBuffer(state.currentFrame, state);
        nbr->cc_start();
        nbrIndex.build(cPS.getParticlePositions(), cPS.getNumParticles());
        nbr->cc_stop_async();
        sf->cc_start();
        sfBuilder.build(nbrIndex.view());
        sf->cc_stop_async();
        mc->cc_start();
        if (!OBJ_INSTANCING)
        {
            int nv = mcExtractor.extract(sfBuilder.getField(), state.reconGrid, config.mReconParams.mcIsovalue);
            mcExtractor.copyToBuffer(state.mcVertexCudaPtr[state.currentFrame], nv);
            state.mcVertexCount = (uint32_t)nv;
        }
        else if (DEBUG_SHOW_SCALAR_FIELD)
            sfBuilder.fillInstanceBuffer(state.instanceBuffersMapped[state.currentFrame],
                                         state.sfInstanceCount, state.SF_MAX_INSTANCES);
        else
            sfBuilder.packParticleInstances(cPS.getParticlePositions(), cPS.getNumParticles(),
                                            state.instanceBuffersMapped[state.currentFrame]);
        mc->cc_stop_async();
        cudaStreamSynchronize(state.reconStream); 
        
        stats.addSimTime(sim->readElapsed());
        stats.addNbrTime(nbr->readElapsed());
        stats.addSfTime(sf->readElapsed());
        stats.addMcTime(mc->readElapsed());

        if (counter % 10 == 0)
        {
            std::cout << "Frame " << counter << " generated." << std::endl;
        }

        vkResetFences(state.device, 1, &inFlightFences[state.currentFrame]); 

        vkResetCommandBuffer(state.commandBuffers[state.currentFrame], 0);
        recordCommandBuffer(state.commandBuffers[state.currentFrame], 0, state);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &state.commandBuffers[state.currentFrame];

        if (vkQueueSubmit(state.graphicsQueue, 1, &submitInfo, inFlightFences[state.currentFrame]) != VK_SUCCESS)
            throw std::runtime_error("failed to submit off-screen command buffer!");

        state.lastRenderedSlot = state.currentFrame;

        if (state.ffmpegPipe != nullptr)
        {
            
            state.slotHasPendingReadback[state.currentFrame] = true;
            state.offscreenFrameIndex++;
        }
        else
        {
            
            vkWaitForFences(state.device, 1, &inFlightFences[state.currentFrame], VK_TRUE, UINT64_MAX);
        }

        state.currentFrame = (state.currentFrame + 1) % state.MAX_FRAMES_IN_FLIGHT;
        return;
    }


    vkWaitForFences(state.device, 1, &inFlightFences[state.currentFrame], VK_TRUE, UINT64_MAX);
    uint32_t imageIndex;
    // gets the image from swapchain
    VkResult result = vkAcquireNextImageKHR(state.device, state.swapChain, UINT64_MAX, imageAvailableSemaphores[state.currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR ||
        result == VK_SUBOPTIMAL_KHR ||
        state.framebufferResized)
    {
        state.framebufferResized = false;
        recreateSwapChain(state);
        return;
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if (!DEBUG_STATIC_PARTICLES)
        cPS.updateSystem();

    updateUniformBuffer(state.currentFrame, state);
    nbrIndex.build(cPS.getParticlePositions(), cPS.getNumParticles());
    sfBuilder.build(nbrIndex.view());
    if (!OBJ_INSTANCING)
    {
        int nv = mcExtractor.extract(sfBuilder.getField(), state.reconGrid, config.mReconParams.mcIsovalue);
        mcExtractor.copyToBuffer(state.mcVertexCudaPtr[state.currentFrame], nv);
        state.mcVertexCount = (uint32_t)nv;
    }
    else if (DEBUG_SHOW_SCALAR_FIELD)
        sfBuilder.fillInstanceBuffer(state.instanceBuffersMapped[state.currentFrame],
                                     state.sfInstanceCount, state.SF_MAX_INSTANCES);
    else
        sfBuilder.packParticleInstances(cPS.getParticlePositions(), cPS.getNumParticles(),
                                        state.instanceBuffersMapped[state.currentFrame]);
    
    cudaStreamSynchronize(state.reconStream);

    vkResetFences(state.device, 1, &inFlightFences[state.currentFrame]);

    vkResetCommandBuffer(state.commandBuffers[state.currentFrame], 0);
    recordCommandBuffer(state.commandBuffers[state.currentFrame], imageIndex, state);

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[state.currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &state.commandBuffers[state.currentFrame];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[state.currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(state.graphicsQueue, 1, &submitInfo, inFlightFences[state.currentFrame]) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkSwapchainKHR swapChains[] = {state.swapChain};

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    presentInfo.pResults = nullptr; // Optional

    result = vkQueuePresentKHR(state.presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
    {
        recreateSwapChain(state);
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("failed to present swap chain image!");
    }

    state.currentFrame = (state.currentFrame + 1) % state.MAX_FRAMES_IN_FLIGHT;
}

void simulation_data()
{
    int totalFrames = (int)(config.duration * config.videoFPS);
    
    chrono_cuda sim("sim", state.reconStream), nbr("nbr", state.reconStream),
        sf("sf", state.reconStream), mc("mc", state.reconStream);
    for (int i = 0; i < totalFrames; ++i)
    {
        sim.cc_start();
        cPS.updateSystem();
        sim.cc_stop_async();
        updateUniformBuffer(state.currentFrame, state);
        nbr.cc_start();
        nbrIndex.build(cPS.getParticlePositions(), cPS.getNumParticles());
        nbr.cc_stop_async();
        sf.cc_start();
        sfBuilder.build(nbrIndex.view());
        sf.cc_stop_async();
        mc.cc_start();
        int nv = mcExtractor.extract(sfBuilder.getField(), state.reconGrid, config.mReconParams.mcIsovalue);
        mcExtractor.copyToBuffer(state.mcVertexCudaPtr[state.currentFrame], nv);
        state.mcVertexCount = (uint32_t)nv;
        mc.cc_stop_async();
        cudaDeviceSynchronize(); 
        stats.addSimTime(sim.readElapsed());
        stats.addNbrTime(nbr.readElapsed());
        stats.addSfTime(sf.readElapsed());
        stats.addMcTime(mc.readElapsed());
    }
}

void render_video()
{
    std::cout << "Rendering video to " << config.output.name << " at " << config.videoFPS << " fps for " << config.duration << " seconds." << std::endl;

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -f rawvideo -pixel_format bgra "
             "-video_size %ux%u -framerate %u -i pipe:0 "
             "-c:v libx264 -preset %s -pix_fmt yuv420p %s",
             state.swapChainExtent.width, state.swapChainExtent.height, state.videoFPS,
             config.output.preset.c_str(), config.output.name.c_str());
    printf("[offscreen] opening pipe: %s\n", cmd);
    state.ffmpegPipe = popen(cmd, "w");
    if (!state.ffmpegPipe)
        throw std::runtime_error("failed to open ffmpeg pipe!");

    
    state.readbackThisFrame = true;

    frameWriter.start(state.ffmpegPipe,
                      (size_t)state.swapChainExtent.width * state.swapChainExtent.height * 4);

    uint32_t totalFrames = static_cast<uint32_t>(state.maxDuration * state.videoFPS);
    printf("max duration = %f, fps = %u, total frames = %u\n", state.maxDuration, state.videoFPS, totalFrames);
    while (state.offscreenFrameIndex < totalFrames)
    {
        drawFrame();
    }

    vkDeviceWaitIdle(state.device);

    uint32_t drainStart =
        state.offscreenFrameIndex >= (uint32_t)state.MAX_FRAMES_IN_FLIGHT
            ? state.offscreenFrameIndex - (uint32_t)state.MAX_FRAMES_IN_FLIGHT
            : 0;
    for (uint32_t f = drainStart; f < state.offscreenFrameIndex; f++)
    {
        uint32_t slot = f % state.MAX_FRAMES_IN_FLIGHT;
        if (state.slotHasPendingReadback[slot])
        {
            frameWriter.enqueue(state.readbackBuffersMapped[slot]);
            state.slotHasPendingReadback[slot] = false;
        }
    }

    frameWriter.stop(); 
    int ret = pclose(state.ffmpegPipe);
    state.ffmpegPipe = nullptr;
    if (ret != 0)
        fprintf(stderr, "[offscreen] ffmpeg exited with code %d — "
                        "make sure ffmpeg is installed.\n",
                ret);
    else
        printf("[offscreen] video saved to %s\n", config.output.name.c_str());
}

void render_images()
{
    printf("[render_images] readbackBuffers=%zu\n", state.readbackBuffers.size());
    printf("[render_images] offscreenImages size=%zu\n", state.offscreenImages.size());

    if (state.readbackBuffers.empty() || state.offscreenImages.empty())
    {
        printf("[render_images] ERROR: Resources not allocated correctly!\n");
        return;
    }

    mkdir("frames", 0755);
    printf("[render_images] saving frames to ./frames/\n");
    uint32_t totalFrames = static_cast<uint32_t>(state.maxDuration * state.videoFPS);
    printf("[render_images] total frames = %u, saving every 10\n", totalFrames);

    uint32_t w = state.swapChainExtent.width;
    uint32_t h = state.swapChainExtent.height;
    VkDeviceSize bufSize = (VkDeviceSize)w * h * 4;

    auto save_png_helper = [&](const std::string &suffix)
    {
        std::vector<uint8_t> pixels(bufSize);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(state.readbackBuffersMapped[state.lastRenderedSlot]);
        for (uint32_t i = 0; i < w * h; i++)
        {
            pixels[i * 4 + 0] = src[i * 4 + 2];
            pixels[i * 4 + 1] = src[i * 4 + 1];
            pixels[i * 4 + 2] = src[i * 4 + 0];
            pixels[i * 4 + 3] = 255;
        }

        char filename[128];
        snprintf(filename, sizeof(filename), "frames/frame_%05u_%s.png", state.offscreenFrameIndex, suffix.c_str());
        stbi_write_png(filename, (int)w, (int)h, 4, pixels.data(), (int)(w * 4));
        printf("[render_images] saved %s\n", filename);
    };

    while (state.offscreenFrameIndex < totalFrames)
    {
        glfwPollEvents();

        bool saveThisFrame = (state.offscreenFrameIndex % 10 == 0);
        
        state.readbackThisFrame = saveThisFrame;

        if (saveThisFrame)
        {
            drawFrame();
            save_png_helper("mesh");
        }
        else
        {
            OBJ_INSTANCING = false;
            drawFrame();
        }

        state.offscreenFrameIndex++;
    }

    printf("[render_images] done — process completed\n");
}
void mainLoop()
{
    std::cout << "Starting main loop with output type: " << config.output.output_type << std::endl;
    if (config.output.output_type == "simulation_data")
    {
        simulation_data();
    }
    else if (config.output.output_type == "video")
    {
        render_video();
    }
    else if (config.output.output_type == "images")
    {
        std::cout << "Rendering images to frames/ directory." << std::endl;
        render_images();
    }
}

void cleanup()
{
    cPS.clean();
    cudaStreamDestroy(state.reconStream);
    cleanupSwapChain(state);

    if (USE_OFF_SCREEN_RENDERING)
    {
        cleanupOffscreenResources(state);
    }

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroyBuffer(state.device, state.uniformBuffers[i], nullptr);
        vkFreeMemory(state.device, state.uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorSetLayout(state.device, state.descriptorSetLayout, nullptr);

    vkDestroyDescriptorPool(state.device, state.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(state.device, state.descriptorSetLayout, nullptr);

    vkDestroyBuffer(state.device, state.indexBuffer, nullptr);
    vkFreeMemory(state.device, state.indexBufferMemory, nullptr);

    vkDestroyBuffer(state.device, state.vertexBuffer, nullptr);
    vkFreeMemory(state.device, state.vertexBufferMemory, nullptr);

    for (size_t i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(state.device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(state.device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(state.device, inFlightFences[i], nullptr);

        vkDestroyBuffer(state.device, state.instanceBuffers[i], nullptr);
        vkFreeMemory(state.device, state.instanceBuffersMemory[i], nullptr);

        if (!OBJ_INSTANCING)
        {
            if (state.mcVertexCudaPtr[i])
                cudaFree(state.mcVertexCudaPtr[i]);
            if (state.cudaExtMem[i])
                cudaDestroyExternalMemory(state.cudaExtMem[i]);
            vkDestroyBuffer(state.device, state.mcVertexBuffers[i], nullptr);
            vkFreeMemory(state.device, state.mcVertexBuffersMemory[i], nullptr);
        }
    }

    vkDestroyCommandPool(state.device, state.commandPool, nullptr);
    
    vkDestroyPipeline(state.device, state.graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(state.device, state.pipelineLayout, nullptr);
    vkDestroyRenderPass(state.device, state.renderPass, nullptr);

    vkDestroyDevice(state.device, nullptr);
    
    if (!USE_OFF_SCREEN_RENDERING)
    {
        vkDestroySurfaceKHR(state.instance, state.surface, nullptr);
    }

    vkDestroyInstance(state.instance, nullptr);

    if (!USE_OFF_SCREEN_RENDERING)
    {
        glfwDestroyWindow(state.window);
        glfwTerminate();
    }
}

void checkArgv(int argc, char *argv[])
{
    if (argc > 1)
    {
        for (int i = 1; i < argc; i++)
        {
            std::string arg = argv[i];
            if (arg == "-c")
            {
                config = parseSimulationConfig(argv[i + 1]);
                USE_OFF_SCREEN_RENDERING = config.useOffScreenRendering;
                i++;
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << std::endl;
                std::cerr << "Usage: " << argv[0] << " [-c config.json]" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}

void appendJson()
{
    nlohmann::json output;
    output[config.configName]["arch"] = "CUDA";
    output[config.configName]["particles"] = config.numParticles;
    output[config.configName]["grid"] = {config.grid_size.x, config.grid_size.y, config.grid_size.z};
    output[config.configName]["cell_size"] = config.grid_size.cellSize;
    output[config.configName]["dt"] = cPS.getDeltaTime();
    output[config.configName]["duration"] = config.duration;
    output[config.configName]["fps"] = config.videoFPS;
    output[config.configName]["total_frames"] = (int)(config.duration * config.videoFPS);

    float invFPS = 1.0f / config.videoFPS;
    output[config.configName]["sub_steps"] = std::ceil(invFPS / cPS.getDeltaTime());
    output[config.configName]["gravity"] = -9.81f;
    output[config.configName]["flip_ratio"] = config.fluidProps.flipRatio;
    output[config.configName]["density"] = config.fluidProps.density;
    output[config.configName]["pressure_iters"] = config.num_iterations;
    output[config.configName]["average_simulation_ms"] = stats.getSimTime();
    float reconTime = stats.getNbrTime() + stats.getSfTime() + stats.getMcTime();
    output[config.configName]["average_reconstruction_ms"] = reconTime;
    output[config.configName]["average_total_frame_ms"] = stats.getTotalTime() / stats.getNumberOfFrames();
    std::ofstream outputFile(config.outputJson, std::ios::app);
    if (outputFile.is_open())
    {
        outputFile << output.dump() << "\n";
        outputFile.close();
    }
}


int main(int argc, char *argv[])
{
    try
    {
        chrono_cuda t1("Total execution time"), bsm("buildSphereMesh"), initWin("initWindow"), initCudaTime("initCuda"), initVulkanTime("initVulkan");
        t1.cc_start();
        checkArgv(argc, argv);
        initAppState(state, config);

        bsm.cc_start();
        buildSphereMesh(state);
        bsm.cc_stop(true);

        initWin.cc_start();
        initWindow(state);
        initWin.cc_stop(true);

        initCudaTime.cc_start();
        initCuda();
        initCudaTime.cc_stop(true);

        initVulkanTime.cc_start();
        initVulkan();
        initVulkanTime.cc_stop(true);

        mainLoop();
        cleanup();
        t1.cc_stop(false);

        stats.setNumberOfFrames(config.duration * config.videoFPS);
        stats.calculateAvgTimes();
        appendJson();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}