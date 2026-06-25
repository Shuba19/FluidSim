#include "offscreen.h"
#include "buffer.h"   // findMemoryType, createBuffer

#include <stdexcept>
#include <cstring>
#include <cstdio>

// ---------------------------------------------------------------------------
static void createImage(uint32_t width, uint32_t height,
                        VkFormat format,
                        VkImageTiling tiling,
                        VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties,
                        VkImage& image,
                        VkDeviceMemory& imageMemory,
                        appState& state)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width  = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth  = 1;
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.format        = format;
    imageInfo.tiling        = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage         = usage;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(state.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("failed to create off-screen image!");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(state.device, image, &memReq);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties, state);

    if (vkAllocateMemory(state.device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate off-screen image memory!");

    vkBindImageMemory(state.device, image, imageMemory, 0);
}

void createOffscreenResources(appState& state)
{
    uint32_t w = state.swapChainExtent.width;
    uint32_t h = state.swapChainExtent.height;

    VkFormat fmt = state.swapChainImageFormat;

    state.offscreenImages.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.offscreenImageMemories.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.offscreenImageViews.resize(state.MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++) {
        createImage(
            w, h, fmt,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            state.offscreenImages[i],
            state.offscreenImageMemories[i],
            state
        );

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image                           = state.offscreenImages[i];
        viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format                          = fmt;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(state.device, &viewInfo, nullptr, &state.offscreenImageViews[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create off-screen image view!");
    }

    VkDeviceSize bufSize = (VkDeviceSize)w * h * 4;

    state.readbackBuffers.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.readbackBufferMemories.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.readbackBuffersMapped.resize(state.MAX_FRAMES_IN_FLIGHT);
    state.slotHasPendingReadback.assign(state.MAX_FRAMES_IN_FLIGHT, false);
    for (int i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++)
    {
        createBuffer(
            bufSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            state.readbackBuffers[i],
            state.readbackBufferMemories[i],
            state);
        vkMapMemory(state.device, state.readbackBufferMemories[i], 0, bufSize, 0, &state.readbackBuffersMapped[i]);
    }
}

void cleanupOffscreenResources(appState& state)
{
    for (size_t i = 0; i < state.readbackBuffers.size(); i++) {
        if (state.readbackBuffersMapped[i] != nullptr)
            vkUnmapMemory(state.device, state.readbackBufferMemories[i]);
        vkDestroyBuffer(state.device, state.readbackBuffers[i], nullptr);
        vkFreeMemory(state.device, state.readbackBufferMemories[i], nullptr);
    }
    state.readbackBuffersMapped.clear();

    for (int i = 0; i < state.MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyImageView(state.device, state.offscreenImageViews[i], nullptr);
        vkDestroyImage(state.device, state.offscreenImages[i], nullptr);
        vkFreeMemory(state.device, state.offscreenImageMemories[i], nullptr);
    }
}

void FrameWriter::start(FILE* pipe, size_t frameBytes, size_t maxQueued)
{
    pipe_       = pipe;
    frameBytes_ = frameBytes;
    maxQueued_  = maxQueued > 0 ? maxQueued : 1;
    stopping_   = false;
    error_      = false;
    running_    = true;
    worker_     = std::thread(&FrameWriter::run, this);
}

void FrameWriter::enqueue(const void* src)
{
    std::vector<uint8_t> buf;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cvNotFull_.wait(lock, [this] { return ready_.size() < maxQueued_ || stopping_; });
        if (stopping_)
            return;
        if (!pool_.empty())
        {
            buf = std::move(pool_.back());
            pool_.pop_back();
        }
    }
    buf.resize(frameBytes_);
    std::memcpy(buf.data(), src, frameBytes_);
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ready_.push(std::move(buf));
    }
    cvNotEmpty_.notify_one();
}

void FrameWriter::run()
{
    while (true)
    {
        std::vector<uint8_t> buf;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cvNotEmpty_.wait(lock, [this] { return !ready_.empty() || stopping_; });
            if (ready_.empty())
            {
                if (stopping_)
                    break;
                continue;
            }
            buf = std::move(ready_.front());
            ready_.pop();
        }
        cvNotFull_.notify_one(); 
        if (fwrite(buf.data(), 1, frameBytes_, pipe_) != frameBytes_)
            error_ = true;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            pool_.push_back(std::move(buf)); // recycle
        }
    }
}

void FrameWriter::stop()
{
    if (!running_)
        return;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopping_ = true;
    }
    cvNotEmpty_.notify_all();
    cvNotFull_.notify_all();
    if (worker_.joinable())
        worker_.join();
    running_ = false;
    if (error_)
        fprintf(stderr, "[offscreen] frame writer: short write to ffmpeg pipe "
                        "(encoder may have failed).\n");
}

FrameWriter::~FrameWriter()
{
    stop();
}
