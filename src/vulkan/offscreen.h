#pragma once

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include "utils.h"

void createOffscreenResources(appState& state);
void cleanupOffscreenResources(appState& state);

class FrameWriter
{
public:
    FrameWriter() = default;
    ~FrameWriter();
    FrameWriter(const FrameWriter&) = delete;
    FrameWriter& operator=(const FrameWriter&) = delete;

    void start(FILE* pipe, size_t frameBytes, size_t maxQueued = 8);
    void enqueue(const void* src);
    void stop();

private:
    void run();

    FILE*  pipe_       = nullptr;
    size_t frameBytes_ = 0;
    size_t maxQueued_  = 8;
    bool   running_    = false;
    bool   stopping_   = false;
    bool   error_      = false;

    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cvNotFull_;
    std::condition_variable cvNotEmpty_;
    std::queue<std::vector<uint8_t>> ready_;
    std::vector<std::vector<uint8_t>> pool_; 
};
