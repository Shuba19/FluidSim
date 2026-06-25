// #pragma once
#ifndef INIT_VULKAN_H
#define INIT_VULKAN_H

struct appState;
void initWindow(appState& state);
void initVulkan();
void createInstance(appState & state);
void createSurface(appState & state);

#endif