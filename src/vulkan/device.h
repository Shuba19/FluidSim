#ifndef DEVICE_H
#define DEVICE_H
struct appState;
struct QueueFamilyIndices;
struct SwapChainSupportDetails;

QueueFamilyIndices findQueueFamilies(appState & state) ;
SwapChainSupportDetails querySwapChainSupport(appState & state);
bool checkDeviceExtensionSupport(appState & state);
bool isDeviceSuitable(appState & state);
void pickPhysicalDevice(appState & state);
void createLogicalDevice(appState & state);

#endif