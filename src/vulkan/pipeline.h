#ifndef PIPELINE_H
#define PIPELINE_H

#include "utils.h"

std::vector<char> readFile(const std::string& filename);
VkShaderModule createShaderModule(const std::vector<char>& code, appState& state);
void createGraphicsPipeline(appState& state);
void createInstancingPipeline(appState& state);
void createGridPipeline(appState& state);
void createRenderPass(appState& state);
void createDescriptorSetLayout(appState& state);
void createDescriptorPool(appState & state);
void createDescriptorSets(appState& state);

#endif
