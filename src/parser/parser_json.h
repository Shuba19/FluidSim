#ifndef PARSER_JSON_H
#define PARSER_JSON_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include "../simulation/cudaCommons.h"
struct SimulationConfig
{
    float duration, videoFPS;
    int maxFramesInFlight;
    bool useOffScreenRendering = true, objInstancing = true; 
    std::string outputJson = "outputStats.json"; // Default output file for simulation data
    std::string configName = "default_config"; // Default configuration name
    // PARTICLE SIMULTOR
    float deltaTime;
    fluidProperties fluidProps;
    int  deviceId, num_iterations;
    int numParticles;
    environment env;
    gridSize grid_size;

    float particleSpacing, spacing, radius_percentage, jitter;

    // SPHERE RESOLUTION
    int sphereStacks, sphereSectors;
    struct output
    {
        std::string output_type; 
        std::string name;       
        std::string preset = "veryfast"; 
    } output;
    // MESH RECONSTRUCTION
    struct meshReconstructionParams
    {
        float reconRadiusInSpacings = 15.56f; 
        float reconSupportScale = 1.0;
        float mcIsovalue = 2.5f; 
        ReconParams reconParams;
    } mReconParams;

    int reconResolution;

    struct framesInFlight
    {
        bool useDoubleBuffering = true;
        int maxFramesInFlight = 2;
    } framesInFlight;
};

SimulationConfig defaultConfig();
SimulationConfig parseSimulationConfig(const std::string &filename);
nlohmann::json parseSimulationConfigToJson(SimulationConfig &config);
#endif // PARSER_JSON_H