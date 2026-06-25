#include "parser_json.h"

SimulationConfig defaultConfig()
{
    SimulationConfig config;
    config.duration = 5.0f;                                     // Default duration in seconds
    config.videoFPS = 60.0f;                                    // Default frames per second
    config.maxFramesInFlight = 2;                               // Default max frames in flight
    config.sphereStacks = 8;                                    // Default sphere stacks
    config.sphereSectors = 16;                                  // Default sphere sectors
    config.objInstancing = false;                               // Default to marching cubes surface mesh
    config.reconResolution = 100;                               // Default reconstruction resolution
    config.useOffScreenRendering = true;                        // Default to off-screen rendering
    config.deltaTime = 0.001f;                                  // Default time step (s)
    config.fluidProps.density = 1000.0f;                        // Default fluid density (kg/m^3)
    config.fluidProps.viscosity = 0.00001f;                     // Default fluid viscosity (Pa·s)
    config.fluidProps.flipRatio = 0.95f;                        // Default FLIP ratio
    config.fluidProps.pressureSolverType = "multigrid";         // Default pressure solver type
    config.fluidProps.useNarrowBand = false;                    // Default to using narrow band
    config.numParticles = 635255;                               // Default number of particles
    config.env.externalForce = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // Default external force (x, y, z) and time range (start, end)
    config.env.numberOfSphere = 1;                              // Default number of spheres
    config.env.initialVelocities.push_back({0.0f, 0.0f, 0.0f}); // Default initial velocities (empty)
    config.deviceId = 0;                                        // Default GPU device ID
    config.num_iterations = 10;                                 // Default number of iterations per frame
    config.grid_size = {100, 100, 100, 0.04f};                  // Default grid size and cell size
    config.particleSpacing = 0.08f;                             // Default particle spacing
    config.spacing = 0.04f;                                     // Default particle spacing
    config.radius_percentage = 0.4f;                            // Default radius percentage
    config.jitter = 0.1f;                                       // Default jitter
    config.output.output_type = "video";                        // Default output type
    config.output.name = "fluid_simulation_output.mp4";
    config.output.preset = "veryfast"; // Default libx264 encode preset         // Default output name
    config.mReconParams.reconParams = {true, true, 0.9f, 4.0f, 0.5f, 25};
    config.mReconParams.reconRadiusInSpacings = 15.56f; // Default Zhu-Bridson influence radius
    config.mReconParams.reconSupportScale = 1.0f;       // Default support scale
    config.mReconParams.mcIsovalue = 2.5f;              // Default
    return config;
}

SimulationConfig parseSimulationConfig(const std::string &filename)
{
    SimulationConfig config = defaultConfig(); // Start with default config

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Parse the JSON content
    nlohmann::json jsonConfig;
    file >> jsonConfig;

    // Extract simulation parameters
    config.outputJson = jsonConfig["outputJson"].get<std::string>();
    config.configName = jsonConfig["configName"].get<std::string>();
    config.duration = jsonConfig["duration"].get<float>();
    config.videoFPS = jsonConfig["videoFPS"].get<float>();
    config.maxFramesInFlight = jsonConfig["maxFramesInFlight"].get<int>();
    config.sphereStacks = jsonConfig["sphereStacks"].get<int>();
    config.sphereSectors = jsonConfig["sphereSectors"].get<int>();
    config.objInstancing = jsonConfig["objInstancing"].get<bool>();
    config.useOffScreenRendering = jsonConfig["useOffScreenRendering"].get<bool>();
    config.deltaTime = jsonConfig["deltaTime"].get<float>();
    config.fluidProps.density = jsonConfig["fluidProperties"]["density"].get<float>();
    config.fluidProps.viscosity = jsonConfig["fluidProperties"]["viscosity"].get<float>();
    config.fluidProps.flipRatio = jsonConfig["fluidProperties"]["flipRatio"].get<float>();
    config.fluidProps.pressureSolverType = jsonConfig["fluidProperties"]["pressureSolverType"].get<std::string>();
    config.fluidProps.useNarrowBand = jsonConfig["fluidProperties"]["useNarrowBand"].get<bool>();
    config.numParticles = jsonConfig["numParticles"].get<int>();
    config.env.externalForce.x = jsonConfig["environment"]["externalForce"]["x"].get<float>();
    config.env.externalForce.y = jsonConfig["environment"]["externalForce"]["y"].get<float>();
    config.env.externalForce.z = jsonConfig["environment"]["externalForce"]["z"].get<float>();
    config.env.externalForce.time_start = jsonConfig["environment"]["externalForce"]["time_start"].get<float>();
    config.env.externalForce.time_end = jsonConfig["environment"]["externalForce"]["time_end"].get<float>();
    config.env.numberOfSphere = jsonConfig["environment"]["numberOfSphere"].get<int>();
    config.env.initialVelocities.clear();
    config.env.initialVelocities.reserve(config.env.numberOfSphere);
    for (const auto &vel : jsonConfig["environment"]["initialVelocities"])
    {
        config.env.initialVelocities.push_back({vel["x"].get<float>(), vel["y"].get<float>(), vel["z"].get<float>()});
    }
    config.deviceId = jsonConfig["deviceId"].get<int>();
    config.num_iterations = jsonConfig["num_iterations"].get<int>();
    config.reconResolution = jsonConfig["reconResolution"].get<int>();
    config.grid_size.x = jsonConfig["gridSize"]["x"].get<int>();
    config.grid_size.y = jsonConfig["gridSize"]["y"].get<int>();
    config.grid_size.z = jsonConfig["gridSize"]["z"].get<int>();
    config.grid_size.cellSize = jsonConfig["gridSize"]["cellSize"].get<float>() + 0.000001f;
    config.particleSpacing = jsonConfig["particleSpacing"].get<float>();
    config.spacing = jsonConfig["spacing"].get<float>();
    config.radius_percentage = jsonConfig["radius_percentage"].get<float>();
    config.jitter = jsonConfig["jitter"].get<float>();

    config.output.output_type = jsonConfig["output"]["output_type"].get<std::string>();
    config.output.name = jsonConfig["output"]["name"].get<std::string>();
    if (jsonConfig["output"].contains("preset"))
        config.output.preset = jsonConfig["output"]["preset"].get<std::string>();

    config.mReconParams.reconRadiusInSpacings = jsonConfig["meshReconstruction"]["reconRadiusInSpacings"].get<float>();
    config.mReconParams.reconSupportScale = jsonConfig["meshReconstruction"]["reconSupportScale"].get<float>();
    config.mReconParams.mcIsovalue = jsonConfig["meshReconstruction"]["mcIsovalue"].get<float>();
    config.mReconParams.reconParams.anisotropic = jsonConfig["meshReconstruction"]["reconParams"]["anisotropic"].get<bool>();
    config.mReconParams.reconParams.smoothing = jsonConfig["meshReconstruction"]["reconParams"]["smoothing"].get<bool>();
    config.mReconParams.reconParams.lambda = jsonConfig["meshReconstruction"]["reconParams"]["lambda"].get<float>();
    config.mReconParams.reconParams.kr = jsonConfig["meshReconstruction"]["reconParams"]["kr"].get<float>();
    config.mReconParams.reconParams.kn = jsonConfig["meshReconstruction"]["reconParams"]["kn"].get<float>();
    config.mReconParams.reconParams.nEps = jsonConfig["meshReconstruction"]["reconParams"]["nEps"].get<int>();
    return config;
}

nlohmann::json parseSimulationConfigToJson(SimulationConfig &config)
{
    nlohmann::json jsonConfig;
    jsonConfig["outputJson"] = config.outputJson;
    jsonConfig["configName"] = config.configName;
    jsonConfig["duration"] = config.duration;
    jsonConfig["videoFPS"] = config.videoFPS;
    jsonConfig["maxFramesInFlight"] = config.maxFramesInFlight;
    jsonConfig["sphereStacks"] = config.sphereStacks;
    jsonConfig["sphereSectors"] = config.sphereSectors;
    jsonConfig["objInstancing"] = config.objInstancing;
    jsonConfig["useOffScreenRendering"] = config.useOffScreenRendering;
    jsonConfig["deltaTime"] = config.deltaTime;
    
    jsonConfig["fluidProperties"]["density"] = config.fluidProps.density;
    jsonConfig["fluidProperties"]["viscosity"] = config.fluidProps.viscosity;
    jsonConfig["fluidProperties"]["flipRatio"] = config.fluidProps.flipRatio;
    jsonConfig["fluidProperties"]["pressureSolverType"] = config.fluidProps.pressureSolverType;
    jsonConfig["fluidProperties"]["useNarrowBand"] = config.fluidProps.useNarrowBand;
    
    jsonConfig["numParticles"] = config.numParticles;
    
    jsonConfig["environment"]["externalForce"]["x"] = config.env.externalForce.x;
    jsonConfig["environment"]["externalForce"]["y"] = config.env.externalForce.y;
    jsonConfig["environment"]["externalForce"]["z"] = config.env.externalForce.z;
    jsonConfig["environment"]["externalForce"]["time_start"] = config.env.externalForce.time_start;
    jsonConfig["environment"]["externalForce"]["time_end"] = config.env.externalForce.time_end;
    jsonConfig["environment"]["numberOfSphere"] = config.env.numberOfSphere;
    
    jsonConfig["environment"]["initialVelocities"] = nlohmann::json::array();
    for (const auto &vel : config.env.initialVelocities)
    {
        nlohmann::json velObj;
        velObj["x"] = vel.x;
        velObj["y"] = vel.y;
        velObj["z"] = vel.z;
        jsonConfig["environment"]["initialVelocities"].push_back(velObj);
    }
    
    jsonConfig["deviceId"] = config.deviceId;
    jsonConfig["num_iterations"] = config.num_iterations;
    jsonConfig["reconResolution"] = config.reconResolution;
 
    jsonConfig["gridSize"]["x"] = config.grid_size.x;
    jsonConfig["gridSize"]["y"] = config.grid_size.y;
    jsonConfig["gridSize"]["z"] = config.grid_size.z;
    jsonConfig["gridSize"]["cellSize"] = config.grid_size.cellSize;
    
    jsonConfig["particleSpacing"] = config.particleSpacing;
    jsonConfig["spacing"] = config.spacing;
    jsonConfig["radius_percentage"] = config.radius_percentage;
    jsonConfig["jitter"] = config.jitter;

    jsonConfig["output"]["output_type"] = config.output.output_type;
    jsonConfig["output"]["name"] = config.output.name;
    jsonConfig["output"]["preset"] = config.output.preset;

    jsonConfig["meshReconstruction"]["reconRadiusInSpacings"] = config.mReconParams.reconRadiusInSpacings;
    jsonConfig["meshReconstruction"]["reconSupportScale"] = config.mReconParams.reconSupportScale;
    jsonConfig["meshReconstruction"]["mcIsovalue"] = config.mReconParams.mcIsovalue;
    
    jsonConfig["meshReconstruction"]["reconParams"]["anisotropic"] = config.mReconParams.reconParams.anisotropic;
    jsonConfig["meshReconstruction"]["reconParams"]["smoothing"] = config.mReconParams.reconParams.smoothing;
    jsonConfig["meshReconstruction"]["reconParams"]["lambda"] = config.mReconParams.reconParams.lambda;
    jsonConfig["meshReconstruction"]["reconParams"]["kr"] = config.mReconParams.reconParams.kr;
    jsonConfig["meshReconstruction"]["reconParams"]["kn"] = config.mReconParams.reconParams.kn;
    jsonConfig["meshReconstruction"]["reconParams"]["nEps"] = config.mReconParams.reconParams.nEps;

    return jsonConfig;
}