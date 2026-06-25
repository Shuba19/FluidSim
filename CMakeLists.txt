cmake_minimum_required(VERSION 3.15)

project(FluidSim LANGUAGES CXX CUDA)
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-O3>")

# Ottimizzazione -O3 per il compilatore CUDA (nvcc)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
endif()
add_executable(FluidSimulation
    src/main.cpp
    src/vulkan/initVulkan.cpp
    src/vulkan/buffer.cpp
    src/vulkan/pipeline.cpp
    src/vulkan/device.cpp
    src/vulkan/swapChain.cpp
    src/vulkan/utils.cpp
    src/vulkan/offscreen.cpp
    src/simulation/ChronoCuda/ChronoCuda.cu
    src/simulation/cudaParticleSimulator/cudaParticleSimulator.cu
    src/simulation/cudaParticleSimulator/Handlers/extForces.cu
    src/simulation/cudaParticleSimulator/Handlers/pressure_solver.cu
    src/simulation/cudaParticleSimulator/Handlers/transfer.cu
    src/simulation/cudaParticleSimulator/Handlers/updateSystem.cu
    src/simulation/cudaParticleSimulator/Handlers/extrapolateVelocity.cu
    src/simulation/cudaParticleSimulator/Handlers/markNarrowBand.cu
    src/simulation/cudaParticleSimulator/Handlers/multigrid.cu
    src/MeshReconstruction/scalarField.cu
    src/MeshReconstruction/marchingCubes.cu
    src/MeshReconstruction/neighborIndex.cu
    src/parser/parser_json.cu
)

set_target_properties(FluidSimulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Dependencies
find_package(Vulkan REQUIRED)
target_link_libraries(FluidSimulation PRIVATE Vulkan::Vulkan)

# std::thread (FrameWriter background encoder) needs pthread on Linux.
find_package(Threads REQUIRED)
target_link_libraries(FluidSimulation PRIVATE Threads::Threads)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(FluidSimulation PRIVATE CUDA::cudart)

include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8 # Pinning a tag prevents unexpected remote updates/loops
)
FetchContent_MakeAvailable(glfw)
target_link_libraries(FluidSimulation PRIVATE glfw)

# GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0.9.9.8
)
FetchContent_MakeAvailable(glm)
target_link_libraries(FluidSimulation PRIVATE glm)

# JSON
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(FluidSimulation PRIVATE nlohmann_json::nlohmann_json)