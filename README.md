# PIC/FLIP Fluid Simulation with Marching Cube mesh reconstruction
This project aims to reproduce a physics based fluid simulator leveraging CUDA and Nvidia GPUs.
We propose a PIC/FLIP solver for the simulation of the system combined with a Marching Cubes for the mesh Reconstruction.

## Folder Hierarchy
### src
src is the main folder, containing all the directories for the project and the main.cpp file.
#### MeshReconstruction
Contains all the file for the mesh reconstruction  
#### parser
Json parser library for setting the config
#### shaders
Shader vert and spv files for rendering
#### simulation
Contains a library for the implementation of the PIC/FLIP solver and 
#### stats
Library for keeping tracks of time for each step
#### vulkan
All the files regarding the configuration and implementation of Vulkan in our system.
### testResult
We runned a suite of 54 test, compraing our framework with Taichi and SplashSurf.
It is possible to see the full report of the test inside this folder.
Config.md contains a table with the characteristcs of each  config.
report_performance.md the full report of the test.
summary_table.md shows a summary of the result including geometric mean and avg speedup.

## Examples
# Particles
<img src="/testResults/images/renderParticle.png" width="200">

### Fluid under external force

<img src="/testResults/images/ExtForcebefore.png" width="200">
<img src="/testResults/images/ExtForceDuring.png" width="200">
<img src="/testResults/images/ExtForceAfter.png" width="200">

### GIF
<img src="/testResults/images/fluid_render.gif" width="200">

## How to compile

### Shader
First of all you have to compile shaders. In the main directory:
```
cd src/shaders
./compile.sh
```
### Compiling the project
Then you can proceed to compile the project
```
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="89"
make -j$(nproc)
```
### Requirement To Compile
Cmake version >= 3.15
Cuda Toolkit >= 13.1
G++ >= 13.3.0

Other dipendences, like Vulkan or nhlomannJson will be installed by Cmake.

## How to Run
It is possible to launch the program by simply
```
./FluidSimulation
```
If the program will return an error as ```cudaImportExternalMemory failed!```, try launching the program with:
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json ./FluidSimulation
```
It is also possible to specify other parameters, each one described by the config_schema.json, by running the program as:
```
./FluidSimulation -c ../config.json
```
An example config.json has been made available.