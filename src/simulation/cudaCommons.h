#ifndef CUDA_COMMONS_H
#define CUDA_COMMONS_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include <driver_types.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <GLFW/glfw3.h>
#include <vector>
/*
@params density: density of the fluid
@params viscosity: viscosity of the fluid
@params flipRatio: ratio of FLIP to PIC, between 0 and 1
*/
struct fluidProperties
{
    float density;
    float viscosity;
    float flipRatio;
    std::string pressureSolverType; // "jacobi", "gs", or "multigrid"
    bool useNarrowBand;
};

struct particleDeviceData
{
    float3 *pos;
    float3 *vel, *old_vel; //la old vel serve per il calcolo dell'accelerazione, che è data da (vel - old_vel) / deltaTime
};
struct gridDeviceData {
   
    float *u, *u_old; // Velocità sulle facce x
    float *v, *v_old; // Velocità sulle facce y
    float *w, *w_old; // Velocità sulle facce z

    float *uWeight, *vWeight, *wWeight;  //PESI
    float *divergence; // Divergenza al centro della cella
    float *pBuffer;

    float *p; // Pressione al centro della cella
    int *cellType;
};
struct gridSize {
    int x, y, z;
    int gridCells;
    float cellSize;
    float3 origin;   // world-space position del corner (0,0,0) della griglia
    gridSize() : x(0), y(0), z(0), cellSize(0.1f), gridCells(0), origin{0,0,0} {}
    gridSize(int x, int y, int z, float cellSize = 0.1f, float3 org = {0,0,0})
        : x(x), y(y), z(z), cellSize(cellSize), gridCells(x * y * z), origin(org) {}

    // World extent of the domain along each axis (origin assumed at 0).
    float worldSizeX() const { return x * cellSize; }
    float worldSizeY() const { return y * cellSize; }
    float worldSizeZ() const { return z * cellSize; }

    static gridSize matchingDomain(const gridSize& base, int resX, int resY, int resZ) {
        float cs = base.worldSizeX() / resX;
        float3 org = {base.origin.x - cs, base.origin.y - cs, base.origin.z - cs};
        return gridSize(resX + 2, resY + 2, resZ + 2, cs, org);
    }
    static gridSize matchingDomain(const gridSize& base, int res) {
        return matchingDomain(base, res, res, res);
    }
};

struct vulkanData{
    VkBuffer posBuffer;
};

// Tipi di cella della MAC grid (sim + narrow band). Definiti qui (non in
// cudaParticleSimulator.h) così la mesh recon può usarli senza includere la sim.
enum cellType {
    AIR         = 0,
    FLUID       = 1,
    SOLID       = 2,
    NARROW_BAND = 3
};

struct ReconParams {
    bool  anisotropic = false;   // false → force isotropic kernels (A/B baseline)
    bool  smoothing   = false;   // Laplacian position smoothing on/off
    float lambda      = 0.9f;   // smoothing blend: x' = (1-l)x + l*weightedMean
    float kr          = 4.0f;   // max anisotropy ratio (clamp on singular values)
    float kn          = 0.5f;   // isotropic kernel scale for sparse neighbourhoods
    int   nEps        = 25;     // neighbour-count threshold for the anisotropic branch
};
struct InitialVelocity
{
    float x, y, z;
};

struct environment
{
    struct {
        float x, y, z, time_start, time_end;
    } externalForce;
    int numberOfSphere;
    std::vector<InitialVelocity> initialVelocities;
};
#endif