#ifndef CUDA_PARTICLE_SIMULATOR_H
#define CUDA_PARTICLE_SIMULATOR_H
#include "../cudaCommons.h"
#include "../../vulkan/utils.h"
#include "../ChronoCuda/ChronoCuda.h"

class cudaParticleSimulator
{
private:
    cudaStream_t stream;

    cudaStream_t uStream, vStream, wStream;
    appState *state;
    fluidProperties fluidProps;
    particleDeviceData deviceData;
    cellType *d_cellTypes;
    gridSize grid_size;
    gridDeviceData grid_data;
    float spacing, radius_percentage, jitter, sphereRadius;
    int numParticles;
    environment env;
    int deviceId;
    float deltaTime, currentTime = 0.0f;
    int num_iterations;
    bool narrowBandMarked = false;
    // multigrid
    float *coarse_p;
    float *coarse_buf;
    float *coarse_div;
    int *coarse_cellType;

public:
    cudaParticleSimulator() = default;
    cudaParticleSimulator(SimulationConfig config, cudaStream_t &stream_state);
    ~cudaParticleSimulator();
    void clean();
    void initParticles();
    void init_new_particles();
    void loadStaticParticles(const std::vector<glm::vec3> &positions);
    void computeBoundary();
    void updateSystem();
    void markNarrowBand();
    void p2g();
    void saveGridVelocities();
    void applyExternalForces();
    // pressure solver
    void pressureSolve();
    void cellClassification();
    void boundaryConditions();
    void computeDivergence();
    void jacobi();
    void applyPressureGradient();
    // extra
    void extrapolateVelocity();
    // transfer
    void g2p();
    void computeAdvection();
    void cuda2vulkan(uint32_t currentImage, appState &state);
    float getDeltaTime() const { return deltaTime; }
    float3 *getParticlePositions() const { return deviceData.pos; }
    int getNumParticles() const { return numParticles; }
    const int *getCellTypes() const { return grid_data.cellType; }
    gridSize getSimGrid() const { return grid_size; }
    void setSphereRadius(float radius) { sphereRadius = radius; }
    void setAppState(appState *appState) { state = appState; }

    // MUTLIGRID
    void restrictGrid(const float *fine, float *coarse, int nxF, int nyF, int nzF);
    void prolongateGrid(const float *coarse, float *fine, int nxF, int nyF, int nzF);
    void smoothJacobi(float *p, float *pBuf, float *div, int *cellType, int nx, int ny, int nz, int iters);
    void multigridVCycle(float *p, float *div, int *cellType, int nx, int ny, int nz);

    // jacobi kernel declaration for multigrid
};

#endif