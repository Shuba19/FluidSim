#include "cudaParticleSimulator.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <random>

#define OLD false

#define CHECK(call)                                                         \
    {                                                                       \
        const cudaError_t error = call;                                     \
        if (error != cudaSuccess)                                           \
        {                                                                   \
            printf("Error %s : %d\n", __FILE__, __LINE__);                  \
            printf("code:%d, reason:%s", error, cudaGetErrorString(error)); \
            exit(1);                                                        \
        }                                                                   \
    }

// init all
/*
@param config: SimulationConfig object containing simulation parameters
@param stream_state: CUDA stream to be used for asynchronous operations

*/
cudaParticleSimulator::cudaParticleSimulator(SimulationConfig config, cudaStream_t &stream_state)
{
    srand(time(NULL));
    this->deviceId = config.deviceId;
    cudaSetDevice(this->deviceId);
    // cudaStreamCreate(&this->stream);
    this->stream = stream_state;

    CHECK(cudaStreamCreate(&uStream));
    CHECK(cudaStreamCreate(&vStream));
    CHECK(cudaStreamCreate(&wStream));
    this->numParticles = config.numParticles;
    this->deltaTime = config.deltaTime;
    this->fluidProps = config.fluidProps;
    this->num_iterations = config.num_iterations;
    this->env = config.env;
    this->grid_size = config.grid_size;
    this->spacing = config.particleSpacing;
    this->radius_percentage = config.radius_percentage;
    this->jitter = config.jitter;
    this->narrowBandMarked = config.fluidProps.useNarrowBand;
    printf("Initializing CUDA Particle Simulator with %d particles, grid size (%d, %d, %d), and %d pressure iterations\n",
           numParticles, grid_size.x, grid_size.y, grid_size.z, num_iterations);
    cudaMallocAsync(&deviceData.pos, numParticles * sizeof(float3), stream);
    cudaMallocAsync(&deviceData.vel, numParticles * sizeof(float3), stream);
    cudaMallocAsync(&deviceData.old_vel, numParticles * sizeof(float3), stream);
    // debug environemnt
    std::cout << "Environment settings:" << std::endl;
    std::cout << "External Force: (" << env.externalForce.x << ", " << env.externalForce.y << ", " << env.externalForce.z << ")" << std::endl;
    std::cout << "External Force Time Start: " << env.externalForce.time_start << std::endl;
    std::cout << "External Force Time End: " << env.externalForce.time_end << std::endl;
    std::cout << "Number of Spheres: " << env.numberOfSphere << std::endl;
    std::cout << "Initial Velocities: " << std::endl;
    for (const auto &vel : env.initialVelocities)
    {
        std::cout << "  (" << vel.x << ", " << vel.y << ", " << vel.z << ")" << std::endl;
    }

    // MAc num components calculations
    int uCells = (grid_size.x + 1) * grid_size.y * grid_size.z;
    int vCells = grid_size.x * (grid_size.y + 1) * grid_size.z;
    int wCells = grid_size.x * grid_size.y * (grid_size.z + 1);
    int centerCells = grid_size.gridCells; // x * y * z

    // ALLOCAZIONE CELL TYPE
    cudaMallocAsync(&this->grid_data.cellType, centerCells * sizeof(int), stream);
    // Allocazione facce della griglia (Velocità)
    cudaMallocAsync(&this->grid_data.u, uCells * sizeof(float), stream);
    cudaMallocAsync(&this->grid_data.v, vCells * sizeof(float), stream);
    cudaMallocAsync(&this->grid_data.w, wCells * sizeof(float), stream);
    // allocazione old uvw
    cudaMallocAsync(&this->grid_data.u_old, uCells * sizeof(float), stream);
    cudaMallocAsync(&this->grid_data.v_old, vCells * sizeof(float), stream);
    cudaMallocAsync(&this->grid_data.w_old, wCells * sizeof(float), stream);

    // PESI
    cudaMallocAsync(&grid_data.uWeight, uCells * sizeof(float), stream);
    cudaMallocAsync(&grid_data.vWeight, vCells * sizeof(float), stream);
    cudaMallocAsync(&grid_data.wWeight, wCells * sizeof(float), stream);

    cudaMemsetAsync(grid_data.uWeight, 0, uCells * sizeof(float), stream);
    cudaMemsetAsync(grid_data.vWeight, 0, vCells * sizeof(float), stream);
    cudaMemsetAsync(grid_data.wWeight, 0, wCells * sizeof(float), stream);

    // Allocazione centri delle celle (Pressione e Tipo di cella)
    cudaMallocAsync(&this->grid_data.p, centerCells * sizeof(float), stream);

    // allocazione per la divergenza e il buffer della pressione
    cudaMallocAsync(&grid_data.divergence, grid_size.x * grid_size.y * grid_size.z * sizeof(float), stream);
    cudaMallocAsync(&grid_data.pBuffer, grid_size.x * grid_size.y * grid_size.z * sizeof(float), stream);

    cudaMemsetAsync(grid_data.divergence, 0, grid_size.x * grid_size.y * grid_size.z * sizeof(float), stream);
    cudaMemsetAsync(grid_data.pBuffer, 0, grid_size.x * grid_size.y * grid_size.z * sizeof(float), stream);

    // conviene sempre fare un po di memsetr per i valori
    cudaMemsetAsync(this->grid_data.u, 0, uCells * sizeof(float), stream);
    cudaMemsetAsync(this->grid_data.v, 0, vCells * sizeof(float), stream);
    cudaMemsetAsync(this->grid_data.w, 0, wCells * sizeof(float), stream);
    cudaMemsetAsync(this->grid_data.p, 0, centerCells * sizeof(float), stream);
    // tutta le cella iniziallizata come aria. POoi si mette  la prima
    cudaMemsetAsync(this->grid_data.cellType, int(cellType::AIR), centerCells * sizeof(int), stream);

    // multigrid allocations
    int coarseCells = (grid_size.x / 2) * (grid_size.y / 2) * (grid_size.z / 2);
    cudaMallocAsync(&coarse_p, coarseCells * sizeof(float), stream);
    cudaMallocAsync(&coarse_buf, coarseCells * sizeof(float), stream);
    cudaMallocAsync(&coarse_div, coarseCells * sizeof(float), stream);
    cudaMallocAsync(&coarse_cellType, coarseCells * sizeof(int), stream);
}

cudaParticleSimulator::~cudaParticleSimulator()
{
    //NOT USED 
}

void cudaParticleSimulator::clean()
{
    CHECK(cudaFreeAsync(deviceData.pos, stream));
    CHECK(cudaFreeAsync(deviceData.vel, stream));
    CHECK(cudaFreeAsync(deviceData.old_vel, stream));
    CHECK(cudaFreeAsync(grid_data.u, stream));
    CHECK(cudaFreeAsync(grid_data.v, stream));
    CHECK(cudaFreeAsync(grid_data.w, stream));
    CHECK(cudaFreeAsync(grid_data.u_old, stream));
    CHECK(cudaFreeAsync(grid_data.v_old, stream));
    CHECK(cudaFreeAsync(grid_data.w_old, stream));
    CHECK(cudaFreeAsync(grid_data.uWeight, stream));
    CHECK(cudaFreeAsync(grid_data.vWeight, stream));
    CHECK(cudaFreeAsync(grid_data.wWeight, stream));
    CHECK(cudaFreeAsync(grid_data.divergence, stream));
    CHECK(cudaFreeAsync(grid_data.pBuffer, stream));
    CHECK(cudaFreeAsync(grid_data.p, stream));
    CHECK(cudaFreeAsync(grid_data.cellType, stream));
    CHECK(cudaFreeAsync(coarse_p, stream));
    CHECK(cudaFreeAsync(coarse_buf, stream));
    CHECK(cudaFreeAsync(coarse_div, stream));
    CHECK(cudaFreeAsync(coarse_cellType, stream));
    CHECK(cudaStreamDestroy(uStream));
    CHECK(cudaStreamDestroy(vStream));
    CHECK(cudaStreamDestroy(wStream));
    CHECK(cudaStreamSynchronize(stream));
    
}
void cudaParticleSimulator::init_new_particles()
{
    float dx = grid_size.cellSize;
    float p_spacing =  spacing;

    float worldWidth = grid_size.x * dx;
    float worldHeight = grid_size.y * dx;
    float worldDepth = grid_size.z * dx;

    int num_spheres = env.numberOfSphere;

    if (num_spheres <= 0)
    {
        std::cerr << "Error: Number of spheres must be greater than 0." << std::endl;
        return;
    }

    // Ogni sfera deve contenere esattamente questa quota di particelle
    int particlesPerSphere = this->numParticles / num_spheres;

    // Calcoliamo teoricamente il raggio necessario per contenere quel numero di particelle
    // Volume sfera = (4/3) * pi * R^3. Ogni particella occupa circa (p_spacing)^3 di volume.
    // R = root_3( (3 * N * p_spacing^3) / (4 * pi) )
    float volumePerParticle = p_spacing * p_spacing * p_spacing;
    float targetVolume = particlesPerSphere * volumePerParticle;
    float radius = std::cbrt((3.0f * targetVolume) / (4.0f * M_PI));

    std::cout << "Initializing particles with Dynamic Radius:" << std::endl;
    std::cout << "Total targeted particles: " << this->numParticles << std::endl;
    std::cout << "Number of spheres: " << num_spheres << std::endl;
    std::cout << "Particles per sphere: " << particlesPerSphere << std::endl;
    std::cout << "Calculated Sphere radius: " << radius << std::endl;
    std::cout << "Particle spacing: " << p_spacing << std::endl;

    float3 containerCenter = {worldWidth / 2.0f, worldHeight * 0.75f, worldDepth / 2.0f};
    float gap = radius * this->radius_percentage;
    float totalWidth = num_spheres * 2.0f * radius + (num_spheres - 1) * gap;

    if (totalWidth > worldWidth)
    {
        std::cerr << "WARNING: Spheres are too large for the domain width!" << std::endl;
    }

    float startX = containerCenter.x - totalWidth / 2.0f + radius;

    std::vector<float3> h_pos;
    std::vector<float3> h_vel;

    // Riserviamo lo spazio esatto per evitare riallocazioni continue del vettore
    h_pos.reserve(this->numParticles);
    h_vel.reserve(this->numParticles);

    for (int s = 0; s < num_spheres; ++s)
    {
        float3 center = {
            startX + s * (2.0f * radius + gap),
            containerCenter.y,
            containerCenter.z};

        // Per assicurarci di prendere esattamente le particelle più vicine al centro ed evitare i vuoti
        // del bounding box cubico, generiamo i punti in una griglia locale leggermente più ampia
        // e li ordiniamo per distanza dal centro della sfera.
        struct PotentialParticle
        {
            float3 pos;
            float distSq;
        };
        std::vector<PotentialParticle> candidates;

        // Estendiamo il bounding box di sicurezza per catturare abbastanza punti nel campionamento discreto
        float searchRadius = radius + p_spacing * 2.0f;
        float3 minB = {center.x - searchRadius, center.y - searchRadius, center.z - searchRadius};
        float3 maxB = {center.x + searchRadius, center.y + searchRadius, center.z + searchRadius};

        for (float x = minB.x; x <= maxB.x; x += p_spacing)
        {
            for (float y = minB.y; y <= maxB.y; y += p_spacing)
            {
                for (float z = minB.z; z <= maxB.z; z += p_spacing)
                {
                    float distSq = (x - center.x) * (x - center.x) +
                                   (y - center.y) * (y - center.y) +
                                   (z - center.z) * (z - center.z);

                    float jx = ((rand() / (float)RAND_MAX) - 0.5f) * p_spacing * jitter;
                    float jy = ((rand() / (float)RAND_MAX) - 0.5f) * p_spacing * jitter;
                    float jz = ((rand() / (float)RAND_MAX) - 0.5f) * p_spacing * jitter;

                    if (distSq <= searchRadius * searchRadius) 
                        candidates.push_back({{x + jx, y + jy, z + jz}, distSq});
                }
            }
        }

        // Ordiniamo i candidati: i più vicini al centro della sfera vanno per primi
        std::sort(candidates.begin(), candidates.end(), [](const PotentialParticle &a, const PotentialParticle &b)
                  { return a.distSq < b.distSq; });

        // Prendiamo esattamente il numero richiesto di particelle per questa sfera
        int added = 0;
        InitialVelocity initVel = env.initialVelocities[s % env.initialVelocities.size()];

        for (const auto &candidate : candidates)
        {
            if (added >= particlesPerSphere)
                break;

            h_pos.push_back(candidate.pos);
            h_vel.push_back({initVel.x, initVel.y, initVel.z});
            added++;
        }
    }

    // Aggiorna il numero finale effettivo di particelle generate complessivamente
    this->numParticles = h_pos.size();
    std::cout << "Successfully generated " << this->numParticles << " active particles." << std::endl;

    // Trasferimento asincrono sulla GPU
    cudaMemcpyAsync(deviceData.pos, h_pos.data(), this->numParticles * sizeof(float3), cudaMemcpyHostToDevice, this->stream);
    cudaMemcpyAsync(deviceData.vel, h_vel.data(), this->numParticles * sizeof(float3), cudaMemcpyHostToDevice, this->stream);
    cudaMemcpyAsync(deviceData.old_vel, h_vel.data(), this->numParticles * sizeof(float3), cudaMemcpyHostToDevice, this->stream);
}

void cudaParticleSimulator::initParticles()
{
    //LEGACY
    init_new_particles();
}
__global__ void initBoundariesKernel(int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    int idx = i + nx * (j + ny * k);

    if (i == 0 || i == nx - 1 ||
        j == 0 || j == ny - 1 ||
        k == 0 || k == nz - 1)
    {
        cellType[idx] = (int)cellType::SOLID;
    }
}
void cudaParticleSimulator::loadStaticParticles(const std::vector<glm::vec3> &positions)
{
    int n = std::min((int)positions.size(), numParticles);
    std::vector<float3> tmp(n);
    for (int i = 0; i < n; ++i)
        tmp[i] = {positions[i].x, positions[i].y, positions[i].z};
    CHECK(cudaMemcpyAsync(deviceData.pos, tmp.data(), n * sizeof(float3), cudaMemcpyHostToDevice, this->stream));
}

void cudaParticleSimulator::computeBoundary()
{
    dim3 block(8, 8, 8);
    dim3 grid(
        (grid_size.x + block.x - 1) / block.x,
        (grid_size.y + block.y - 1) / block.y,
        (grid_size.z + block.z - 1) / block.z);
    initBoundariesKernel<<<grid, block, 0, this->stream>>>(
        grid_data.cellType,
        grid_size.x, grid_size.y, grid_size.z);
    cudaStreamSynchronize(this->stream);
}