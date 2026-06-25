#include "../cudaParticleSimulator.h"

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


// Forward declarations
__global__ void cell_classification_kernel(int *cellType, int nx, int ny, int nz);
__global__ void markFluidCellsKernel(int *cellType, const float3 *pos, int numParticles, int nx, int ny, int nz, float cellSize);
__global__ void applyUBoundaryKernel(float *u, const int *cellType, int nx, int ny, int nz);
__global__ void applyVBoundaryKernel(float *v, const int *cellType, int nx, int ny, int nz);
__global__ void applyWBoundaryKernel(float *w, const int *cellType, int nx, int ny, int nz);
__global__ void compute_divergence_kernel(const float *u, const float *v, const float *w, float *divergence, const int *cellType, int nx, int ny, int nz, float invDx);
__global__ void applyUGradientKernel(float *u, const float *p, const int *cellType, int nx, int ny, int nz, float scale);
__global__ void applyVGradientKernel(float *v, const float *p, const int *cellType, int nx, int ny, int nz, float scale);
__global__ void applyWGradientKernel(float *w, const float *p, const int *cellType, int nx, int ny, int nz, float scale);

__global__ void jacobiKernel(const float *pSrc, float *pDst, const float *divergence, const int *cellType, int nx, int ny, int nz, float scale);
__global__ void cell_classification_kernel(int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz)
        return;

    int cidx = i + nx * (j + ny * k);

    // Solo pavimento e pareti laterali sono SOLID
    // Il soffitto (j == ny-1) resta AIR — superficie libera
    if (i == 0 || i == nx - 1 ||
        j == 0 || // ← pavimento SOLID
        k == 0 || k == nz - 1)
    {
        cellType[cidx] = (int)cellType::SOLID;
    }
    else
    {
        cellType[cidx] = (int)cellType::AIR;
    }
}

__global__ void markFluidCellsKernel(int *cellType, const float3 *pos, int numParticles, int nx, int ny, int nz, float cellSize)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= numParticles)
        return;

    float3 p = pos[pid];
    int i = __float2int_rd(p.x / cellSize);
    int j = __float2int_rd(p.y / cellSize);
    int k = __float2int_rd(p.z / cellSize);

    i = max(0, min(i, nx - 1));
    j = max(0, min(j, ny - 1));
    k = max(0, min(k, nz - 1));

    int cidx = i + nx * (j + ny * k);
    if (cellType[cidx] == (int)cellType::AIR)
        atomicExch(&cellType[cidx], (int)cellType::FLUID);
}

void cudaParticleSimulator::cellClassification()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    int size = nx * ny * nz;

    // Reset pressione
    cudaMemset(grid_data.p, 0, size * sizeof(float));

    // Classifica celle
    dim3 block(8, 8, 4);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    cell_classification_kernel<<<grid, block>>>(grid_data.cellType, nx, ny, nz);
    CHECK(cudaDeviceSynchronize());

    // Marca celle fluide
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    markFluidCellsKernel<<<gridSize, blockSize>>>(grid_data.cellType, deviceData.pos, numParticles, nx, ny, nz, grid_size.cellSize);
    CHECK(cudaDeviceSynchronize());
}

__global__ void applyUBoundaryKernel(float *u, const int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j >= ny || k >= nz)
        return;

    int uidx = i + (nx + 1) * (j + ny * k);
    if (i == 0 || i == nx)
    {
        u[uidx] = 0.0f;
        return;
    }

    int idxL = (i - 1) + nx * (j + ny * k);
    int idxR = i + nx * (j + ny * k);
}

__global__ void applyVBoundaryKernel(float *v, const int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j > ny || k >= nz)
        return;

    int vidx = i + nx * (j + (ny + 1) * k);

    if (j == 0)
    {
        v[vidx] = 0.0f;
        return;
    }

    int idxB = i + nx * ((j - 1) + ny * k);
    int idxT = i + nx * (j + ny * k);
}

__global__ void applyWBoundaryKernel(float *w, const int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k > nz)
        return;

    int widx = i + nx * (j + ny * k);
    if (k == 0 || k == nz)
    {
        w[widx] = 0.0f;
        return;
    }

    int idxBk = i + nx * (j + ny * (k - 1));
    int idxFr = i + nx * (j + ny * k);
}

void cudaParticleSimulator::boundaryConditions()
{
    CHECK(cudaStreamSynchronize(this->stream));
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    dim3 block(8, 8, 4);

    dim3 gridU((nx + 1 + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    applyUBoundaryKernel<<<gridU, block, 0, uStream>>>(grid_data.u, grid_data.cellType, nx, ny, nz);

    dim3 gridV((nx + 7) / 8, (ny + 1 + 7) / 8, (nz + 3) / 4);
    applyVBoundaryKernel<<<gridV, block, 0, vStream>>>(grid_data.v, grid_data.cellType, nx, ny, nz);

    dim3 gridW((nx + 7) / 8, (ny + 7) / 8, (nz + 1 + 3) / 4);
    applyWBoundaryKernel<<<gridW, block, 0, wStream>>>(grid_data.w, grid_data.cellType, nx, ny, nz);

    CHECK(cudaStreamSynchronize(uStream));
    CHECK(cudaStreamSynchronize(vStream));
    CHECK(cudaStreamSynchronize(wStream));
}

__global__ void compute_divergence_kernel(
    const float *u, const float *v, const float *w,
    float *divergence, const int *cellType,
    int nx, int ny, int nz, float invDx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz)
        return;

    int cidx = i + nx * (j + ny * k);
    if (cellType[cidx] != (int)cellType::FLUID)
    {
        divergence[cidx] = 0.0f;
        return;
    }

    float uL = u[i + (nx + 1) * (j + ny * k)];
    float uR = u[(i + 1) + (nx + 1) * (j + ny * k)];
    float vB = v[i + nx * (j + (ny + 1) * k)];
    float vT = v[i + nx * ((j + 1) + (ny + 1) * k)];
    float wBk = w[i + nx * (j + ny * k)];
    float wFr = w[i + nx * (j + ny * (k + 1))];

    divergence[cidx] = ((uR - uL) + (vT - vB) + (wFr - wBk)) * invDx;
}

void cudaParticleSimulator::computeDivergence()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    CHECK(cudaMemsetAsync(grid_data.divergence, 0, nx * ny * nz * sizeof(float), this->stream));

    dim3 block(8, 8, 4);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    compute_divergence_kernel<<<grid, block, 0, this->stream>>>(
        grid_data.u, grid_data.v, grid_data.w,
        grid_data.divergence, grid_data.cellType,
        nx, ny, nz, 1.0f / grid_size.cellSize);
}

// JACOBI
__global__ void jacobiKernel(
    const float *pSrc, float *pDst,
    const float *divergence, const int *cellType,
    int nx, int ny, int nz,
    float scale) // scale = ρ*dx²/dt
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz)
        return;

    int cidx = i + nx * (j + ny * k);
    if (cellType[cidx] != (int)cellType::FLUID)
    {
        pDst[cidx] = 0.0f;
        return;
    }

    float pC = pSrc[cidx];
    float sum = 0.0f;
    int count = 0;

#define READ_NEIGHBOR(nb, inBounds, isCeiling)                               \
    {                                                                        \
        if (inBounds)                                                        \
        {                                                                    \
            int _t = cellType[nb];                                           \
            if (_t == (int)cellType::SOLID)                                  \
            {                                                                \
                sum += pC;                                                   \
                count++;                                                     \
            }                                                                \
            else if (_t == (int)cellType::AIR)                               \
            {                                                                \
                sum += 0.0f;                                                 \
                count++;                                                     \
            }                                                                \
            else                                                             \
            {                                                                \
                sum += pSrc[nb];                                             \
                count++;                                                     \
            }                                                                \
        }                                                                    \
        else                                                                 \
        {                                                                    \
            if (isCeiling)                                                   \
            {                                                                \
                sum += 0.0f;                                                 \
                count++;                                                     \
            }                                                                \
            else                                                             \
            {                                                                \
                sum += pC;                                                   \
                count++;                                                     \
            }                                                                \
        }                                                                    \
    }

    READ_NEIGHBOR(cidx - 1, i > 0, false)
    READ_NEIGHBOR(cidx + 1, i < nx - 1, false)
    READ_NEIGHBOR(cidx - nx, j > 0, false)    
    READ_NEIGHBOR(cidx + nx, j < ny - 1, true) 
    READ_NEIGHBOR(cidx - nx * ny, k > 0, false)
    READ_NEIGHBOR(cidx + nx * ny, k < nz - 1, false)

#undef READ_NEIGHBOR

    if (count == 0)
    {
        pDst[cidx] = 0.0f;
        return;
    }

    
    float p_new = (sum - scale * divergence[cidx]) / count;

    if (p_new < 0.0f)
        p_new = 0.0f;
    pDst[cidx] = p_new;
}

void cudaParticleSimulator::jacobi()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    int size = nx * ny * nz;

    CHECK(cudaMemsetAsync(grid_data.p, 0, size * sizeof(float), this->stream));
    CHECK(cudaMemsetAsync(grid_data.pBuffer, 0, size * sizeof(float), this->stream));

    dim3 block(8, 8, 4);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);

    // scale = ρ * dx² / dt  da sand as fluid
    float scale = fluidProps.density * grid_size.cellSize * grid_size.cellSize / deltaTime;

    float *src = grid_data.p;
    float *dst = grid_data.pBuffer;

    for (int iter = 0; iter < num_iterations; iter++)
    {
        jacobiKernel<<<grid, block, 0, this->stream>>>(src, dst, grid_data.divergence, grid_data.cellType, nx, ny, nz, scale);
        CHECK(cudaStreamSynchronize(this->stream));

        float *tmp = src;
        src = dst;
        dst = tmp;
    }

    if (src != grid_data.p)
        CHECK(cudaMemcpyAsync(grid_data.p, src, size * sizeof(float), cudaMemcpyDeviceToDevice, this->stream));

    CHECK(cudaStreamSynchronize(this->stream));
}

void cudaParticleSimulator::pressureSolve()
{
    cellClassification();
    boundaryConditions();
    computeDivergence();
    if(fluidProps.pressureSolverType == "jacobi")
        jacobi();
    else if(fluidProps.pressureSolverType == "multigrid")
        multigridVCycle(grid_data.p, grid_data.divergence, grid_data.cellType, grid_size.x, grid_size.y, grid_size.z);
    else
        std::cerr << "Unknown pressure solver type: " << fluidProps.pressureSolverType << std::endl;
    
    applyPressureGradient();
}

__global__ void applyUGradientKernel(float *u, const float *p, const int *cellType, int nx, int ny, int nz, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j >= ny || k >= nz)
        return;

    if (i == 0 || i == nx)
    {
        u[i + (nx + 1) * (j + ny * k)] = 0.0f;
        return;
    }

    int idxL = (i - 1) + nx * (j + ny * k);
    int idxR = i + nx * (j + ny * k);

    if (cellType[idxL] == (int)cellType::FLUID || cellType[idxR] == (int)cellType::FLUID)
        u[i + (nx + 1) * (j + ny * k)] -= scale * (p[idxR] - p[idxL]);
}

__global__ void applyVGradientKernel(float *v, const float *p, const int *cellType, int nx, int ny, int nz, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j > ny || k >= nz)
        return;

    if (j == 0 || j == ny)
    {
        v[i + nx * (j + (ny + 1) * k)] = 0.0f;
        return;
    }

    int idxB = i + nx * ((j - 1) + ny * k);
    int idxT = i + nx * (j + ny * k);

    if (cellType[idxB] == (int)cellType::FLUID || cellType[idxT] == (int)cellType::FLUID)
        v[i + nx * (j + (ny + 1) * k)] -= scale * (p[idxT] - p[idxB]);
}

__global__ void applyWGradientKernel(float *w, const float *p, const int *cellType, int nx, int ny, int nz, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k > nz)
        return;

    if (k == 0 || k == nz)
    {
        w[i + nx * (j + ny * k)] = 0.0f;
        return;
    }

    int idxBk = i + nx * (j + ny * (k - 1));
    int idxFr = i + nx * (j + ny * k);

    if (cellType[idxBk] == (int)cellType::FLUID || cellType[idxFr] == (int)cellType::FLUID)
        w[i + nx * (j + ny * k)] -= scale * (p[idxFr] - p[idxBk]);
}

void cudaParticleSimulator::applyPressureGradient()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;

    // scale = dt / (ρ * dx)
    float scale = deltaTime / (fluidProps.density * grid_size.cellSize);

    dim3 block(8, 8, 4);

    dim3 gridU((nx + 1 + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    applyUGradientKernel<<<gridU, block>>>(grid_data.u, grid_data.p, grid_data.cellType, nx, ny, nz, scale);

    dim3 gridV((nx + 7) / 8, (ny + 1 + 7) / 8, (nz + 3) / 4);
    applyVGradientKernel<<<gridV, block>>>(grid_data.v, grid_data.p, grid_data.cellType, nx, ny, nz, scale);

    dim3 gridW((nx + 7) / 8, (ny + 7) / 8, (nz + 1 + 3) / 4);
    applyWGradientKernel<<<gridW, block>>>(grid_data.w, grid_data.p, grid_data.cellType, nx, ny, nz, scale);

    CHECK(cudaStreamSynchronize(this->stream));
}

