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

__global__ void jacobiKernel(const float *pSrc, float *pDst, const float *divergence, const int *cellType, int nx, int ny, int nz, float scale);

__global__ void restrictKernel(const float *fine, float *coarse, int nxF, int nyF, int nzF)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int nxC = nxF / 2, nyC = nyF / 2, nzC = nzF / 2;
    if (i >= nxC || j >= nyC || k >= nzC)
        return;

    int fi = i * 2, fj = j * 2, fk = k * 2;
    float sum = 0.0f;
    int count = 0;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
            {
                int ni = fi + di, nj = fj + dj, nk = fk + dk;
                if (ni < nxF && nj < nyF && nk < nzF)
                {
                    sum += fine[ni + nxF * (nj + nyF * nk)];
                    count++;
                }
            }
    coarse[i + nxC * (j + nyC * k)] = sum / count;
}

__global__ void prolongateKernel(const float *coarse, float *fine, int nxF, int nyF, int nzF)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nxF || j >= nyF || k >= nzF)
        return;

    int nxC = nxF / 2, nyC = nyF / 2, nzC = nzF / 2;
    int ci = min(i / 2, nxC - 1), cj = min(j / 2, nyC - 1), ck = min(k / 2, nzC - 1);
    fine[i + nxF * (j + nyF * k)] += coarse[ci + nxC * (cj + nyC * ck)];
}

__global__ void restrictCellTypeKernel(const int *fine, int *coarse, int nxF, int nyF, int nzF)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int nxC = nxF / 2, nyC = nyF / 2, nzC = nzF / 2;
    if (i >= nxC || j >= nyC || k >= nzC)
        return;

    bool hasFluid = false, hasSolid = false;
    for (int di = 0; di < 2; di++)
        for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++)
            {
                int ni = i * 2 + di, nj = j * 2 + dj, nk = k * 2 + dk;
                if (ni < nxF && nj < nyF && nk < nzF)
                {
                    int t = fine[ni + nxF * (nj + nyF * nk)];
                    if (t == (int)cellType::FLUID)
                        hasFluid = true;
                    if (t == (int)cellType::SOLID)
                        hasSolid = true;
                }
            }
    coarse[i + nxC * (j + nyC * k)] = hasSolid ? (int)cellType::SOLID : hasFluid ? (int)cellType::FLUID
                                                                                 : (int)cellType::AIR;
}

void cudaParticleSimulator::restrictGrid(const float *fine, float *coarse, int nxF, int nyF, int nzF)
{
    dim3 block(8, 8, 4);
    dim3 grid((nxF / 2 + 7) / 8, (nyF / 2 + 7) / 8, (nzF / 2 + 3) / 4);
    restrictKernel<<<grid, block, 0, stream>>>(fine, coarse, nxF, nyF, nzF);
}

void cudaParticleSimulator::prolongateGrid(const float *coarse, float *fine, int nxF, int nyF, int nzF)
{
    dim3 block(8, 8, 4);
    dim3 grid((nxF + 7) / 8, (nyF + 7) / 8, (nzF + 3) / 4);
    prolongateKernel<<<grid, block, 0, stream>>>(coarse, fine, nxF, nyF, nzF);
}

void cudaParticleSimulator::smoothJacobi(float *p, float *pBuf, float *div, int *cellType, int nx, int ny, int nz, int iters)
{
    float scale = fluidProps.density * grid_size.cellSize * grid_size.cellSize / deltaTime;
    dim3 block(8, 8, 4);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    float *src = p;
    float *dst = pBuf;
    for (int i = 0; i < iters; i++)
    {
        jacobiKernel<<<grid, block, 0, stream>>>(src, dst, div, cellType, nx, ny, nz, scale);
        CHECK(cudaStreamSynchronize(stream));
        float *tmp = src;
        src = dst;
        dst = tmp;
    }
    if (src != p)
        CHECK(cudaMemcpyAsync(p, src, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

void cudaParticleSimulator::multigridVCycle(float *p, float *div, int *cellType, int nx, int ny, int nz)
{
    dim3 block(8, 8, 4);
    int nxC = nx / 2, nyC = ny / 2, nzC = nz / 2;
    int jacobiIters = 2; // Number of Jacobi iterations at each level
    // pre-smooth
    smoothJacobi(p, grid_data.pBuffer, div, cellType, nx, ny, nz, jacobiIters);

    // restrict divergence e cellType
    restrictGrid(div, coarse_div, nx, ny, nz);
    dim3 gridC((nxC + 7) / 8, (nyC + 7) / 8, (nzC + 3) / 4);
    restrictCellTypeKernel<<<gridC, block, 0, stream>>>(cellType, coarse_cellType, nx, ny, nz); // mancava
    CHECK(cudaStreamSynchronize(stream));
    // reset pressione  e solve
    CHECK(cudaMemsetAsync(coarse_p, 0, nxC * nyC * nzC * sizeof(float), stream));
    CHECK(cudaMemsetAsync(coarse_buf, 0, nxC * nyC * nzC * sizeof(float), stream));
    smoothJacobi(coarse_p, coarse_buf, coarse_div, coarse_cellType, nxC, nyC, nzC, jacobiIters*2);

    // prolongate e correggi
    prolongateGrid(coarse_p, p, nx, ny, nz);
    CHECK(cudaStreamSynchronize(stream));

    // post-smooth
    smoothJacobi(p, grid_data.pBuffer, div, cellType, nx, ny, nz, jacobiIters);
}