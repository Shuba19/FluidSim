#include "../cudaParticleSimulator.h"



__global__ void markNarrowBandKernel(
    float3 *pos, int *cellType, int numParticles,
    int nx, int ny, int nz, float invDx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    float3 p = pos[idx];
    int cx = (int)(p.x * invDx);
    int cy = (int)(p.y * invDx);
    int cz = (int)(p.z * invDx);

    int radius = 2;

    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            for (int z = -radius; z <= radius; ++z) {
                int ni = cx + x;
                int nj = cy + y;
                int nk = cz + z;

                if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz) {
                    int cidx = ni + nx * (nj + ny * nk);
                    
                    if (cellType[cidx] != (int)cellType::SOLID) {
                        cellType[cidx] = (int)cellType::FLUID; 
                    }
                }
            }
        }
    }
}




void cudaParticleSimulator::markNarrowBand() {
    float3 *d_pos = this->deviceData.pos;
    int *d_cellType = this->grid_data.cellType;

    int nx = this->grid_size.x;
    int ny = this->grid_size.y;
    int nz = this->grid_size.z;
    float invDx = 1.0f / this->grid_size.cellSize;

    cudaMemsetAsync(d_cellType, (int)cellType::AIR, nx * ny * nz * sizeof(int), this->stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (this->numParticles + threadsPerBlock - 1) / threadsPerBlock;
    markNarrowBandKernel<<<blocksPerGrid, threadsPerBlock, 0 ,this->stream>>>(d_pos, d_cellType, numParticles, nx, ny, nz, invDx);

}