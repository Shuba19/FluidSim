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
__global__ void applyExternalForcesKernel(
    float *u, float *v, float *w,
    int *cellType,
    int nx, int ny, int nz,
    float fx, float fy, float fz,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int cidx = i + nx * (j + ny * k);
    if (cellType[cidx] != (int)cellType::FLUID) return;

    if (i + 1 <= nx) {
        int uidx = i + (nx + 1) * (j + ny * k);
        u[uidx] += fx * dt;
    }

    if (j + 1 <= ny) {
        int vidx = i + nx * (j + (ny + 1) * k);
        v[vidx] += (fy - 9.81f) * dt;
    }

    if (k + 1 <= nz) {
        int widx = i + nx * (j + ny * k);
        w[widx] += fz * dt;
    }
}

void cudaParticleSimulator::applyExternalForces()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;

    dim3 block(8, 8, 4);
    dim3 grid(
        (nx + block.x - 1) / block.x,
        (ny + 1 + block.y - 1) / block.y,
        (nz + block.z - 1) / block.z);
    float fx = 0, fy = -9.81f, fz = 0; 
    if(currentTime >= env.externalForce.time_start && currentTime <= env.externalForce.time_end) {
        fx += env.externalForce.x;
        fy += env.externalForce.y;
        fz += env.externalForce.z;
    }


    applyExternalForcesKernel<<<grid, block>>>(
        grid_data.u, grid_data.v, grid_data.w,
        grid_data.cellType,
        nx, ny, nz,
        fx, fy, fz,
        deltaTime);

    CHECK(cudaStreamSynchronize(this->stream));
}