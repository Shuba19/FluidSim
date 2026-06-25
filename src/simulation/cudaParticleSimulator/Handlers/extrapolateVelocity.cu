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

__global__ void extrapolateUKernel(float* u, const int* cellType, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j >= ny || k >= nz) return;

    int uidx = i + (nx + 1) * (j + ny * k);
    if (i == 0 || i == nx) return; 

    int idxL = (i - 1) + nx * (j + ny * k);
    int idxR = i + nx * (j + ny * k);

    if (cellType[idxL] != 1 && cellType[idxR] != 1) {
        if (i > 1 && cellType[(i - 2) + nx * (j + ny * k)] == 1) {
            u[uidx] = u[(i - 1) + (nx + 1) * (j + ny * k)];
        } else if (i < nx - 1 && cellType[(i + 1) + nx * (j + ny * k)] == 1) {
            u[uidx] = u[(i + 1) + (nx + 1) * (j + ny * k)];
        }
    }
}

__global__ void extrapolateVKernel(float* v, const int* cellType, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j > ny || k >= nz) return;

    int vidx = i + nx * (j + (ny + 1) * k);
    if (j == 0 || j == ny) return;

    int idxB = i + nx * ((j - 1) + ny * k);
    int idxT = i + nx * (j + ny * k);

    if (cellType[idxB] != 1 && cellType[idxT] != 1) {
        if (j > 1 && cellType[i + nx * ((j - 2) + ny * k)] == 1) {
            v[vidx] = v[i + nx * ((j - 1) + (ny + 1) * k)];
        } else if (j < ny - 1 && cellType[i + nx * ((j + 1) + ny * k)] == 1) {
            v[vidx] = v[i + nx * ((j + 1) + (ny + 1) * k)];
        }
    }
}

__global__ void extrapolateWKernel(float* w, const int* cellType, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k > nz) return;

    int widx = i + nx * (j + ny * k);
    if (k == 0 || k == nz) return;

    int idxBk = i + nx * (j + ny * (k - 1));
    int idxFr = i + nx * (j + ny * k);

    if (cellType[idxBk] != 1 && cellType[idxFr] != 1) {
        if (k > 1 && cellType[i + nx * (j + ny * (k - 2))] == 1) {
            w[widx] = w[i + nx * (j + ny * (k - 1))];
        } else if (k < nz - 1 && cellType[i + nx * (j + ny * (k + 1))] == 1) {
            w[widx] = w[i + nx * (j + ny * (k + 1))];
        }
    }
}

void cudaParticleSimulator::extrapolateVelocity() {
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    dim3 block(8, 8, 4);
    CHECK(cudaStreamSynchronize(stream));
    dim3 gridU((nx + 1 + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    extrapolateUKernel<<<gridU, block, 0, uStream>>>(grid_data.u, grid_data.cellType, nx, ny, nz);
    extrapolateUKernel<<<gridU, block, 0, uStream>>>(grid_data.u, grid_data.cellType, nx, ny, nz);

    dim3 gridV((nx + 7) / 8, (ny + 1 + 7) / 8, (nz + 3) / 4);
    extrapolateVKernel<<<gridV, block, 0, vStream>>>(grid_data.v, grid_data.cellType, nx, ny, nz);
    extrapolateVKernel<<<gridV, block, 0, vStream>>>(grid_data.v, grid_data.cellType, nx, ny, nz);

    dim3 gridW((nx + 7) / 8, (ny + 7) / 8, (nz + 1 + 3) / 4);
    extrapolateWKernel<<<gridW, block, 0, wStream>>>(grid_data.w, grid_data.cellType, nx, ny, nz);
    extrapolateWKernel<<<gridW, block, 0, wStream>>>(grid_data.w, grid_data.cellType, nx, ny, nz);

    CHECK(cudaStreamSynchronize(uStream));
    CHECK(cudaStreamSynchronize(vStream));
    CHECK(cudaStreamSynchronize(wStream));
}
