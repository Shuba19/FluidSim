#include "marchingCubes.cuh"
#include <cstdio>

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
namespace cg = cooperative_groups;

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    }


struct MCVertexDevice {
    float px, py, pz;
    float nx, ny, nz;
};

#include "marchingCubesTables.cuh"

__device__ inline float sampleField(const float* field, int nx, int ny, int nz,
                                    int x, int y, int z)
{
    x = max(0, min(nx - 1, x));
    y = max(0, min(ny - 1, y));
    z = max(0, min(nz - 1, z));
    return field[x * ny * nz + y * nz + z];
}

__device__ inline float3 cornerGradient(const float* field, int nx, int ny, int nz,
                                        int x, int y, int z)
{
    float3 g;
    g.x = sampleField(field, nx, ny, nz, x + 1, y, z) - sampleField(field, nx, ny, nz, x - 1, y, z);
    g.y = sampleField(field, nx, ny, nz, x, y + 1, z) - sampleField(field, nx, ny, nz, x, y - 1, z);
    g.z = sampleField(field, nx, ny, nz, x, y, z + 1) - sampleField(field, nx, ny, nz, x, y, z - 1);
    return g;
}

__device__ inline float lerpT(float iso, float v1, float v2)
{
    return (fabsf(v2 - v1) > 1e-6f) ? (iso - v1) / (v2 - v1) : 0.5f;
}

__global__ void computeGradientKernel(
    const float* field, int nx, int ny, int nz, float3* grad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int z = idx % nz;
    int y = (idx / nz) % ny;
    int x = idx / (ny * nz);
    grad[idx] = cornerGradient(field, nx, ny, nz, x, y, z);
}

__global__ void marchingCubesKernel(
    const float* field, const float3* gradField,
    int nx, int ny, int nz,
    float cellSize, float3 origin, float iso,
    MCVertexDevice* out, int* counter, int maxOut)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= nx - 1 || iy >= ny - 1 || iz >= nz - 1) return;

    float  val[8];
    float3 pos[8];
    float3 grad[8];
    for (int c = 0; c < 8; ++c) {
        int cx = ix + cornerOffset[c][0];
        int cy = iy + cornerOffset[c][1];
        int cz = iz + cornerOffset[c][2];
        int cidx = cx * ny * nz + cy * nz + cz;
        val[c]  = field[cidx];
        pos[c]  = make_float3(origin.x + (cx + 0.5f) * cellSize,
                              origin.y + (cy + 0.5f) * cellSize,
                              origin.z + (cz + 0.5f) * cellSize);
        grad[c] = gradField[cidx];      
    }

    int cubeIndex = 0;
    for (int c = 0; c < 8; ++c)
        if (val[c] < iso) cubeIndex |= (1 << c);

    int edges = edgeTable[cubeIndex];
    if (edges == 0) return;

    float3 vertList[12];
    float3 normList[12];
    for (int e = 0; e < 12; ++e) {
        if (edges & (1 << e)) {
            int a = edgeConn[e][0], b = edgeConn[e][1];
            float t = lerpT(iso, val[a], val[b]);
            vertList[e] = make_float3(
                pos[a].x + t * (pos[b].x - pos[a].x),
                pos[a].y + t * (pos[b].y - pos[a].y),
                pos[a].z + t * (pos[b].z - pos[a].z));
            float3 g = make_float3(
                grad[a].x + t * (grad[b].x - grad[a].x),
                grad[a].y + t * (grad[b].y - grad[a].y),
                grad[a].z + t * (grad[b].z - grad[a].z));
            float len = sqrtf(g.x * g.x + g.y * g.y + g.z * g.z);
            if (len > 1e-8f) normList[e] = make_float3(-g.x / len, -g.y / len, -g.z / len);
            else             normList[e] = make_float3(0.f, 1.f, 0.f);
        }
    }

    int myVerts = 0;
    for (int t = 0; triTable[cubeIndex][t] != -1; t += 3)
        myVerts += 3;

    cg::coalesced_group active = cg::coalesced_threads();
    int prefix = cg::exclusive_scan(active, myVerts);          // verts before me
    int warpTotal = active.shfl(prefix + myVerts, active.num_threads() - 1);
    int base = 0;
    if (active.thread_rank() == 0)
        base = atomicAdd(counter, warpTotal);
    base = active.shfl(base, 0);
    int writeBase = base + prefix;

    int v = 0;
    for (int t = 0; triTable[cubeIndex][t] != -1; t += 3) {
        int dst = writeBase + v;
        if (dst + 2 < maxOut) {
            int es[3] = {triTable[cubeIndex][t], triTable[cubeIndex][t + 1], triTable[cubeIndex][t + 2]};
            for (int k = 0; k < 3; ++k) {
                float3 vv = vertList[es[k]];
                float3 nn = normList[es[k]];
                out[dst + k].px = vv.x; out[dst + k].py = vv.y; out[dst + k].pz = vv.z;
                out[dst + k].nx = nn.x; out[dst + k].ny = nn.y; out[dst + k].nz = nn.z;
            }
        }
        v += 3;
    }
}

MarchingCubesExtractor::MarchingCubesExtractor(int maxVerts, int maxFieldCells)
    : maxVertices(maxVerts), maxFieldCells(maxFieldCells), stream(0)
{
    CHECK(cudaMalloc(&d_vertices, (size_t)maxVertices * sizeof(MCVertexDevice)));
    CHECK(cudaMalloc(&d_grad,     (size_t)maxFieldCells * sizeof(float3)));
    CHECK(cudaMalloc(&d_counter, sizeof(int)));
}

MarchingCubesExtractor::~MarchingCubesExtractor()
{
    if (d_vertices) cudaFree(d_vertices);
    if (d_grad)     cudaFree(d_grad);
    if (d_counter)  cudaFree(d_counter);
}

int MarchingCubesExtractor::extract(const float* d_field, gridSize grid, float isovalue)
{
    CHECK(cudaMemsetAsync(d_counter, 0, sizeof(int), stream));

    // Pass 1 — precompute the field gradient once per grid point.
    const int totalCells = grid.gridCells;
    const int gradBlock  = 256;
    const int gradGrid   = (totalCells + gradBlock - 1) / gradBlock;
    computeGradientKernel<<<gradGrid, gradBlock, 0, stream>>>(
        d_field, grid.x, grid.y, grid.z, (float3*)d_grad);

    // Pass 2 — marching cubes, one thread per cell.
    dim3 block(8, 8, 8);
    dim3 gridDim(
        (grid.x - 1 + block.x - 1) / block.x,
        (grid.y - 1 + block.y - 1) / block.y,
        (grid.z - 1 + block.z - 1) / block.z);

    marchingCubesKernel<<<gridDim, block, 0, stream>>>(
        d_field, (const float3*)d_grad, grid.x, grid.y, grid.z,
        grid.cellSize, grid.origin, isovalue,
        (MCVertexDevice*)d_vertices, d_counter, maxVertices);

    // The vertex count is needed on the host to size the copy → sync the stream once.
    int count = 0;
    CHECK(cudaMemcpyAsync(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    return min(count, maxVertices);
}

void MarchingCubesExtractor::copyToBuffer(void* dst, int vertexCount)
{
    if (vertexCount <= 0) return;
    CHECK(cudaMemcpyAsync(dst, d_vertices,
                          (size_t)vertexCount * sizeof(MCVertexDevice),
                          cudaMemcpyDeviceToDevice, stream));
}
