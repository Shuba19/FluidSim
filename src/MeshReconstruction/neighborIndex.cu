#include "neighborIndex.cuh"

#include <cstdio>
#include <vector>

#include <cub/cub.cuh>

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    }

__global__ void niHashKernel(const float3* pos, int n,
                             float3 origin, float invCellSize, int3 gridDim,
                             int* cellHash, int* particleIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int3 c = niCellCoord(pos[i], origin, invCellSize, gridDim);
    cellHash[i]      = niFlatten(c, gridDim);
    particleIndex[i] = i;
}


__global__ void niReorderPosKernel(const float3* pos, const int* sortedIndex, int n,
                                   float3* sortedPos)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    sortedPos[s] = pos[sortedIndex[s]];
}

__global__ void niFindCellBoundsKernel(const int* sortedHash, int n,
                                       int* cellStart, int* cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int hash = sortedHash[i];
    if (i == 0 || sortedHash[i - 1] != hash) {
        cellStart[hash] = i;                        // first particle of this cell
        if (i > 0) cellEnd[sortedHash[i - 1]] = i;  // one past the previous cell
    }
    if (i == n - 1)
        cellEnd[hash] = n;                          // close the last cell
}

NeighborIndex::NeighborIndex(float3 domainOrigin, int3 gridDim, float cellSize, int maxParticles)
    : d_cellHash(nullptr), d_particleIndex(nullptr), d_sortedPos(nullptr),
      d_cellHashAlt(nullptr), d_particleIndexAlt(nullptr),
      d_tempStorage(nullptr), tempStorageBytes(0), sortEndBit(0),
      d_cellStart(nullptr), d_cellEnd(nullptr),
      gridDim(gridDim), cellSize(cellSize), invCellSize(1.0f / cellSize),
      domainOrigin(domainOrigin),
      numCells(gridDim.x * gridDim.y * gridDim.z),
      capacity(maxParticles), activeParticles(0), stream(0)
{
    CHECK(cudaMalloc(&d_cellHash,         (size_t)capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_particleIndex,    (size_t)capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_sortedPos,        (size_t)capacity * sizeof(float3)));
    CHECK(cudaMalloc(&d_cellHashAlt,      (size_t)capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_particleIndexAlt, (size_t)capacity * sizeof(int)));
    CHECK(cudaMalloc(&d_cellStart,        (size_t)numCells * sizeof(int)));
    CHECK(cudaMalloc(&d_cellEnd,          (size_t)numCells * sizeof(int)));

    unsigned int maxHash = (numCells > 0) ? (unsigned int)(numCells - 1) : 0u;
    sortEndBit = 0;
    while (maxHash > 0u) { maxHash >>= 1; ++sortEndBit; }
    if (sortEndBit < 1)  sortEndBit = 1;
    if (sortEndBit > 32) sortEndBit = 32;

    cub::DeviceRadixSort::SortPairs(
        nullptr, tempStorageBytes,
        d_cellHashAlt, d_cellHash, d_particleIndexAlt, d_particleIndex,
        capacity, 0, sortEndBit);
    CHECK(cudaMalloc(&d_tempStorage, tempStorageBytes));
}

NeighborIndex::~NeighborIndex()
{
    if (d_cellHash)         cudaFree(d_cellHash);
    if (d_particleIndex)    cudaFree(d_particleIndex);
    if (d_sortedPos)        cudaFree(d_sortedPos);
    if (d_cellHashAlt)      cudaFree(d_cellHashAlt);
    if (d_particleIndexAlt) cudaFree(d_particleIndexAlt);
    if (d_tempStorage)      cudaFree(d_tempStorage);
    if (d_cellStart)        cudaFree(d_cellStart);
    if (d_cellEnd)          cudaFree(d_cellEnd);
}

void NeighborIndex::build(const float3* d_positions, int numParticles)
{
   
    CHECK(cudaMemsetAsync(d_cellStart, -1, (size_t)numCells * sizeof(int), stream));
    CHECK(cudaMemsetAsync(d_cellEnd,   -1, (size_t)numCells * sizeof(int), stream));
    activeParticles = 0;

    if (numParticles <= 0) return;
    if (numParticles > capacity) {
        printf("[NeighborIndex] build: numParticles %d exceeds capacity %d — skipped\n", numParticles, capacity);
        return;
    }
    activeParticles = numParticles;

    const int blockSize = 256;
    const int numBlocks = (numParticles + blockSize - 1) / blockSize;

    
    niHashKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_positions, numParticles, domainOrigin, invCellSize, gridDim,
        d_cellHashAlt, d_particleIndexAlt);

   
    size_t tmpBytes = tempStorageBytes;
    cub::DeviceRadixSort::SortPairs(
        d_tempStorage, tmpBytes,
        d_cellHashAlt, d_cellHash, d_particleIndexAlt, d_particleIndex,
        numParticles, 0, sortEndBit, stream);

    niReorderPosKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_positions, d_particleIndex, numParticles, d_sortedPos);

    niFindCellBoundsKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_cellHash, numParticles, d_cellStart, d_cellEnd);

}

bool NeighborIndex::debugValidate() const
{
    std::vector<int> hStart(numCells), hEnd(numCells);
    CHECK(cudaMemcpy(hStart.data(), d_cellStart, (size_t)numCells * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hEnd.data(),   d_cellEnd,   (size_t)numCells * sizeof(int), cudaMemcpyDeviceToHost));

    long long binned = 0;
    int nonEmpty = 0;
    for (int h = 0; h < numCells; ++h) {
        if (hStart[h] < 0) continue;            // empty cell (sentinel -1)
        ++nonEmpty;
        int count = hEnd[h] - hStart[h];
        if (hEnd[h] < 0 || count <= 0) {        // a populated cell must have a valid range
            printf("[NeighborIndex] VALIDATION FAILED: cell %d has start=%d end=%d\n",
                   h, hStart[h], hEnd[h]);
            return false;
        }
        binned += count;
    }

    bool ok = (binned == (long long)activeParticles);
    printf("[NeighborIndex] binned %lld / %d particles across %d non-empty cells (of %d) — %s\n",
           binned, activeParticles, nonEmpty, numCells, ok ? "OK" : "MISMATCH");
    return ok;
}
