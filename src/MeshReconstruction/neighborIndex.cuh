#pragma once



#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

struct NeighborGridView {
    const int*    cellStart;
    const int*    cellEnd;
    const int*    sortedIndex;
    const float3* sortedPos;
    float3 origin;
    float  cellSize;
    float  invCellSize;
    int3   gridDim;
    int    numCells;
    int    numActive;
};

class NeighborIndex
{
    
    int*    d_cellHash;       // hash of the cell each particle falls in (sorted output)
    int*    d_particleIndex;  // particle ids, sorted alongside d_cellHash (sorted output)
    float3* d_sortedPos;      // positions reordered into sorted-slot order

    
    int*    d_cellHashAlt;
    int*    d_particleIndexAlt;
    void*   d_tempStorage;
    size_t  tempStorageBytes;
    int     sortEndBit;      

   
    int* d_cellStart;      // first sorted slot of the cell, -1 if empty
    int* d_cellEnd;        // one past the last sorted slot of the cell

   
    int3   gridDim;        // cells per axis
    float  cellSize;       // cell edge length (sized on the reconstruction radius)
    float  invCellSize;    // 1/cellSize, precomputed (hot path uses a multiply)
    float3 domainOrigin;   // world position of the (0,0,0) cell's lower corner
    int    numCells;

    int    capacity;        // max particles the per-particle buffers can hold
    int    activeParticles; // particles binned by the last build()

    cudaStream_t stream;    // non-owning; the shared reconstruction stream (see setStream)

public:
    NeighborIndex()
        : d_cellHash(nullptr), d_particleIndex(nullptr), d_sortedPos(nullptr),
          d_cellHashAlt(nullptr), d_particleIndexAlt(nullptr),
          d_tempStorage(nullptr), tempStorageBytes(0), sortEndBit(0),
          d_cellStart(nullptr), d_cellEnd(nullptr),
          gridDim(make_int3(0, 0, 0)), cellSize(0.f), invCellSize(0.f),
          domainOrigin(make_float3(0.f, 0.f, 0.f)), numCells(0),
          capacity(0), activeParticles(0), stream(0) {}

   
    NeighborIndex(float3 domainOrigin, int3 gridDim, float cellSize, int maxParticles);
    ~NeighborIndex();

    NeighborIndex(const NeighborIndex&) = delete;
    NeighborIndex& operator=(const NeighborIndex&) = delete;

    NeighborIndex(NeighborIndex&& o) noexcept
        : d_cellHash(o.d_cellHash), d_particleIndex(o.d_particleIndex),
          d_sortedPos(o.d_sortedPos),
          d_cellHashAlt(o.d_cellHashAlt), d_particleIndexAlt(o.d_particleIndexAlt),
          d_tempStorage(o.d_tempStorage), tempStorageBytes(o.tempStorageBytes),
          sortEndBit(o.sortEndBit),
          d_cellStart(o.d_cellStart), d_cellEnd(o.d_cellEnd),
          gridDim(o.gridDim), cellSize(o.cellSize), invCellSize(o.invCellSize),
          domainOrigin(o.domainOrigin),
          numCells(o.numCells), capacity(o.capacity), activeParticles(o.activeParticles),
          stream(o.stream)
    {
        o.d_cellHash = nullptr; o.d_particleIndex = nullptr; o.d_sortedPos = nullptr;
        o.d_cellHashAlt = nullptr; o.d_particleIndexAlt = nullptr;
        o.d_tempStorage = nullptr; o.tempStorageBytes = 0;
        o.d_cellStart = nullptr; o.d_cellEnd = nullptr;
    }

    NeighborIndex& operator=(NeighborIndex&& o) noexcept {
        if (this != &o) {
            if (d_cellHash)         cudaFree(d_cellHash);
            if (d_particleIndex)    cudaFree(d_particleIndex);
            if (d_sortedPos)        cudaFree(d_sortedPos);
            if (d_cellHashAlt)      cudaFree(d_cellHashAlt);
            if (d_particleIndexAlt) cudaFree(d_particleIndexAlt);
            if (d_tempStorage)      cudaFree(d_tempStorage);
            if (d_cellStart)        cudaFree(d_cellStart);
            if (d_cellEnd)          cudaFree(d_cellEnd);
            d_cellHash = o.d_cellHash; d_particleIndex = o.d_particleIndex;
            d_sortedPos = o.d_sortedPos;
            d_cellHashAlt = o.d_cellHashAlt; d_particleIndexAlt = o.d_particleIndexAlt;
            d_tempStorage = o.d_tempStorage; tempStorageBytes = o.tempStorageBytes;
            sortEndBit = o.sortEndBit;
            d_cellStart = o.d_cellStart; d_cellEnd = o.d_cellEnd;
            gridDim = o.gridDim; cellSize = o.cellSize; invCellSize = o.invCellSize;
            domainOrigin = o.domainOrigin;
            numCells = o.numCells; capacity = o.capacity; activeParticles = o.activeParticles;
            stream = o.stream;
            o.d_cellHash = nullptr; o.d_particleIndex = nullptr; o.d_sortedPos = nullptr;
            o.d_cellHashAlt = nullptr; o.d_particleIndexAlt = nullptr;
            o.d_tempStorage = nullptr; o.tempStorageBytes = 0;
            o.d_cellStart = nullptr; o.d_cellEnd = nullptr;
        }
        return *this;
    }

    void setStream(cudaStream_t s) { stream = s; }

    void build(const float3* d_positions, int numParticles);

   
    const int*    cellStart()       const { return d_cellStart; }
    const int*    cellEnd()         const { return d_cellEnd; }
    const int*    sortedIndices()   const { return d_particleIndex; }
    const float3* sortedPositions() const { return d_sortedPos; }

    
    int3   gridDimensions() const { return gridDim; }
    float  cellSizeValue()  const { return cellSize; }
    float3 origin()         const { return domainOrigin; }
    int    cellCount()      const { return numCells; }

    
    NeighborGridView view() const {
        return NeighborGridView{ d_cellStart, d_cellEnd, d_particleIndex, d_sortedPos,
                                 domainOrigin, cellSize, invCellSize, gridDim, numCells,
                                 activeParticles };
    }

   
    bool debugValidate() const;
};

#if defined(__CUDACC__)

// Branch-free int clamp (avoids host/device min/max name ambiguity).
__device__ inline int niClampi(int v, int lo, int hi) {
    v = v < lo ? lo : v;
    v = v > hi ? hi : v;
    return v;
}


__device__ inline int3 niCellCoord(float3 p, float3 origin, float invCellSize, int3 gridDim) {
    int cx = (int)floorf((p.x - origin.x) * invCellSize);
    int cy = (int)floorf((p.y - origin.y) * invCellSize);
    int cz = (int)floorf((p.z - origin.z) * invCellSize);
    return make_int3(niClampi(cx, 0, gridDim.x - 1),
                     niClampi(cy, 0, gridDim.y - 1),
                     niClampi(cz, 0, gridDim.z - 1));
}

// Cell coords → flat hash in [0, numCells). Row-major: x outer, z inner.
__device__ inline int niFlatten(int3 c, int3 gridDim) {
    return (c.x * gridDim.y + c.y) * gridDim.z + c.z;
}

template <typename F>
__device__ inline void niForEachNeighbor(float3 p, const NeighborGridView& g, F callback)
{
    int3 c = niCellCoord(p, g.origin, g.invCellSize, g.gridDim);
    for (int dz = -1; dz <= 1; ++dz) {
        int z = c.z + dz;
        if (z < 0 || z >= g.gridDim.z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int y = c.y + dy;
            if (y < 0 || y >= g.gridDim.y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int x = c.x + dx;
                if (x < 0 || x >= g.gridDim.x) continue;

                int h = niFlatten(make_int3(x, y, z), g.gridDim);
                int start = g.cellStart[h];
                if (start < 0) continue;        // empty cell — explicitly skipped
                int end = g.cellEnd[h];
                for (int s = start; s < end; ++s)
                    callback(g.sortedIndex[s]);
            }
        }
    }
}

template <typename F>
__device__ inline void niForEachNeighborSlot(float3 p, const NeighborGridView& g, F callback)
{
    int3 c = niCellCoord(p, g.origin, g.invCellSize, g.gridDim);
    for (int dz = -1; dz <= 1; ++dz) {
        int z = c.z + dz;
        if (z < 0 || z >= g.gridDim.z) continue;
        for (int dy = -1; dy <= 1; ++dy) {
            int y = c.y + dy;
            if (y < 0 || y >= g.gridDim.y) continue;
            for (int dx = -1; dx <= 1; ++dx) {
                int x = c.x + dx;
                if (x < 0 || x >= g.gridDim.x) continue;

                int h = niFlatten(make_int3(x, y, z), g.gridDim);
                int start = g.cellStart[h];
                if (start < 0) continue;        // empty cell — explicitly skipped
                int end = g.cellEnd[h];
                for (int s = start; s < end; ++s)
                    callback(s);
            }
        }
    }
}

#endif 