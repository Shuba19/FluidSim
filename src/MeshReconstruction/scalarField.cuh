#pragma once
#ifndef SCALAR_FIELD_BUILDER_CUH
#define SCALAR_FIELD_BUILDER_CUH
#include "../simulation/cudaCommons.h"
#include "../vulkan/utils.h"
#include "neighborIndex.cuh"   


class ScalarFieldBuilder
{
    float *d_field;          // device buffer, size = nx*ny*nz
    gridSize grid;
    float baseRadius;        // h: neighbour-search radius AND base kernel support
    int   capacity;          // max particles the per-particle buffers can hold
    ReconParams params;

   
    float3 *d_smoothedPos;   
    void   *d_anisotropy;    

    cudaStream_t stream;     

    const int* d_simCellType = nullptr;
    gridSize   simGrid;

public:
    ScalarFieldBuilder() : d_field(nullptr), baseRadius(0), capacity(0),
                           d_smoothedPos(nullptr), d_anisotropy(nullptr), stream(0),
                           d_instanceBuffer(nullptr), d_instanceCounter(nullptr) {}
    ScalarFieldBuilder(gridSize grid, float baseRadius, int maxParticles);
    ~ScalarFieldBuilder();

    ScalarFieldBuilder(const ScalarFieldBuilder&) = delete;
    ScalarFieldBuilder& operator=(const ScalarFieldBuilder&) = delete;

    ScalarFieldBuilder(ScalarFieldBuilder&& o) noexcept
        : d_field(o.d_field), grid(o.grid), baseRadius(o.baseRadius),
          capacity(o.capacity), params(o.params),
          d_smoothedPos(o.d_smoothedPos), d_anisotropy(o.d_anisotropy), stream(o.stream),
          d_instanceBuffer(o.d_instanceBuffer), d_instanceCounter(o.d_instanceCounter),
          d_simCellType(o.d_simCellType), simGrid(o.simGrid)
    {
        o.d_field = nullptr; o.d_smoothedPos = nullptr; o.d_anisotropy = nullptr;
        o.d_instanceBuffer = nullptr; o.d_instanceCounter = nullptr;
        o.d_simCellType = nullptr;
    }

    ScalarFieldBuilder& operator=(ScalarFieldBuilder&& o) noexcept {
        if (this != &o) {
            if (d_field)           cudaFree(d_field);
            if (d_smoothedPos)     cudaFree(d_smoothedPos);
            if (d_anisotropy)      cudaFree(d_anisotropy);
            if (d_instanceBuffer)  cudaFree(d_instanceBuffer);
            if (d_instanceCounter) cudaFree(d_instanceCounter);
            d_field = o.d_field; grid = o.grid; baseRadius = o.baseRadius;
            capacity = o.capacity; params = o.params;
            d_smoothedPos = o.d_smoothedPos; d_anisotropy = o.d_anisotropy; stream = o.stream;
            d_instanceBuffer = o.d_instanceBuffer; d_instanceCounter = o.d_instanceCounter;
            d_simCellType = o.d_simCellType; simGrid = o.simGrid;
            o.d_field = nullptr; o.d_smoothedPos = nullptr; o.d_anisotropy = nullptr;
            o.d_instanceBuffer = nullptr; o.d_instanceCounter = nullptr;
            o.d_simCellType = nullptr;
        }
        return *this;
    }

    void build(const NeighborGridView& nbr);

    void setParams(const ReconParams& p) { params = p; }
    ReconParams getParams() const { return params; }

    void setSimCellMask(const int* d, gridSize sg) { d_simCellType = d; simGrid = sg; }

    void setStream(cudaStream_t s) { stream = s; }

    void fillInstanceBuffer(void* mappedVkMemory, uint32_t& outCount,
                            uint32_t maxInstances,
                            float threshold = 0.1f, float maxValue = 3.0f);

   
    void packParticleInstances(float3* d_positions, int numParticles, void* mappedVkMemory);

    float *getField() const { return d_field; }
    int    fieldSize() const { return grid.gridCells; }

private:
    void* d_instanceBuffer  = nullptr; 
    int*  d_instanceCounter = nullptr;
};


#endif