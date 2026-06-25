#pragma once
#include "../simulation/cudaCommons.h"
#include "../vulkan/utils.h"

class MarchingCubesExtractor
{
    void* d_vertices;   // device buffer of MCVertex, size = maxVertices
    void* d_grad;       // device buffer of float3 gradients, size = maxFieldCells
    int*  d_counter;    // device-side atomic vertex counter
    int   maxVertices;
    int   maxFieldCells; // capacity of d_grad (>= grid.gridCells of any extract)
    cudaStream_t stream; // non-owning; the shared reconstruction stream (see setStream)

public:
    MarchingCubesExtractor()
        : d_vertices(nullptr), d_grad(nullptr), d_counter(nullptr),
          maxVertices(0), maxFieldCells(0), stream(0) {}
    MarchingCubesExtractor(int maxVerts, int maxFieldCells);
    ~MarchingCubesExtractor();

    MarchingCubesExtractor(const MarchingCubesExtractor&) = delete;
    MarchingCubesExtractor& operator=(const MarchingCubesExtractor&) = delete;

    MarchingCubesExtractor(MarchingCubesExtractor&& o) noexcept
        : d_vertices(o.d_vertices), d_grad(o.d_grad), d_counter(o.d_counter),
          maxVertices(o.maxVertices), maxFieldCells(o.maxFieldCells), stream(o.stream)
    {
        o.d_vertices = nullptr; o.d_grad = nullptr; o.d_counter = nullptr;
        o.maxVertices = 0; o.maxFieldCells = 0;
    }

    MarchingCubesExtractor& operator=(MarchingCubesExtractor&& o) noexcept {
        if (this != &o) {
            if (d_vertices) cudaFree(d_vertices);
            if (d_grad)     cudaFree(d_grad);
            if (d_counter)  cudaFree(d_counter);
            d_vertices = o.d_vertices; d_grad = o.d_grad; d_counter = o.d_counter;
            maxVertices = o.maxVertices; maxFieldCells = o.maxFieldCells; stream = o.stream;
            o.d_vertices = nullptr; o.d_grad = nullptr; o.d_counter = nullptr;
            o.maxVertices = 0; o.maxFieldCells = 0;
        }
        return *this;
    }

    void setStream(cudaStream_t s) { stream = s; }

    int extract(const float* d_field, gridSize grid, float isovalue);

    void copyToBuffer(void* dst, int vertexCount);
};
