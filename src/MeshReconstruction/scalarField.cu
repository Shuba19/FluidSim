#include "scalarField.cuh"
#include "sym3eig.cuh"
#include "../simulation/ChronoCuda/ChronoCuda.h" // timing per-kernel (diagnostica)
#include <cstdio>
#include <algorithm> // std::max (initializer-list overload)

#define CHECK(call)                                                             \
    {                                                                           \
        const cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                               \
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,              \
                   cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    }

struct Anisotropy {
    Sym3  G;       // symmetric 3x3: maps a world offset into kernel space
    float wnorm;   // det(G)/det(G_iso): per-particle weight, normalized to ~1
};

__device__ inline float  dot3(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline float3 sub3(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }

__device__ inline float3 sym3_apply(const Sym3& M, float3 v) {
    return make_float3(M.xx*v.x + M.xy*v.y + M.xz*v.z,
                       M.xy*v.x + M.yy*v.y + M.yz*v.z,
                       M.xz*v.x + M.yz*v.y + M.zz*v.z);
}

__device__ inline float niWeight(float r2, float invH) {
    float q = sqrtf(r2) * invH;          // r/h
    if (q >= 1.0f) return 0.0f;
    return 1.0f - q*q*q;
}

__global__ void splatSmoothKernel(NeighborGridView g, int n, float h, float lambda,
                                  float3* smoothedPos)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;

    float3 xi = g.sortedPos[s];
    float invH = 1.0f / h, h2 = h*h;

    float3 acc = make_float3(0.f, 0.f, 0.f);
    float  wsum = 0.f;
    niForEachNeighborSlot(xi, g, [&](int sj) {
        float3 xj = g.sortedPos[sj];
        float  r2 = dot3(sub3(xj, xi), sub3(xj, xi));
        if (r2 < h2) {
            float w = niWeight(r2, invH);
            acc.x += w * xj.x; acc.y += w * xj.y; acc.z += w * xj.z;
            wsum  += w;
        }
    });

    float3 mean = (wsum > 0.f)
        ? make_float3(acc.x/wsum, acc.y/wsum, acc.z/wsum)
        : xi;
    smoothedPos[s] = make_float3((1.f-lambda)*xi.x + lambda*mean.x,
                                 (1.f-lambda)*xi.y + lambda*mean.y,
                                 (1.f-lambda)*xi.z + lambda*mean.z);
}

__global__ void wpcaKernel(NeighborGridView g, int n, float h,
                           bool anisotropic, float kr, float kn, int nEps,
                           const float3* smoothedPos, Anisotropy* aniso)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;

    float3 xi = smoothedPos[s];
    float invH = 1.0f / h, h2 = h*h;

    float  sw = 0.f;
    float3 sm = make_float3(0.f, 0.f, 0.f);         
    float  Sxx=0.f, Syy=0.f, Szz=0.f, Sxy=0.f, Sxz=0.f, Syz=0.f; 
    int    count = 0;
    niForEachNeighborSlot(xi, g, [&](int sj) {
        float3 e = sub3(smoothedPos[sj], xi);
        float  r2 = dot3(e, e);
        if (r2 < h2) {
            float w = niWeight(r2, invH);
            sw += w; count++;
            sm.x += w*e.x; sm.y += w*e.y; sm.z += w*e.z;
            Sxx += w*e.x*e.x; Syy += w*e.y*e.y; Szz += w*e.z*e.z;
            Sxy += w*e.x*e.y; Sxz += w*e.x*e.z; Syz += w*e.y*e.z;
        }
    });

    Anisotropy out;

    bool useIsotropic = (!anisotropic) || (count <= nEps) || (sw <= 0.f);
    if (useIsotropic) {
        float gd = (anisotropic ? kn : 1.0f) / h;    
        out.G = Sym3{ gd, gd, gd, 0.f, 0.f, 0.f };
        out.wnorm = gd*gd*gd * (h*h*h);              
        aniso[s] = out;
        return;
    }

    
    float3 mo = make_float3(sm.x/sw, sm.y/sw, sm.z/sw);
    Sym3 C;
    C.xx = Sxx/sw - mo.x*mo.x; C.yy = Syy/sw - mo.y*mo.y; C.zz = Szz/sw - mo.z*mo.z;
    C.xy = Sxy/sw - mo.x*mo.y; C.xz = Sxz/sw - mo.x*mo.z; C.yz = Syz/sw - mo.y*mo.z;

    float eval[3]; float3 evec[3];
    sym3_eig(C, eval, evec);                         

    float s1 = eval[0];
    if (s1 <= 1e-12f) {                               
        float gd = 1.0f / h;
        out.G = Sym3{ gd, gd, gd, 0.f, 0.f, 0.f };
        out.wnorm = 1.0f;
        aniso[s] = out;
        return;
    }

    float invKr = 1.0f / kr;
    float rho1 = 1.0f;
    float rho2 = fmaxf(eval[1] / s1, invKr);
    float rho3 = fmaxf(eval[2] / s1, invKr);

    float d0 = 1.0f / (h * rho1);
    float d1 = 1.0f / (h * rho2);
    float d2 = 1.0f / (h * rho3);

   
    float3 e0 = evec[0], e1 = evec[1], e2 = evec[2];
    out.G.xx = d0*e0.x*e0.x + d1*e1.x*e1.x + d2*e2.x*e2.x;
    out.G.yy = d0*e0.y*e0.y + d1*e1.y*e1.y + d2*e2.y*e2.y;
    out.G.zz = d0*e0.z*e0.z + d1*e1.z*e1.z + d2*e2.z*e2.z;
    out.G.xy = d0*e0.x*e0.y + d1*e1.x*e1.y + d2*e2.x*e2.y;
    out.G.xz = d0*e0.x*e0.z + d1*e1.x*e1.z + d2*e2.x*e2.z;
    out.G.yz = d0*e0.y*e0.z + d1*e1.y*e1.z + d2*e2.y*e2.z;

    
    out.wnorm = 1.0f / (rho1 * rho2 * rho3);
    aniso[s] = out;
}

__global__ void gatherFieldKernel(NeighborGridView g, float* field,
                                  int nx, int ny, int nz, float cellSize, float3 origin,
                                  const float3* smoothedPos, const Anisotropy* aniso,
                                  float maxSupport2,
                                  const int* simCellType,
                                  int snx, int sny, int snz,
                                  float simCellSize, float3 simOrigin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (ny * nz);

    float3 x = make_float3(origin.x + (ix + 0.5f) * cellSize,
                           origin.y + (iy + 0.5f) * cellSize,
                           origin.z + (iz + 0.5f) * cellSize);

    
    if (simCellType) {
        float invSim = 1.0f / simCellSize;
        int si = (int)floorf((x.x - simOrigin.x) * invSim);
        int sj = (int)floorf((x.y - simOrigin.y) * invSim);
        int sk = (int)floorf((x.z - simOrigin.z) * invSim);
        bool hasFluid = false;
        for (int di = -1; di <= 1 && !hasFluid; ++di) {
            int ni = si + di;
            if (ni < 0 || ni >= snx) continue;
            for (int dj = -1; dj <= 1 && !hasFluid; ++dj) {
                int nj = sj + dj;
                if (nj < 0 || nj >= sny) continue;
                for (int dk = -1; dk <= 1 && !hasFluid; ++dk) {
                    int nk = sk + dk;
                    if (nk < 0 || nk >= snz) continue;
                    if (simCellType[ni + snx * (nj + sny * nk)] == (int)cellType::FLUID)
                        hasFluid = true;
                }
            }
        }
        if (!hasFluid) { field[idx] = 0.f; return; }
    }

    float acc = 0.f;
    niForEachNeighborSlot(x, g, [&](int sj) {
        
        float3 d = sub3(x, smoothedPos[sj]);
        float  r2 = dot3(d, d);
        if (r2 >= maxSupport2) return;
        float3 q = sym3_apply(aniso[sj].G, d);
        float  q2 = dot3(q, q);
        if (q2 < 1.0f) {
            float p = 1.0f - q2;
            acc += aniso[sj].wnorm * (p*p*p);
        }
    });

    field[idx] = acc;
}

struct SFInstanceDevice {
    float px, py, pz;
    float r, g, b, a;
};

__global__ void packParticleInstancesKernel(
    const float3* pos, SFInstanceDevice* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i].px = pos[i].x; out[i].py = pos[i].y; out[i].pz = pos[i].z;
    out[i].r = 1.f; out[i].g = 1.f; out[i].b = 1.f; out[i].a = 1.f;
}

__global__ void fillSFInstancesKernel(
    const float* field,
    int nx, int ny, int nz, float cellSize, float3 origin,
    float threshold, float maxValue,
    SFInstanceDevice* out, int* counter, int maxOut)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    float val = field[idx];
    if (val <= threshold) return;

    int slot = atomicAdd(counter, 1);
    if (slot >= maxOut) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (ny * nz);

    float t = fminf(val / maxValue, 1.0f);

    out[slot].px = origin.x + (ix + 0.5f) * cellSize;
    out[slot].py = origin.y + (iy + 0.5f) * cellSize;
    out[slot].pz = origin.z + (iz + 0.5f) * cellSize;
    out[slot].r = t; out[slot].g = t; out[slot].b = t; out[slot].a = t;
}


ScalarFieldBuilder::ScalarFieldBuilder(gridSize grid, float baseRadius, int maxParticles)
    : grid(grid), baseRadius(baseRadius), capacity(maxParticles)
{
    CHECK(cudaMalloc(&d_field, grid.gridCells * sizeof(float)));
    CHECK(cudaMalloc(&d_smoothedPos, (size_t)capacity * sizeof(float3)));
    CHECK(cudaMalloc(&d_anisotropy,  (size_t)capacity * sizeof(Anisotropy)));
    // Size covers SF-cell compaction (SF_MAX_INSTANCES), recon-grid cells, and particle packing (capacity).
    size_t instBufSize = std::max({(size_t)appState::SF_MAX_INSTANCES, (size_t)grid.gridCells, (size_t)capacity}) * sizeof(SFInstanceDevice);
    CHECK(cudaMalloc(&d_instanceBuffer, instBufSize));
    CHECK(cudaMalloc(&d_instanceCounter, sizeof(int)));
}

ScalarFieldBuilder::~ScalarFieldBuilder()
{
    if (d_field)           cudaFree(d_field);
    if (d_smoothedPos)     cudaFree(d_smoothedPos);
    if (d_anisotropy)      cudaFree(d_anisotropy);
    if (d_instanceBuffer)  cudaFree(d_instanceBuffer);
    if (d_instanceCounter) cudaFree(d_instanceCounter);
}

void ScalarFieldBuilder::build(const NeighborGridView& nbr)
{
    
    const int numParticles = nbr.numActive;

    if (numParticles <= 0) {
        CHECK(cudaMemsetAsync(d_field, 0, grid.gridCells * sizeof(float), stream));
        return;
    }
    if (numParticles > capacity) {
        printf("[ScalarFieldBuilder] build: numParticles %d exceeds capacity %d — skipped\n", numParticles, capacity);
        return;
    }

    const float h = baseRadius;
    const int blockSize = 256;
    const int nBlocksP  = (numParticles + blockSize - 1) / blockSize;

    
    float support = h;
    if (params.anisotropic && params.kn > 0.f && params.kn < 1.f)
        support = h / params.kn;
    const float maxSupport2 = support * support;

    
    static chrono_cuda* tSmooth = new chrono_cuda("sf_smooth", stream);
    static chrono_cuda* tWpca   = new chrono_cuda("sf_wpca",   stream);
    static chrono_cuda* tGather = new chrono_cuda("sf_gather", stream);
    static double accSmooth = 0.0, accWpca = 0.0, accGather = 0.0;
    static long   timedFrames = 0;
    static bool   havePrev = false, prevSmoothed = false;
    if (havePrev) {
        if (prevSmoothed) accSmooth += tSmooth->readElapsed();
        accWpca   += tWpca->readElapsed();
        accGather += tGather->readElapsed();
        ++timedFrames;
        if (timedFrames % 100 == 0)
            printf("[sf timing avg/%ld frames] smooth=%.3f  wpca=%.3f  gather=%.3f ms\n",
                   timedFrames, accSmooth / timedFrames, accWpca / timedFrames,
                   accGather / timedFrames);
    }

    Anisotropy* aniso = (Anisotropy*)d_anisotropy;

    
    const float3* posSrc;
    if (params.smoothing) {
        tSmooth->cc_start();
        splatSmoothKernel<<<nBlocksP, blockSize, 0, stream>>>(nbr, numParticles, h, params.lambda, d_smoothedPos);
        tSmooth->cc_stop_async();
        posSrc = d_smoothedPos;
    } else {
        posSrc = nbr.sortedPos;
    }

   
    tWpca->cc_start();
    wpcaKernel<<<nBlocksP, blockSize, 0, stream>>>(nbr, numParticles, h,
                                        params.anisotropic, params.kr, params.kn, params.nEps,
                                        posSrc, aniso);
    tWpca->cc_stop_async();

    
    const int totalCells = grid.gridCells;
    const int nBlocksC   = (totalCells + blockSize - 1) / blockSize;
    tGather->cc_start();
    gatherFieldKernel<<<nBlocksC, blockSize, 0, stream>>>(nbr, d_field,
                                               grid.x, grid.y, grid.z, grid.cellSize, grid.origin,
                                               posSrc, aniso,
                                               maxSupport2,
                                               d_simCellType,
                                               simGrid.x, simGrid.y, simGrid.z,
                                               simGrid.cellSize, simGrid.origin);
    tGather->cc_stop_async();

    havePrev = true;
    prevSmoothed = params.smoothing;

   
}

void ScalarFieldBuilder::fillInstanceBuffer(
    void* mappedVkMemory, uint32_t& outCount,
    uint32_t maxInstances,
    float threshold, float maxValue)
{
    CHECK(cudaMemsetAsync(d_instanceCounter, 0, sizeof(int), stream));

    int total     = grid.gridCells;
    int blockSize = 256;
    int gridDim   = (total + blockSize - 1) / blockSize;

    fillSFInstancesKernel<<<gridDim, blockSize, 0, stream>>>(
        d_field,
        grid.x, grid.y, grid.z, grid.cellSize, grid.origin,
        threshold, maxValue,
        (SFInstanceDevice*)d_instanceBuffer, d_instanceCounter,
        (int)maxInstances);

    
    int count = 0;
    CHECK(cudaMemcpyAsync(&count, d_instanceCounter, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));
    outCount = (uint32_t)min(count, (int)maxInstances);

    CHECK(cudaMemcpyAsync(mappedVkMemory, d_instanceBuffer,
                          outCount * sizeof(SFInstanceDevice),
                          cudaMemcpyDeviceToHost, stream));
}

void ScalarFieldBuilder::packParticleInstances(float3* d_positions, int numParticles,
                                               void* mappedVkMemory)
{
    int blockSize = 256;
    int gridDim   = (numParticles + blockSize - 1) / blockSize;
    packParticleInstancesKernel<<<gridDim, blockSize, 0, stream>>>(
        d_positions, (SFInstanceDevice*)d_instanceBuffer, numParticles);
    CHECK(cudaMemcpyAsync(mappedVkMemory, d_instanceBuffer,
                          numParticles * sizeof(SFInstanceDevice),
                          cudaMemcpyDeviceToHost, stream));
}
