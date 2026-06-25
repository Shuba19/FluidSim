#include "../cudaParticleSimulator.h"
#include <cmath>

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

void cudaParticleSimulator::updateSystem()
{
    float invFPS = 1.0f / state->videoFPS;
    float num_steps = std::ceil(invFPS / deltaTime);

    for (int step = 0; step < (int)num_steps; ++step)
    {
        currentTime += deltaTime;
        //if (narrowBandMarked)
        //    markNarrowBand();
        p2g();
        saveGridVelocities();
        applyExternalForces();
        pressureSolve();
        g2p();
        computeAdvection();
    }
}

void cudaParticleSimulator::cuda2vulkan(uint32_t currentImage, appState &state)
{
    // printf("Transferring data from CUDA to Vulkan...\n");
    chrono_cuda t("cuda2vulkan");
    t.cc_start();
    std::vector<glm::vec3> positions(this->numParticles);
    CHECK(cudaMemcpy(state.instanceBuffersMapped[currentImage], deviceData.pos, this->numParticles * sizeof(float3), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    t.cc_stop(true);
}

__global__ void p2gNormalizeKernel(
    float *u, float *v, float *w,
    float *uW, float *vW, float *wW,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Normalize u
    int uSize = (nx + 1) * ny * nz;
    if (idx < uSize && uW[idx] > 1e-6f)
        u[idx] /= uW[idx];

    // Normalize v
    int vSize = nx * (ny + 1) * nz;
    if (idx < vSize && vW[idx] > 1e-6f)
        v[idx] /= vW[idx];

    // Normalize w
    int wSize = nx * ny * (nz + 1);
    if (idx < wSize && wW[idx] > 1e-6f)
        w[idx] /= wW[idx];
}

__global__ void p2gKernel(
    float3 *pos, float3 *vel,
    float *u, float *v, float *w,
    float *uW, float *vW, float *wW,
    int *ctype,
    int nx, int ny, int nz,
    float invDx,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;
    float3 p = pos[idx];
    float3 v_l = vel[idx];
    // coord
    float gx = p.x * invDx;
    float gy = p.y * invDx;
    float gz = p.z * invDx;
    // shift U
    {
        float ux = gx, uy = gy - 0.5f, uz = gz - 0.5f;
        int i = (int)ux, j = (int)uy, k = (int)uz;
        float fx = ux - i, fy = uy - j, fz = uz - k;

        // 8 pesi trilineari
        // i pesi stanno as significare ( quanto la particella è vicina al nodo)
        // il nodo essenzialmente è il vertice del cubo di riferimenti di ora
        // essendo una mac grid, i valori vanno shiftati ogni volta
        // questo perchè, altrimenti non sarebbe possibile analizzare la velocità in maniera corretta, visto che la velocità di u indipendente dalle altre e vicevesa
        float w000 = (1 - fx) * (1 - fy) * (1 - fz), w100 = fx * (1 - fy) * (1 - fz);
        float w010 = (1 - fx) * fy * (1 - fz), w110 = fx * fy * (1 - fz);
        float w001 = (1 - fx) * (1 - fy) * fz, w101 = fx * (1 - fy) * fz;
        float w011 = (1 - fx) * fy * fz, w111 = fx * fy * fz;

// Stride griglia u: (nx+1) * ny * nz
#define U_IDX(i, j, k) ((i) + (nx + 1) * ((j) + ny * (k)))

        // Clamp 
        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&u[U_IDX(i, j, k)], w000 * v_l.x);
            atomicAdd(&uW[U_IDX(i, j, k)], w000);
        }
        if (i + 1 <= nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&u[U_IDX(i + 1, j, k)], w100 * v_l.x);
            atomicAdd(&uW[U_IDX(i + 1, j, k)], w100);
        }
        if (i >= 0 && i < nx && j + 1 < ny && k >= 0 && k < nz)
        {
            atomicAdd(&u[U_IDX(i, j + 1, k)], w010 * v_l.x);
            atomicAdd(&uW[U_IDX(i, j + 1, k)], w010);
        }
        if (i + 1 <= nx && j + 1 < ny && k >= 0 && k < nz)
        {
            atomicAdd(&u[U_IDX(i + 1, j + 1, k)], w110 * v_l.x);
            atomicAdd(&uW[U_IDX(i + 1, j + 1, k)], w110);
        }
        if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 < nz)
        {
            atomicAdd(&u[U_IDX(i, j, k + 1)], w001 * v_l.x);
            atomicAdd(&uW[U_IDX(i, j, k + 1)], w001);
        }
        if (i + 1 <= nx && j >= 0 && j < ny && k + 1 < nz)
        {
            atomicAdd(&u[U_IDX(i + 1, j, k + 1)], w101 * v_l.x);
            atomicAdd(&uW[U_IDX(i + 1, j, k + 1)], w101);
        }
        if (i >= 0 && i < nx && j + 1 < ny && k + 1 < nz)
        {
            atomicAdd(&u[U_IDX(i, j + 1, k + 1)], w011 * v_l.x);
            atomicAdd(&uW[U_IDX(i, j + 1, k + 1)], w011);
        }
        if (i + 1 <= nx && j + 1 < ny && k + 1 < nz)
        {
            atomicAdd(&u[U_IDX(i + 1, j + 1, k + 1)], w111 * v_l.x);
            atomicAdd(&uW[U_IDX(i + 1, j + 1, k + 1)], w111);
        }
#undef U_IDX
    }
    // V
    {
        float vx = gx - 0.5f, vy = gy, vz = gz - 0.5f;
        int i = (int)vx, j = (int)vy, k = (int)vz;
        float fx = vx - i, fy = vy - j, fz = vz - k;

        float w000 = (1 - fx) * (1 - fy) * (1 - fz), w100 = fx * (1 - fy) * (1 - fz);
        float w010 = (1 - fx) * fy * (1 - fz), w110 = fx * fy * (1 - fz);
        float w001 = (1 - fx) * (1 - fy) * fz, w101 = fx * (1 - fy) * fz;
        float w011 = (1 - fx) * fy * fz, w111 = fx * fy * fz;

#define V_IDX(i, j, k) ((i) + nx * ((j) + (ny + 1) * (k)))

        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&v[V_IDX(i, j, k)], w000 * v_l.y);
            atomicAdd(&vW[V_IDX(i, j, k)], w000);
        }
        if (i + 1 < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&v[V_IDX(i + 1, j, k)], w100 * v_l.y);
            atomicAdd(&vW[V_IDX(i + 1, j, k)], w100);
        }
        if (i >= 0 && i < nx && j + 1 <= ny && k >= 0 && k < nz)
        {
            atomicAdd(&v[V_IDX(i, j + 1, k)], w010 * v_l.y);
            atomicAdd(&vW[V_IDX(i, j + 1, k)], w010);
        }
        if (i + 1 < nx && j + 1 <= ny && k >= 0 && k < nz)
        {
            atomicAdd(&v[V_IDX(i + 1, j + 1, k)], w110 * v_l.y);
            atomicAdd(&vW[V_IDX(i + 1, j + 1, k)], w110);
        }
        if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 < nz)
        {
            atomicAdd(&v[V_IDX(i, j, k + 1)], w001 * v_l.y);
            atomicAdd(&vW[V_IDX(i, j, k + 1)], w001);
        }
        if (i + 1 < nx && j >= 0 && j < ny && k + 1 < nz)
        {
            atomicAdd(&v[V_IDX(i + 1, j, k + 1)], w101 * v_l.y);
            atomicAdd(&vW[V_IDX(i + 1, j, k + 1)], w101);
        }
        if (i >= 0 && i < nx && j + 1 <= ny && k + 1 < nz)
        {
            atomicAdd(&v[V_IDX(i, j + 1, k + 1)], w011 * v_l.y);
            atomicAdd(&vW[V_IDX(i, j + 1, k + 1)], w011);
        }
        if (i + 1 < nx && j + 1 <= ny && k + 1 < nz)
        {
            atomicAdd(&v[V_IDX(i + 1, j + 1, k + 1)], w111 * v_l.y);
            atomicAdd(&vW[V_IDX(i + 1, j + 1, k + 1)], w111);
        }
#undef V_IDX
    }
    // W
    {
        float wx = gx - 0.5f, wy = gy - 0.5f, wz = gz;
        int i = (int)wx, j = (int)wy, k = (int)wz;
        float fx = wx - i, fy = wy - j, fz = wz - k;

        float w000 = (1 - fx) * (1 - fy) * (1 - fz), w100 = fx * (1 - fy) * (1 - fz);
        float w010 = (1 - fx) * fy * (1 - fz), w110 = fx * fy * (1 - fz);
        float w001 = (1 - fx) * (1 - fy) * fz, w101 = fx * (1 - fy) * fz;
        float w011 = (1 - fx) * fy * fz, w111 = fx * fy * fz;

#define W_IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

        if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&w[W_IDX(i, j, k)], w000 * v_l.z);
            atomicAdd(&wW[W_IDX(i, j, k)], w000);
        }
        if (i + 1 < nx && j >= 0 && j < ny && k >= 0 && k < nz)
        {
            atomicAdd(&w[W_IDX(i + 1, j, k)], w100 * v_l.z);
            atomicAdd(&wW[W_IDX(i + 1, j, k)], w100);
        }
        if (i >= 0 && i < nx && j + 1 < ny && k >= 0 && k < nz)
        {
            atomicAdd(&w[W_IDX(i, j + 1, k)], w010 * v_l.z);
            atomicAdd(&wW[W_IDX(i, j + 1, k)], w010);
        }
        if (i + 1 < nx && j + 1 < ny && k >= 0 && k < nz)
        {
            atomicAdd(&w[W_IDX(i + 1, j + 1, k)], w110 * v_l.z);
            atomicAdd(&wW[W_IDX(i + 1, j + 1, k)], w110);
        }
        if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 <= nz)
        {
            atomicAdd(&w[W_IDX(i, j, k + 1)], w001 * v_l.z);
            atomicAdd(&wW[W_IDX(i, j, k + 1)], w001);
        }
        if (i + 1 < nx && j >= 0 && j < ny && k + 1 <= nz)
        {
            atomicAdd(&w[W_IDX(i + 1, j, k + 1)], w101 * v_l.z);
            atomicAdd(&wW[W_IDX(i + 1, j, k + 1)], w101);
        }
        if (i >= 0 && i < nx && j + 1 < ny && k + 1 <= nz)
        {
            atomicAdd(&w[W_IDX(i, j + 1, k + 1)], w011 * v_l.z);
            atomicAdd(&wW[W_IDX(i, j + 1, k + 1)], w011);
        }
        if (i + 1 < nx && j + 1 < ny && k + 1 <= nz)
        {
            atomicAdd(&w[W_IDX(i + 1, j + 1, k + 1)], w111 * v_l.z);
            atomicAdd(&wW[W_IDX(i + 1, j + 1, k + 1)], w111);
        }
#undef W_IDX
    }

    int ci = (int)gx, cj = (int)gy, ck = (int)gz;
    if (ci >= 0 && ci < nx && cj >= 0 && cj < ny && ck >= 0 && ck < nz)
    {
        int cidx = ci + nx * (cj + ny * ck);
        if (ctype[cidx] != (int)cellType::SOLID)
            ctype[cidx] = (int)cellType::FLUID;
    }
}

__global__ void p2gKernel_U(
    const float3 *pos, const float3 *vel,
    float *u, float *uW,
    int nx, int ny, int nz, float invDx, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    float3 p = pos[idx];
    float vx = vel[idx].x;

    float gx = p.x * invDx;
    float gy = p.y * invDx;
    float gz = p.z * invDx;

    // Shift e coordinate specifiche per la faccia U
    float ux = gx, uy = gy - 0.5f, uz = gz - 0.5f;
    int i = (int)ux, j = (int)uy, k = (int)uz;
    float fx = ux - i, fy = uy - j, fz = uz - k;

    // Solo gli 8 pesi necessari per l'asse U
    float m_fx = 1.0f - fx, m_fy = 1.0f - fy, m_fz = 1.0f - fz;
    float w000 = m_fx * m_fy * m_fz, w100 = fx * m_fy * m_fz;
    float w010 = m_fx * fy * m_fz, w110 = fx * fy * m_fz;
    float w001 = m_fx * m_fy * fz, w101 = fx * m_fy * fz;
    float w011 = m_fx * fy * fz, w111 = fx * fy * fz;

#define U_IDX(i, j, k) ((i) + (nx + 1) * ((j) + ny * (k)))

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&u[U_IDX(i, j, k)], w000 * vx);
        atomicAdd(&uW[U_IDX(i, j, k)], w000);
    }
    if (i + 1 <= nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&u[U_IDX(i + 1, j, k)], w100 * vx);
        atomicAdd(&uW[U_IDX(i + 1, j, k)], w100);
    }
    if (i >= 0 && i < nx && j + 1 < ny && k >= 0 && k < nz)
    {
        atomicAdd(&u[U_IDX(i, j + 1, k)], w010 * vx);
        atomicAdd(&uW[U_IDX(i, j + 1, k)], w010);
    }
    if (i + 1 <= nx && j + 1 < ny && k >= 0 && k < nz)
    {
        atomicAdd(&u[U_IDX(i + 1, j + 1, k)], w110 * vx);
        atomicAdd(&uW[U_IDX(i + 1, j + 1, k)], w110);
    }
    if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 < nz)
    {
        atomicAdd(&u[U_IDX(i, j, k + 1)], w001 * vx);
        atomicAdd(&uW[U_IDX(i, j, k + 1)], w001);
    }
    if (i + 1 <= nx && j >= 0 && j < ny && k + 1 < nz)
    {
        atomicAdd(&u[U_IDX(i + 1, j, k + 1)], w101 * vx);
        atomicAdd(&uW[U_IDX(i + 1, j, k + 1)], w101);
    }
    if (i >= 0 && i < nx && j + 1 < ny && k + 1 < nz)
    {
        atomicAdd(&u[U_IDX(i, j + 1, k + 1)], w011 * vx);
        atomicAdd(&uW[U_IDX(i, j + 1, k + 1)], w011);
    }
    if (i + 1 <= nx && j + 1 < ny && k + 1 < nz)
    {
        atomicAdd(&u[U_IDX(i + 1, j + 1, k + 1)], w111 * vx);
        atomicAdd(&uW[U_IDX(i + 1, j + 1, k + 1)], w111);
    }
#undef U_IDX
}

__global__ void p2gKernel_V(
    const float3 *pos, const float3 *vel,
    float *v, float *vW,
    int nx, int ny, int nz, float invDx, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    float3 p = pos[idx];
    float vy = vel[idx].y;

    float gx = p.x * invDx;
    float gy = p.y * invDx;
    float gz = p.z * invDx;

    // Shift e coordinate specifiche per la faccia V (Y speculare)
    float vx = gx - 0.5f, vy_s = gy, vz = gz - 0.5f;
    int i = (int)vx, j = (int)vy_s, k = (int)vz;
    float fx = vx - i, fy = vy_s - j, fz = vz - k;

    // Gli 8 pesi necessari per l'asse V
    float m_fx = 1.0f - fx, m_fy = 1.0f - fy, m_fz = 1.0f - fz;
    float w000 = m_fx * m_fy * m_fz, w100 = fx * m_fy * m_fz;
    float w010 = m_fx * fy * m_fz, w110 = fx * fy * m_fz;
    float w001 = m_fx * m_fy * fz, w101 = fx * m_fy * fz;
    float w011 = m_fx * fy * fz, w111 = fx * fy * fz;

// Stride griglia v: nx * (ny+1) * nz
#define V_IDX(i, j, k) ((i) + nx * ((j) + (ny + 1) * (k)))

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&v[V_IDX(i, j, k)], w000 * vy);
        atomicAdd(&vW[V_IDX(i, j, k)], w000);
    }
    if (i + 1 < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&v[V_IDX(i + 1, j, k)], w100 * vy);
        atomicAdd(&vW[V_IDX(i + 1, j, k)], w100);
    }
    if (i >= 0 && i < nx && j + 1 <= ny && k >= 0 && k < nz)
    {
        atomicAdd(&v[V_IDX(i, j + 1, k)], w010 * vy);
        atomicAdd(&vW[V_IDX(i, j + 1, k)], w010);
    }
    if (i + 1 < nx && j + 1 <= ny && k >= 0 && k < nz)
    {
        atomicAdd(&v[V_IDX(i + 1, j + 1, k)], w110 * vy);
        atomicAdd(&vW[V_IDX(i + 1, j + 1, k)], w110);
    }
    if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 < nz)
    {
        atomicAdd(&v[V_IDX(i, j, k + 1)], w001 * vy);
        atomicAdd(&vW[V_IDX(i, j, k + 1)], w001);
    }
    if (i + 1 < nx && j >= 0 && j < ny && k + 1 < nz)
    {
        atomicAdd(&v[V_IDX(i + 1, j, k + 1)], w101 * vy);
        atomicAdd(&vW[V_IDX(i + 1, j, k + 1)], w101);
    }
    if (i >= 0 && i < nx && j + 1 <= ny && k + 1 < nz)
    {
        atomicAdd(&v[V_IDX(i, j + 1, k + 1)], w011 * vy);
        atomicAdd(&vW[V_IDX(i, j + 1, k + 1)], w011);
    }
    if (i + 1 < nx && j + 1 <= ny && k + 1 < nz)
    {
        atomicAdd(&v[V_IDX(i + 1, j + 1, k + 1)], w111 * vy);
        atomicAdd(&vW[V_IDX(i + 1, j + 1, k + 1)], w111);
    }
#undef V_IDX
}

__global__ void p2gKernel_W(
    const float3 *pos, const float3 *vel,
    float *w, float *wW,
    int nx, int ny, int nz, float invDx, int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    float3 p = pos[idx];
    float vz = vel[idx].z;

    float gx = p.x * invDx;
    float gy = p.y * invDx;
    float gz = p.z * invDx;

    float wx = gx - 0.5f, wy = gy - 0.5f, wz_s = gz;
    int i = (int)wx, j = (int)wy, k = (int)wz_s;
    float fx = wx - i, fy = wy - j, fz = wz_s - k;

    float m_fx = 1.0f - fx, m_fy = 1.0f - fy, m_fz = 1.0f - fz;
    float w000 = m_fx * m_fy * m_fz, w100 = fx * m_fy * m_fz;
    float w010 = m_fx * fy * m_fz, w110 = fx * fy * m_fz;
    float w001 = m_fx * m_fy * fz, w101 = fx * m_fy * fz;
    float w011 = m_fx * fy * fz, w111 = fx * fy * fz;

// Stride griglia w: nx * ny * (nz+1)
#define W_IDX(i, j, k) ((i) + nx * ((j) + ny * (k)))

    if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&w[W_IDX(i, j, k)], w000 * vz);
        atomicAdd(&wW[W_IDX(i, j, k)], w000);
    }
    if (i + 1 < nx && j >= 0 && j < ny && k >= 0 && k < nz)
    {
        atomicAdd(&w[W_IDX(i + 1, j, k)], w100 * vz);
        atomicAdd(&wW[W_IDX(i + 1, j, k)], w100);
    }
    if (i >= 0 && i < nx && j + 1 < ny && k >= 0 && k < nz)
    {
        atomicAdd(&w[W_IDX(i, j + 1, k)], w010 * vz);
        atomicAdd(&wW[W_IDX(i, j + 1, k)], w010);
    }
    if (i + 1 < nx && j + 1 < ny && k >= 0 && k < nz)
    {
        atomicAdd(&w[W_IDX(i + 1, j + 1, k)], w110 * vz);
        atomicAdd(&wW[W_IDX(i + 1, j + 1, k)], w110);
    }
    if (i >= 0 && i < nx && j >= 0 && j < ny && k + 1 <= nz)
    {
        atomicAdd(&w[W_IDX(i, j, k + 1)], w001 * vz);
        atomicAdd(&wW[W_IDX(i, j, k + 1)], w001);
    }
    if (i + 1 < nx && j >= 0 && j < ny && k + 1 <= nz)
    {
        atomicAdd(&w[W_IDX(i + 1, j, k + 1)], w101 * vz);
        atomicAdd(&wW[W_IDX(i + 1, j, k + 1)], w101);
    }
    if (i >= 0 && i < nx && j + 1 < ny && k + 1 <= nz)
    {
        atomicAdd(&w[W_IDX(i, j + 1, k + 1)], w011 * vz);
        atomicAdd(&wW[W_IDX(i, j + 1, k + 1)], w011);
    }
    if (i + 1 < nx && j + 1 < ny && k + 1 <= nz)
    {
        atomicAdd(&w[W_IDX(i + 1, j + 1, k + 1)], w111 * vz);
        atomicAdd(&wW[W_IDX(i + 1, j + 1, k + 1)], w111);
    }
#undef W_IDX
}



__global__ void dilateFluidFlags(int *cellType, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    int cidx = i + nx * (j + ny * k);
    if (cellType[cidx] == (int)cellType::FLUID)
    {
        if (i > 0)
            cellType[cidx - 1] = (int)cellType::FLUID;
        if (i < nx - 1)
            cellType[cidx + 1] = (int)cellType::FLUID;
        if (j > 0)
            cellType[cidx - nx] = (int)cellType::FLUID;
        if (j < ny - 1)
            cellType[cidx + nx] = (int)cellType::FLUID;
        if (k > 0)
            cellType[cidx - nx * ny] = (int)cellType::FLUID;
        if (k < nz - 1)
            cellType[cidx + nx * ny] = (int)cellType::FLUID;
    }
}
void cudaParticleSimulator::p2g()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;
    float invDx = 1.0f / grid_size.cellSize;
    int s1 = (nx + 1) * ny * nz;
    int s2 = nx * (ny + 1) * nz;
    int s3 = nx * ny * (nz + 1);

    CHECK(cudaMemsetAsync(grid_data.u, 0, s1 * sizeof(float), this->stream));
    CHECK(cudaMemsetAsync(grid_data.v, 0, s2 * sizeof(float), this->stream));
    CHECK(cudaMemsetAsync(grid_data.w, 0, s3 * sizeof(float), this->stream));

    CHECK(cudaMemsetAsync(grid_data.uWeight, 0, s1 * sizeof(float), this->stream));
    CHECK(cudaMemsetAsync(grid_data.vWeight, 0, s2 * sizeof(float), this->stream));
    CHECK(cudaMemsetAsync(grid_data.wWeight, 0, s3 * sizeof(float), this->stream));

    computeBoundary();

    //1 thread per particle
    int blockSize = 256;
    int gridDim = (numParticles + blockSize - 1) / blockSize;

    cudaStreamSynchronize(this->stream);
    p2gKernel_U<<<gridDim, blockSize, 0, this->uStream>>>(
        deviceData.pos, deviceData.vel,
        grid_data.u, grid_data.uWeight,
        nx, ny, nz, invDx, numParticles);
    p2gKernel_V<<<gridDim, blockSize, 0, this->vStream>>>(
        deviceData.pos, deviceData.vel,
        grid_data.v, grid_data.vWeight,
        nx, ny, nz, invDx, numParticles);
    p2gKernel_W<<<gridDim, blockSize, 0, this->wStream>>>(
        deviceData.pos, deviceData.vel,
        grid_data.w, grid_data.wWeight,
        nx, ny, nz, invDx, numParticles);
    int maxSize = s1;
    if (s2 > maxSize)
        maxSize = s2;
    if (s3 > maxSize)
        maxSize = s3;
    int gridNorm = (maxSize + blockSize - 1) / blockSize;

    cudaStreamSynchronize(this->uStream);
    cudaStreamSynchronize(this->vStream);
    cudaStreamSynchronize(this->wStream);
    p2gNormalizeKernel<<<gridNorm, blockSize, 0, this->stream>>>(
        grid_data.u, grid_data.v, grid_data.w,
        grid_data.uWeight, grid_data.vWeight, grid_data.wWeight,
        nx, ny, nz);
    //if (!narrowBandMarked)
        dilateFluidFlags<<<dim3((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8), dim3(8, 8, 8), 0, this->stream>>>(grid_data.cellType, nx, ny, nz);
    cudaStreamSynchronize(this->stream);
}

void cudaParticleSimulator::saveGridVelocities()
{
    int nx = grid_size.x, ny = grid_size.y, nz = grid_size.z;

    cudaMemcpyAsync(grid_data.u_old, grid_data.u, (nx + 1) * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice, this->stream);
    cudaMemcpyAsync(grid_data.v_old, grid_data.v, nx * (ny + 1) * nz * sizeof(float), cudaMemcpyDeviceToDevice, this->stream);
    cudaMemcpyAsync(grid_data.w_old, grid_data.w, nx * ny * (nz + 1) * sizeof(float), cudaMemcpyDeviceToDevice, this->stream);
}