#pragma once

#include <vector_types.h>

#if defined(__CUDACC__)

struct Sym3 {
    float xx, yy, zz, xy, xz, yz;
};

__device__ inline void sym3_eig(const Sym3& A, float eval[3], float3 evec[3])
{
   
    float a[3][3] = {
        { A.xx, A.xy, A.xz },
        { A.xy, A.yy, A.yz },
        { A.xz, A.yz, A.zz }
    };
    
    float v[3][3] = {
        { 1.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f },
        { 0.f, 0.f, 1.f }
    };
    const int    SWEEPS = 8;
    const int    pp[3] = { 0, 0, 1 };
    const int    qq[3] = { 1, 2, 2 };
    const float  EPS = 1e-20f;

    for (int sweep = 0; sweep < SWEEPS; ++sweep) {
        for (int k = 0; k < 3; ++k) {
            int p = pp[k], q = qq[k];
            float apq = a[p][q];
            if (apq * apq < EPS) continue;        

            float app = a[p][p], aqq = a[q][q];
            float theta = (aqq - app) / (2.f * apq);
            float t = (theta >= 0.f ? 1.f : -1.f) /
                      (fabsf(theta) + sqrtf(theta * theta + 1.f));
            float c = 1.f / sqrtf(t * t + 1.f);
            float s = t * c;
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = a[q][p] = 0.f;
            int r = 3 - p - q;      
            float arp = a[r][p], arq = a[r][q];
            a[r][p] = a[p][r] = c * arp - s * arq;
            a[r][q] = a[q][r] = s * arp + c * arq;
            for (int i = 0; i < 3; ++i) {
                float vip = v[i][p], viq = v[i][q];
                v[i][p] = c * vip - s * viq;
                v[i][q] = s * vip + c * viq;
            }
        }
    }

    eval[0] = a[0][0]; eval[1] = a[1][1]; eval[2] = a[2][2];
    evec[0] = make_float3(v[0][0], v[1][0], v[2][0]);
    evec[1] = make_float3(v[0][1], v[1][1], v[2][1]);
    evec[2] = make_float3(v[0][2], v[1][2], v[2][2]);
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2 - i; ++j)
            if (eval[j] < eval[j + 1]) {
                float te = eval[j]; eval[j] = eval[j + 1]; eval[j + 1] = te;
                float3 tv = evec[j]; evec[j] = evec[j + 1]; evec[j + 1] = tv;
            }
}

#endif
