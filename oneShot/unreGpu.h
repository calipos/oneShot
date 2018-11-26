

#ifndef UNRE_GPU_H_
#define UNRE_GPU_H_

#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime_api.h"

static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
	if (cudaSuccess != err) printf("Error: %s - file:%s, line:%d, func:%s\n", cudaGetErrorString(err), file, line, func);
}

#if defined(__GNUC__)
#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)    
#endif

enum { VOLUME_SIZE_X = 512, VOLUME_SIZE_Y = 512, VOLUME_SIZE_Z = 512 };
enum { VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512 };

template<typename Dtype>
Dtype* creatGpuData(const int elemCnt, bool fore_zeros = false);

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }


void integrateTsdfVolume(const unsigned short* depth_raw, int rows, int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy,
	const double* R, const double* t, const float tranc_dist, short* volume, float* depthRawScaled);


#endif