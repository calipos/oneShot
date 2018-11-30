

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

struct Mat33
{
	float3 data[3];
	Mat33() {}
	Mat33(float3 f1, float3 f2, float3 f3)
	{
		data[0] = f1;
		data[1] = f2;
		data[2] = f3;
	}
	Mat33(float*data)
	{
		float3 a1;		a1.x = data[0]; a1.y = data[1]; a1.z = data[2];
		float3 a2;		a1.x = data[3]; a1.y = data[4]; a1.z = data[5];
		float3 a3;		a1.x = data[6]; a1.y = data[7]; a1.z = data[8];
		Mat33(a1,a2,a3);
	}
	Mat33(double*data_)
	{
		float3 a1;		a1.x = data_[0]; a1.y = data_[1]; a1.z = data_[2];
		float3 a2;		a1.x = data_[3]; a1.y = data_[4]; a1.z = data_[5];
		float3 a3;		a1.x = data_[6]; a1.y = data_[7]; a1.z = data_[8];
		data[0] = a1;
		data[1] = a2;
		data[2] = a3;
	}
	Mat33(float*d1, float*d2, float*d3)
	{
		float3 a1;		a1.x = d1[0]; a1.y = d1[1]; a1.z = d1[2];
		float3 a2;		a2.x = d2[0]; a2.y = d2[1]; a2.z = d2[2];
		float3 a3;		a3.x = d3[0]; a3.y = d3[1]; a3.z = d3[2];
		data[0] = a1;
		data[1] = a2;
		data[2] = a3;
	}
	Mat33(double*d1, double*d2, double*d3)
	{
		float3 a1;		a1.x = d1[0]; a1.y = d1[1]; a1.z = d1[2];
		float3 a2;		a2.x = d2[0]; a2.y = d2[1]; a2.z = d2[2];
		float3 a3;		a3.x = d3[0]; a3.y = d3[1]; a3.z = d3[2];
		data[0] = a1;
		data[1] = a2;
		data[2] = a3;
	}
};



enum { VOLUME_SIZE_X = 1024, VOLUME_SIZE_Y = 1024, VOLUME_SIZE_Z = 1024};//mm
enum { VOLUME_X = 1024, VOLUME_Y = 1024, VOLUME_Z = 1024
};

template<typename Dtype>
Dtype* creatGpuData(const int elemCnt, bool fore_zeros = false);

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

int initVolu();

void integrateTsdfVolume(const short* depth_raw, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R, float3 t, float tranc_dist, short2* volume, float*&depthRawScaled);


#endif