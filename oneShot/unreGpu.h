

#ifndef UNRE_GPU_H_
#define UNRE_GPU_H_

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
//#include "math_functions.h"
#include "device_functions.h"



static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
{
	if (cudaSuccess != err) printf("%d : Error: %s - file:%s, line:%d, func:%s\n", err,cudaGetErrorString(err), file, line, func);
}

#if defined(__GNUC__)
#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaSafeCall(expr)  ___cudaSafeCall(expr, __FILE__, __LINE__)    
#endif


//#define DOWNSAMPLE3TIMES
#define DOWNSAMPLE2TIMES

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

__device__ __forceinline__ float3
operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
__device__ __forceinline__ float3
operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
__device__ __forceinline__ float3
operator*(const float3& v1, const float& v)
{
	return make_float3(v1.x * v, v1.y * v, v1.z * v);
}
__device__ __forceinline__ float
dot(const float3& v1, const float3& v2)
{
	return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}
__device__ __forceinline__ float3
operator* (const Mat33& m, const float3& vec)
{
	return make_float3(dot(m.data[0], vec), dot(m.data[1], vec), dot(m.data[2], vec));
}

template<typename T> struct numeric_limits;

#define VOLUME_SIZE_X (1.024)
#define VOLUME_SIZE_Y (1.024)
#define VOLUME_SIZE_Z (1.024)


enum 
{
	VOLUME_X = 1024, VOLUME_Y = 1024, VOLUME_Z = 1024,
};
const int DIVISOR = 32767;     // SHRT_MAX;
template<typename Dtype>
Dtype* creatGpuData(const int elemCnt, bool fore_zeros = false);

static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }

////目前只是init一个深度相机的空间
int initVolu(short*&depth_dev, float*&scaledDepth, float3*&dev_vmap,
	short*&depth_midfiltered, short*&depth_filled,
	short2*&depth_2, short2*&depth_3,
	int depthRows, int depthCols);

#ifdef DOWNSAMPLE3TIMES
void midfilter33AndFillHoles44_downsample3t(short*depth_dev1, int rows1, int cols1,
	short*depth_dev1_midfiltered, short*depth_dev1_filled,
	short2*depth_dev2, int rows2, int cols2,
	short2*depth_dev3, int rows3, int cols3,
	short2*depth_dev4, int rows4, int cols4);
#else
void midfilter33AndFillHoles44_downsample2t(short*depth_dev1, int rows1, int cols1,
	short*depth_dev1_midfiltered, short*depth_dev1_filled,
	short2*depth_dev2, int rows2, int cols2,
	short2*depth_dev3, int rows3, int cols3);
#endif // DOWNSAMPLE3TIMES




template<typename T>
int computeNormalsEigen(const T*vmap, T*nmap, const int rows, const int cols);

template<typename T>
int tranformMaps(const T* vmap_src, const T* nmap_src, const T*Rmat_, const T*tvec_, T* vmap_dst, T* nmap_dst, const int& rows, const int& cols);
template<typename T>
int tranformMaps(const T* vmap_src, const T*Rmat_, const T*tvec_, T* vmap_dst, const int& rows, const int& cols);


void integrateTsdfVolume(const short* depth_raw, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R, float3 t, float tranc_dist, short2* volume, float*&depthRawScaled);

////用光线追踪的想法来扫描体素得到点云，此处的点云的角度并不是large-slace，所以t_必须在体素外！！
void
raycastPoint(const short2* volume, float3* vmap, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R_inv, float3 t_, float tranc_dist);



#endif