

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


struct Mat33d
{
	double3 data[3];
	Mat33d() {}
	Mat33d(double3 f1, double3 f2, double3 f3)
	{
		data[0] = f1;
		data[1] = f2;
		data[2] = f3;
	}
	Mat33d(double*data_)
	{
		double3 a1;		a1.x = data_[0]; a1.y = data_[1]; a1.z = data_[2];
		double3 a2;		a1.x = data_[3]; a1.y = data_[4]; a1.z = data_[5];
		double3 a3;		a1.x = data_[6]; a1.y = data_[7]; a1.z = data_[8];
		data[0] = a1;
		data[1] = a2;
		data[2] = a3;
	}

	Mat33d(double*d1, double*d2, double*d3)
	{
		double3 a1;		a1.x = d1[0]; a1.y = d1[1]; a1.z = d1[2];
		double3 a2;		a2.x = d2[0]; a2.y = d2[1]; a2.z = d2[2];
		double3 a3;		a3.x = d3[0]; a3.y = d3[1]; a3.z = d3[2];
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
operator-(const float& v1, const float3& v2)
{
	return make_float3(v1 - v2.x, v1 - v2.y, v1 - v2.z);
}
__device__ __forceinline__ float3
operator*(const float3& v1, const float& v)
{
	return make_float3(v1.x * v, v1.y * v, v1.z * v);
}
__device__ __forceinline__ float3
operator/(const float3& v1, const float& v)
{
	return make_float3(v1.x / v, v1.y / v, v1.z / v);
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

__device__ __forceinline__ double3
operator+(const double3& v1, const double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
__device__ __forceinline__ double3
operator-(const double3& v1, const double3& v2)
{
	return make_double3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
__device__ __forceinline__ double3
operator-(const double& v1, const double3& v2)
{
	return make_double3(v1 - v2.x, v1 - v2.y, v1 - v2.z);
}
__device__ __forceinline__ double3
operator*(const double3& v1, const double& v)
{
	return make_double3(v1.x * v, v1.y * v, v1.z * v);
}
__device__ __forceinline__ double3
operator/(const double3& v1, const double& v)
{
	return make_double3(v1.x / v, v1.y / v, v1.z / v);
}
__device__ __forceinline__ double
dot(const double3& v1, const double3& v2)
{
	return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
}
__device__ __forceinline__ double3
operator* (const Mat33d& m, const double3& vec)
{
	return make_double3(dot(m.data[0], vec), dot(m.data[1], vec), dot(m.data[2], vec));
}


template<typename T> struct numeric_limits;

#define VOLUME_SIZE_X (1.024)
#define VOLUME_SIZE_Y (1.024)
#define VOLUME_SIZE_Z (1.024)


enum 
{
	VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512,
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

int initOneDevDeep(short*&depth_input, float*&depth_output, short*&depth_output_bila,
	float*&depth_dev_med, float*&depth_filled, 
	float2*&depth_2, float2*&depth_3,
	float*&vmap, float*&nmap, float*&nmap_average,
	unsigned char*&rgbData, unsigned char*&newRgbData,
	int depthRows, int depthCols, int colorRows, int colorCols);

#define AVERAGE_DEEP_5 0
#define AVERAGE_DEEP_5_UPDATA 1
#if AVERAGE_DEEP_3 && AVERAGE_DEEP_3_UPDATA || AVERAGE_DEEP_5 && AVERAGE_DEEP_3_UPDATA || AVERAGE_DEEP_3 && AVERAGE_DEEP_5_UPDATA || AVERAGE_DEEP_5 && AVERAGE_DEEP_5_UPDATA
#error "the models above cant be assigned at same time"
#endif // AVERAGE_DEEP_5


#ifdef AVERAGE_DEEP_3 1
int initAverageDeep(short*&deep_average0, short*&deep_average1, short*&deep_average2,
	int rows, int cols);
#elif AVERAGE_DEEP_5
int initAverageDeep(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average3, short*&deep_average4,
	int rows, int cols);
#endif // AVERAGE_DEEP_3

#ifdef AVERAGE_DEEP_3_UPDATA
int initAverageDeep(short*&deep_average0, short*&deep_average1, short*&deep_average2,
	int rows, int cols);
#elif AVERAGE_DEEP_5_UPDATA
int initAverageDeep(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average3, short*&deep_average4,
	int rows, int cols);
#endif // AVERAGE_DEEP_3_UPDATA




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
int createVMap(const float*dataIn, float*dataOut, const T fx, const T fy, const T cx, const T cy, const int rows, const int cols);


template<typename T>
int computeNormalsEigen(const T*vmap, T*nmap, T*nmap_average, int rows, int cols);

template<typename T>
int tranformMaps(const T* vmap_src, const T* nmap_src, const T*Rmat_, const T*tvec_, T* vmap_dst, T* nmap_dst, const int& rows, const int& cols);
template<typename T>
int tranformMaps(const T* vmap_src, const T*Rmat_, const T*tvec_, T* vmap_dst, const int& rows, const int& cols);


void integrateTsdfVolume(const short* depth_raw, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R, float3 t, float3 cameraPos, float tranc_dist, short2* volume, float*&depthRawScaled);

////用光线追踪的想法来扫描体素得到点云，此处的点云的角度并不是large-slace，所以t_必须在体素外！！
void
raycastPoint(const short2* volume, float3* vmap, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R_inv, float3 t_, float3 cameraPos_, float tranc_dist);

template<typename T>
int bilateralFilter(const T*dataIn, T*dataOut, const int rows, const int cols);


void medfilter33_forOneDev(
	float*depth_dev1, int rows1, int cols1,
	float*depth_dev1_midfiltered, float*depth_dev1_filled,
	float2*depth_dev2, int rows2, int cols2,
	float2*depth_dev3, int rows3, int cols3);
void colorize_deepMat(
	const short* depth_old, 
	int depthRows, int depthCols, int colorRows, int colorCols,
	double4 deep_intr,
	Mat33d deep_R, double3 deep_t,
	double4 color_intr,
	Mat33d color_R, double3 color_t,
	float* depth_new
);

#ifdef AVERAGE_DEEP_3
template<typename Dtype>
void combineavgrageDeep(const Dtype*avg0, const Dtype*avg1, const Dtype*avg2, Dtype*out, const int rows, const int cols);
#elif avgRAGE_DEEP_5
template<typename Dtype>
void combineavgrageDeep(const Dtype*avg0, const Dtype*avg1, const Dtype*avg2, const Dtype*avg3, const Dtype*avg4,
	Dtype*out, const int rows, const int cols);
#endif // AVERAGE_DEEP_3

#ifdef AVERAGE_DEEP_3_UPDATA
template<typename Dtype> void
combineAverageDeep(const Dtype*avg0, const Dtype*avg1, const Dtype*avg2, 
	Dtype*out, const int rows, const int cols);
#elif AVERAGE_DEEP_5_UPDATA
template<typename Dtype> void
combineAverageDeep(const Dtype*avg0, const Dtype*avg1, const Dtype*avg2, const Dtype*avg3, const Dtype*avg4,
	Dtype*out, const int rows, const int cols);
#endif // AVERAGE_DEEP_3_UPDATA

void combineNmap2Rgb(
	unsigned char*rgb, float*nmap,
	unsigned char*rgbOut,
	int colorRows, int colorCols);

#endif