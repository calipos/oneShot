#include "unityStruct.h"
#include "unreGpu.h"
float *points = NULL;
float *norms = NULL;
float *rgba = NULL;
static int pointsEleByteCnt = 0;
static int rgbaEleByteCnt = 0;
template<> struct numeric_limits<float>
{
	__device__ __forceinline__ static bool
		isnan(float value) { return __int_as_float(0x7fffffff) == value; };
};

template<> struct numeric_limits<short>
{
	__device__ __forceinline__ static short
		max() { return SHRT_MAX; };
};
int initUnityData(const int h, const int w)
{
	points = creatGpuData<float>(3 * w*h);
	norms = creatGpuData<float>(3 * w*h);
	rgba = creatGpuData<float>(4 * w*h);
	pointsEleByteCnt = 3 * w*h*sizeof(float);
	rgbaEleByteCnt = 4 * w*h * sizeof(float);
	return 0;
}

template <typename Dtype1, typename Dtype2>
__global__ void sampleKernel(const Dtype1*vmap, 
							const Dtype1*nmap, 
							const Dtype2*rgb,
							Dtype1 *points_, 
							Dtype1 *norms_, 
							Dtype1 *rgba_, 
							double4 intr_inv,
							const int sample_h, const int sample_w,const int ori_w)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= sample_w || v >= sample_h)
		return;
	int sample_index = 3 * (v*sample_w + u);
	int sample_rgb_index = 4 * (v*sample_w + u);
	int ori_index = 3 * (3*v*ori_w + 3*u);
	if (isnan(vmap[ori_index])|| (nmap[ori_index] == 0 && nmap[ori_index+1] == 0 && nmap[ori_index+2] == 0))
	{
		points_[sample_index] = 10*(3 * u- intr_inv.y)*intr_inv.w;
		points_[sample_index + 1] = 10*(3 * v - intr_inv.z) * intr_inv.x;
		points_[sample_index + 2] = 10.;
		norms_[sample_index] = 0.;
		norms_[sample_index + 1] = 0.;
		norms_[sample_index + 2] = 0.;
		rgba_[sample_rgb_index] = 0.;//////////////////////
		rgba_[sample_rgb_index + 1] = 0.;//////////////////////
		rgba_[sample_rgb_index + 2] = 0.;//////////////////////
		rgba_[sample_rgb_index + 3] = 0.;/////////
	}
	else
	{
		points_[sample_index] = vmap[ori_index];
		points_[sample_index + 1] = vmap[ori_index + 1];
		points_[sample_index + 2] = vmap[ori_index + 2];
		norms_[sample_index] = nmap[ori_index];
		norms_[sample_index + 1] = nmap[ori_index + 1];
		norms_[sample_index + 2] = nmap[ori_index + 2];
		rgba_[sample_rgb_index] = rgb[ori_index] * 0.0039;//////////////////////
		rgba_[sample_rgb_index +1] = rgb[ori_index +1] * 0.0039;//////////////////////
		rgba_[sample_rgb_index +2] = rgb[ori_index +2] * 0.0039;//////////////////////
		rgba_[sample_rgb_index +3] = 1.;/////////
	}
	
}

int sampleUnityData(const float*vmap, 
	const float*nmap, 
	const unsigned char*rgb,
	double4 intr,
	const int h, const int w,const int ori_w)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(w, block.x);
	grid.y = divUp(h, block.y);
	double4 intr_inv = intr;
	intr_inv.w = 1 / intr_inv.w;
	intr_inv.x = 1 / intr_inv.x;
	sampleKernel<float, unsigned char> << <grid, block >> >(vmap,
		nmap,
		rgb,
		points,
		norms,
		rgba,
		intr_inv,
		h, w,ori_w);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}

int device2Host(float*host_points, float*host_norms, float*host_rgba)
{
	cudaMemcpy(host_points, points, pointsEleByteCnt, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_norms, norms, pointsEleByteCnt, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_rgba, rgba, rgbaEleByteCnt, cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}