
#include "unreGpu.h"
#define DEPTHMAT_TRUC (1.5)

int initOneDevDeep(
	short*&depth_input, float*&depth_output, short*&depth_output_bila,
	float*&depth_dev_med, 
	float*&depth_filled, float2*&depth_2, float2*&depth_3,
	float*&vmap, float*&nmap, float*&nmap_filled,
	unsigned char*&rgbData, unsigned char*&newRgbData,
	int depthRows, int depthCols, int colorRows, int colorCols)
{
#if AVERAGE_DEEP_3_UPDATA || AVERAGE_DEEP_5_UPDATA
	depth_input = creatGpuData<short>(depthRows*depthCols,true);//在更新的策略中，depth_input也作为base，所以初始化会为0，每一次刷数据的时候只会刷到avg...中
#else
	depth_input = creatGpuData<short>(depthRows*depthCols);
#endif
	depth_output = creatGpuData<float>(colorRows*colorCols);
	depth_output_bila = creatGpuData<short>(colorRows*colorCols);
	depth_dev_med = creatGpuData<float>(colorRows*colorCols);
	depth_filled = creatGpuData<float>(colorRows*colorCols);


	int downsample_h2 = colorRows / 4;
	int downsample_w2 = colorCols / 4;
	depth_2 = creatGpuData<float2>(downsample_h2*downsample_w2);
	int downsample_h3 = colorRows / 16;
	int downsample_w3 = colorCols / 16;
	depth_3 = creatGpuData<float2>(downsample_h3*downsample_w3);
	
	vmap = creatGpuData<float>(colorRows*colorCols * 3);
	nmap = creatGpuData<float>(colorRows*colorCols * 3);
	nmap_filled = creatGpuData<float>(colorRows*colorCols * 3);

	rgbData = creatGpuData<unsigned char>(colorRows*colorCols * 3);
	newRgbData = creatGpuData<unsigned char>(colorRows*colorCols * 3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}


#if AVERAGE_DEEP_3
template<>
int initAverageDeep<float>(short*&deep_average0, short*&deep_average1, short*&deep_average2, float*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<float>(rows*cols, true);
	return 0;
}
template<>
int initAverageDeep<short>(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<short>(rows*cols, true);
	return 0;
}
#elif AVERAGE_DEEP_5
template<>
int initAverageDeep<float>(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average3, 
	short*&deep_average4, float*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average3 = creatGpuData<short>(rows*cols, true);
	deep_average4 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<float>(rows*cols, true);
	return 0;
}
template<>
int initAverageDeep<short>(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average3,
	short*&deep_average4, short*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average3 = creatGpuData<short>(rows*cols, true);
	deep_average4 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<short>(rows*cols, true);
	return 0;
}
#endif // AVERAGE_DEEP_3


#if AVERAGE_DEEP_3_UPDATA
template<>
int initAverageDeep<float>(short*&deep_average0, short*&deep_average1, short*&deep_average2,
	float*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<float>(rows*cols, true);
	return 0;
}
#elif AVERAGE_DEEP_5_UPDATA
int initAverageDeep(short*&deep_average0, short*&deep_average1, short*&deep_average2, short*&deep_average3, 
	short*&deep_average4, float*&deep_average_out,
	int rows, int cols)
{
	deep_average0 = creatGpuData<short>(rows*cols, true);
	deep_average1 = creatGpuData<short>(rows*cols, true);
	deep_average2 = creatGpuData<short>(rows*cols, true);
	deep_average3 = creatGpuData<short>(rows*cols, true);
	deep_average4 = creatGpuData<short>(rows*cols, true);
	deep_average_out = creatGpuData<float>(rows*cols, true);
	return 0;
}
#elif AVERAGE_DEEP_15_UPDATA
int initAverageDeep(short*&deep_average15, float*&deep_average_out, int rows, int cols)
{
	deep_average15 = creatGpuData<short>(rows*cols*15, true);
	deep_average_out = creatGpuData<float>(rows*cols, true);
	return 0;
}
#endif // AVERAGE_DEEP_3_UPDATA



//__device__ __forceinline__
//void invert3x3(const float * src, float * dst)
//{
//	float det;
//	dst[0] = +src[4] * src[8] - src[5] * src[7];
//	dst[1] = -src[1] * src[8] + src[2] * src[7];
//	dst[2] = +src[1] * src[5] - src[2] * src[4];
//	dst[3] = -src[3] * src[8] + src[5] * src[6];
//	dst[4] = +src[0] * src[8] - src[2] * src[6];
//	dst[5] = -src[0] * src[5] + src[2] * src[3];
//	dst[6] = +src[3] * src[7] - src[4] * src[6];
//	dst[7] = -src[0] * src[7] + src[1] * src[6];
//	dst[8] = +src[0] * src[4] - src[1] * src[3];
//	det = src[0] * dst[0] + src[1] * dst[3] + src[2] * dst[6];
//	det = 1.0f / det;
//	dst[0] *= det;
//	dst[1] *= det;
//	dst[2] *= det;
//	dst[3] *= det;
//	dst[4] *= det;
//	dst[5] *= det;
//	dst[6] *= det;
//	dst[7] *= det;
//	dst[8] *= det;
//}

__device__ __forceinline__
void invert3x3(const double3 * src, double3 * dst)
{
	float det;
	dst[0].x = +src[1].y * src[2].z - src[1].z * src[2].y;
	dst[0].y = -src[0].y * src[2].z + src[0].z * src[2].y;
	dst[0].z = +src[0].y * src[1].z - src[0].z * src[1].y;
	dst[1].x = -src[1].x * src[2].z + src[1].z * src[2].x;
	dst[1].y = +src[0].x * src[2].z - src[0].z * src[2].x;
	dst[1].z = -src[0].x * src[1].z + src[0].z * src[1].x;
	dst[2].x = +src[1].x * src[2].y - src[1].y * src[2].x;
	dst[2].y = -src[0].x * src[2].y + src[0].y * src[2].x;
	dst[2].z = +src[0].x * src[1].y - src[0].y * src[1].x;
	det = src[0].x * dst[0].x + src[0].y * dst[1].x + src[0].z * dst[2].x;
	det = 1.0f / det;
	dst[0].x *= det;
	dst[0].y *= det;
	dst[0].z *= det;
	dst[1].x *= det;
	dst[1].y *= det;
	dst[1].z *= det;
	dst[2].x *= det;
	dst[2].y *= det;
	dst[2].z *= det;
}



template<typename Dtype> __global__ void
colorize_deepMat_kernel(const Dtype* depth_old,
	int depthRows, int depthCols, int colorRows, int colorCols,
	double4 deep_intr,
	Mat33d deep_R, double3 deep_t,
	double4 color_intr,
	Mat33d color_R, double3 color_t,
	float* depth_new)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= depthCols || y >= depthRows)
		return;

	double cmos_x = (x - deep_intr.y) / deep_intr.w;
	double cmos_y = (y - deep_intr.z) / deep_intr.x;
	double cmos_z = 1.;
	double z = depth_old[y*depthCols + x] * 0.001;

	if (z<0.001 || z>DEPTHMAT_TRUC)
	{
		return;
	}

	Mat33d deep_R_neg = deep_R;
	deep_R_neg.data[0].z = 0. - cmos_x;
	deep_R_neg.data[1].z = 0. - cmos_y;
	deep_R_neg.data[2].z = -1.;


	Mat33d deep_R_neg_inv = deep_R;
	invert3x3(deep_R_neg.data, deep_R_neg_inv.data);

	double3 xyw = (0.f - deep_t);
	xyw.x = xyw.x - deep_R.data[0].z*z;
	xyw.y = xyw.y - deep_R.data[1].z*z;
	xyw.z = xyw.z - deep_R.data[2].z*z;

	double3 xyz = deep_R_neg_inv*xyw;
	xyz.z = z;
	double3 pointInColor = color_R*xyz + color_t;
	pointInColor = pointInColor / pointInColor.z;
	int this_x = 0.5 + pointInColor.x * color_intr.w + color_intr.y;
	int this_y = 0.5 + pointInColor.y * color_intr.x + color_intr.z;

	if (this_x<0 || this_y<0 || this_x >= colorCols || this_y >= colorRows)
	{
		return;
	}
	depth_new[this_y*colorCols + this_x] = z;
}

template<>
void colorize_deepMat<float>(
	const float* depth_old,
	int depthRows, int depthCols, int colorRows, int colorCols,
	double4 deep_intr,
	Mat33d deep_R, double3 deep_t,
	double4 color_intr,
	Mat33d color_R, double3 color_t,
	float* depth_new
)
{
	dim3 block_scale(32, 8);
	dim3 grid_scale(divUp(depthCols, block_scale.x), divUp(depthRows, block_scale.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	cudaMemset(depth_new, 0, sizeof(float)*colorRows * colorCols);
	//scales depth along ray and converts mm -> meters. 
	colorize_deepMat_kernel<float> << <grid_scale, block_scale >> >(depth_old,
		depthRows, depthCols, colorRows, colorCols,
		deep_intr,	deep_R, deep_t,
		color_intr,	color_R, color_t,
		depth_new);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}



template<>
void colorize_deepMat<short>(
	const short* depth_old,
	int depthRows, int depthCols, int colorRows, int colorCols,
	double4 deep_intr,
	Mat33d deep_R, double3 deep_t,
	double4 color_intr,
	Mat33d color_R, double3 color_t,
	float* depth_new
	)
{
	dim3 block_scale(32, 8);
	dim3 grid_scale(divUp(depthCols, block_scale.x), divUp(depthRows, block_scale.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	cudaMemset(depth_new, 0, sizeof(float)*colorRows * colorCols);
	//scales depth along ray and converts mm -> meters. 
	colorize_deepMat_kernel<short> << <grid_scale, block_scale >> >(depth_old,
		depthRows, depthCols, colorRows, colorCols,
		deep_intr, deep_R, deep_t,
		color_intr, color_R, color_t,
		depth_new);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}