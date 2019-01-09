#include "unreGpu.h"
#include "filters.h"



template<> struct numeric_limits<float>
{
	__device__ __forceinline__ static float
		quiet_NaN() { return __int_as_float(0x7fffffff); /*CUDART_NAN_F*/ };
	__device__ __forceinline__ static float
		epsilon() { return 1.192092896e-07f/*FLT_EPSILON*/; };

	__device__ __forceinline__ static float
		min() { return 1.175494351e-38f/*FLT_MIN*/; };
	__device__ __forceinline__ static float
		max() { return 3.402823466e+38f/*FLT_MAX*/; };

	__device__ __forceinline__ static bool
		isnan(float value) { return __int_as_float(0x7fffffff) == value; };
};

template<> struct numeric_limits<short>
{
	__device__ __forceinline__ static short
		max() { return SHRT_MAX; };
};


struct BilateralParam
{
	enum 
	{
		R = 8,
	};
};

template <typename Dtype>
__global__ void bilateralKernel(const Dtype* src, Dtype* dst, const int rows, const int cols,
	float sigma_space2_inv_half, float sigma_color2_inv_half)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols || y >= rows)
		return;

	const int R = BilateralParam::R;       //static_cast<int>(sigma_space * 1.5);
	const int D = R * 2 + 1;

	int pos_ = y*cols + x;
	Dtype value = src[pos_];

	int tx = min(x - D / 2 + D, cols - 1);
	int ty = min(y - D / 2 + D, rows - 1);

	float sum1 = 0;
	float sum2 = 0;

	for (int cy = max(y - D / 2, 0); cy < ty; ++cy)
	{
		for (int cx = max(x - D / 2, 0); cx < tx; ++cx)
		{
			int pos_loop = cy*cols + cx;
			Dtype tmp = src[pos_loop];

			float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
			float color2 = (value - tmp) * (value - tmp);

			float weight = __expf(-(space2 * sigma_space2_inv_half + color2 * sigma_color2_inv_half));

			sum1 += tmp * weight;
			sum2 += weight;
		}
	}
	int res = __float2int_rn(sum1 / sum2);
	dst[pos_] = max((float)0, min((float)res, numeric_limits<float>::max()));
}

template<>
int bilateralFilter<float>(const float*dataIn, float*dataOut, const int rows, const int cols)
{
	float sigma_color = 5;     //in mm
	float sigma_space = 40;     // in pixels
	dim3 block(32, 8);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	cudaFuncSetCacheConfig(bilateralKernel<float>, cudaFuncCachePreferL1);
	bilateralKernel<float> << <grid, block >> > (dataIn, dataOut, rows, cols, 0.5f / (sigma_space * sigma_space), 0.5f / (sigma_color * sigma_color));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}

template<>
int bilateralFilter<short>(const short*dataIn, short*dataOut, const int rows, const int cols)
{
	float sigma_color = 25;     //in mm
	float sigma_space = 35;     // in pixels
	dim3 block(32, 8);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));
	cudaFuncSetCacheConfig(bilateralKernel<float>, cudaFuncCachePreferL1);
	bilateralKernel<short> << <grid, block >> > (dataIn, dataOut, rows, cols, 0.5f / (sigma_space * sigma_space), 0.5f / (sigma_color * sigma_color));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}

//暂时没有去除飞点，考虑了一下不合适

template<typename Dtype>
__global__ void medianFilter3AndDiscardNoisyKernel(Dtype * input, Dtype * output, unsigned int DATA_W, unsigned int DATA_H)
{
	__shared__ float window[BLOCK_W_FILTER*BLOCK_H_FILTER][9];

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int tid = threadIdx.y*blockDim.y + threadIdx.x;

	if (x >= DATA_W && y >= DATA_H)
		return;
	if (input[y*DATA_W + x]>0.001)
	{
		output[y*DATA_W + x] = input[y*DATA_W + x];
		return;
	}

	window[tid][0] = (y == 0 || x == 0) ? 0.0f : input[(y - 1)*DATA_W + x - 1];
	window[tid][1] = (y == 0) ? 0.0f : input[(y - 1)*DATA_W + x];
	window[tid][2] = (y == 0 || x == DATA_W - 1) ? 0.0f : input[(y - 1)*DATA_W + x + 1];
	window[tid][3] = (x == 0) ? 0.0f : input[y*DATA_W + x - 1];
	window[tid][4] = input[y*DATA_W + x];
	window[tid][5] = (x == DATA_W - 1) ? 0.0f : input[y*DATA_W + x + 1];
	window[tid][6] = (y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 1)*DATA_W + x - 1];
	window[tid][7] = (y == DATA_H - 1) ? 0.0f : input[(y + 1)*DATA_W + x];
	window[tid][8] = (y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 1)*DATA_W + x + 1];
		
	int zerosNum = 0;
	for (int i = 0; i < 9; i++)
	{
		if (window[tid][i]<0.001)
		{
			zerosNum++;
		}
	}
	for (unsigned int j = 0; j<(9-zerosNum)/2+1; ++j)
	{
		int max = j;
		for (unsigned int l = j + 1; l<9; ++l)
			if (window[tid][l] > window[tid][max])
				max = l;
		float temp = window[tid][j];
		window[tid][j] = window[tid][max];
		window[tid][max] = temp;
	}
	if (((x < 1) && (y < 1)) || ((x > DATA_W - 1) && (y < 1)) || ((x < 1) && (y > DATA_H - 1)) || ((x > DATA_W - 1) && (y > DATA_H - 1)))
	{
		output[y*DATA_W + x] = input[y*DATA_W + x];
	}
	else
		output[y*DATA_W + x] = window[tid][(9 - zerosNum) / 2];
}

template<typename Dtype>
__global__ void medianFilter5AndDiscardNoisyKernel(Dtype * input, Dtype * output, unsigned int DATA_W, unsigned int DATA_H)
{
	float window[25];

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= DATA_W && y >= DATA_H)
		return;

	window[0] = (y == 0 || y == 1 || x == 0 || x == 1) ? 0.0f : input[(y - 2)*DATA_W + x - 2];
	window[1] = (y == 0 || y == 1 || x == 0) ? 0.0f : input[(y - 2)*DATA_W + x - 1];
	window[2] = (y == 0 || y == 1) ? 0.0f : input[(y - 2)*DATA_W + x];
	window[3] = (y == 0 || y == 1 || x == DATA_W - 1) ? 0.0f : input[(y - 2)*DATA_W + x + 1];
	window[4] = (y == 0 || y == 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y - 2)*DATA_W + x + 2];

	window[5] = (y == 0 || x == 0 || x == 1) ? 0.0f : input[(y - 1)*DATA_W + x - 2];
	window[6] = (y == 0 || x == 0) ? 0.0f : input[(y - 1)*DATA_W + x - 1];
	window[7] = (y == 0) ? 0.0f : input[(y - 1)*DATA_W + x];
	window[8] = (y == 0 || x == DATA_W - 1) ? 0.0f : input[(y - 1)*DATA_W + x + 1];
	window[9] = (y == 0 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y - 1)*DATA_W + x + 2];

	window[10] = (x == 0 || x == 1) ? 0.0f : input[y*DATA_W + x - 2];
	window[11] = (x == 0) ? 0.0f : input[y*DATA_W + x - 1];
	window[12] = input[y*DATA_W + x];
	window[13] = (x == DATA_W - 1) ? 0.0f : input[y*DATA_W + x + 1];
	window[14] = (x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[y*DATA_W + x + 2];

	window[15] = (y == DATA_H - 1 || x == 0 || x == 1) ? 0.0f : input[(y + 1)*DATA_W + x - 2];
	window[16] = (y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 1)*DATA_W + x - 1];
	window[17] = (y == DATA_H - 1) ? 0.0f : input[(y + 1)*DATA_W + x];
	window[18] = (y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 1)*DATA_W + x + 1];
	window[19] = (y == DATA_H - 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y + 1)*DATA_W + x + 2];

	window[20] = (y == DATA_H - 2 || y == DATA_H - 1 || x == 0 || x == 1) ? 0.0f : input[(y + 2)*DATA_W + x - 2];
	window[21] = (y == DATA_H - 2 || y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 2)*DATA_W + x - 1];
	window[22] = (y == DATA_H - 2 || y == DATA_H - 1) ? 0.0f : input[(y + 2)*DATA_W + x];
	window[23] = (y == DATA_H - 2 || y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 2)*DATA_W + x + 1];
	window[24] = (y == DATA_H - 2 || y == DATA_H - 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y + 2)*DATA_W + x + 2];

	// Order elements (only half of them)
	for (unsigned int j = 0; j<13; ++j)
	{
		// Find position of minimum element
		int min = j;
		for (unsigned int l = j + 1; l<25; ++l)
			if (window[l] < window[min])
				min = l;

		// Put found minimum element in its place 
		float temp = window[j];
		window[j] = window[min];
		window[min] = temp;
	}

	if (((x < 2) && (y < 2)) || ((x > DATA_W - 2) && (y < 2)) || ((x < 2) && (y > DATA_H - 2)) || ((x > DATA_W - 2) && (y > DATA_H - 2)))
	{
		output[y*DATA_W + x] = input[y*DATA_W + x];
	}
	else
		output[y*DATA_W + x] = window[12];
};


template<typename Dtype>
__global__ void averageFilter5(Dtype * input, Dtype * output, unsigned int DATA_W, unsigned int DATA_H)
{
	float window[25];

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= DATA_W && y >= DATA_H)
		return;

	window[0] = (y == 0 || y == 1 || x == 0 || x == 1) ? 0.0f : input[(y - 2)*DATA_W + x - 2];
	window[1] = (y == 0 || y == 1 || x == 0) ? 0.0f : input[(y - 2)*DATA_W + x - 1];
	window[2] = (y == 0 || y == 1) ? 0.0f : input[(y - 2)*DATA_W + x];
	window[3] = (y == 0 || y == 1 || x == DATA_W - 1) ? 0.0f : input[(y - 2)*DATA_W + x + 1];
	window[4] = (y == 0 || y == 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y - 2)*DATA_W + x + 2];

	window[5] = (y == 0 || x == 0 || x == 1) ? 0.0f : input[(y - 1)*DATA_W + x - 2];
	window[6] = (y == 0 || x == 0) ? 0.0f : input[(y - 1)*DATA_W + x - 1];
	window[7] = (y == 0) ? 0.0f : input[(y - 1)*DATA_W + x];
	window[8] = (y == 0 || x == DATA_W - 1) ? 0.0f : input[(y - 1)*DATA_W + x + 1];
	window[9] = (y == 0 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y - 1)*DATA_W + x + 2];

	window[10] = (x == 0 || x == 1) ? 0.0f : input[y*DATA_W + x - 2];
	window[11] = (x == 0) ? 0.0f : input[y*DATA_W + x - 1];
	window[12] = input[y*DATA_W + x];
	window[13] = (x == DATA_W - 1) ? 0.0f : input[y*DATA_W + x + 1];
	window[14] = (x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[y*DATA_W + x + 2];

	window[15] = (y == DATA_H - 1 || x == 0 || x == 1) ? 0.0f : input[(y + 1)*DATA_W + x - 2];
	window[16] = (y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 1)*DATA_W + x - 1];
	window[17] = (y == DATA_H - 1) ? 0.0f : input[(y + 1)*DATA_W + x];
	window[18] = (y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 1)*DATA_W + x + 1];
	window[19] = (y == DATA_H - 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y + 1)*DATA_W + x + 2];

	window[20] = (y == DATA_H - 2 || y == DATA_H - 1 || x == 0 || x == 1) ? 0.0f : input[(y + 2)*DATA_W + x - 2];
	window[21] = (y == DATA_H - 2 || y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 2)*DATA_W + x - 1];
	window[22] = (y == DATA_H - 2 || y == DATA_H - 1) ? 0.0f : input[(y + 2)*DATA_W + x];
	window[23] = (y == DATA_H - 2 || y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 2)*DATA_W + x + 1];
	window[24] = (y == DATA_H - 2 || y == DATA_H - 1 || x == DATA_W - 1 || x == DATA_W - 2) ? 0.0f : input[(y + 2)*DATA_W + x + 2];

	float sum = 0.;
	for (unsigned int j = 0; j<25; ++j)
	{
		sum += window[j];
	}
	sum /= 25;
	if (((x < 2) && (y < 2)) || ((x > DATA_W - 2) && (y < 2)) || ((x < 2) && (y > DATA_H - 2)) || ((x > DATA_W - 2) && (y > DATA_H - 2)))
	{
		output[y*DATA_W + x] = input[y*DATA_W + x];
	}
	else
		output[y*DATA_W + x] = sum;
};



__global__ void downSample44Kernel(
	short*depth_dev1_midfiltered, int rows1, int cols1,
	int rows_lmt, int cols_lmt,
	short*depth_dev2, int rows2, int cols2,
	short*depth_dev3, int rows3, int cols3,
	short*depth_dev4, int rows4, int cols4)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols_lmt || y >= rows_lmt)
		return;
	int downsample2_sum = 0; int downsample2_idx = 0;
	int downsample3_sum = 0; int downsample3_idx = 0;
	int downsample4_sum = 0; int downsample4_idx = 0;
}



template<typename D1,typename D2>
__global__ void downSample44_2_Kernel(
	D1*depth_dev1, int rows1, int cols1,
	int rows_lmt, int cols_lmt,
	D2*depth_dev2, int rows2, int cols2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols_lmt || y >= rows_lmt)
		return;
	float downsample2_sum = 0; int downsample2_idx = 0;
	int xstart = 4 * x;
	int ystart = 4 * y;
	int xend = 4 * x+4;
	int yend = 4 * y+4;
	
	for (int i = ystart; i < yend; i++)
	{
		for (int j = xstart; j < xend; j++)
		{
			if (depth_dev1[i*cols1+j]>0)
			{
				downsample2_sum += depth_dev1[i*cols1 + j];
				downsample2_idx++;
			}
		}
	}
	if (downsample2_idx<10)
	{
		depth_dev2[y*cols2 + x].x = 0;
		depth_dev2[y*cols2 + x].y = 0;
	}
	else
	{
		depth_dev2[y*cols2 + x].x = static_cast<D1>(1.0*downsample2_sum / downsample2_idx);
		depth_dev2[y*cols2 + x].y = downsample2_idx;
	}
	
}

template<typename D1, typename D2>
__global__ void downSample44_2_Kernel(
	D2*depth_dev1, int rows1, int cols1,
	int rows_lmt, int cols_lmt,
	D2*depth_dev2, int rows2, int cols2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols_lmt || y >= rows_lmt)
		return;
	float downsample2_sum = 0; int downsample2_idx = 0;
	int xstart = 4 * x;
	int ystart = 4 * y;
	int xend = 4 * x + 4;
	int yend = 4 * y + 4;
	for (int i = ystart; i < yend; i++)
	{
		for (int j = xstart; j < xend; j++)
		{
			if (depth_dev1[i*cols1 + j].x>0)
			{
				downsample2_sum += depth_dev1[i*cols1 + j].y * depth_dev1[i*cols1 + j].x;
				downsample2_idx += depth_dev1[i*cols1 + j].y;
			}
		}
	}
	if (downsample2_idx<10)
	{
		depth_dev2[y*cols2 + x].x = 0;
		depth_dev2[y*cols2 + x].y = 0;
	}
	else
	{		
		depth_dev2[y*cols2 + x].x = static_cast<D1>(1.0*downsample2_sum / downsample2_idx);
		depth_dev2[y*cols2 + x].y = downsample2_idx;
	}

}



template<typename D1, typename D2>
__global__ void fillKernel(
	D1*depth_dev1_midfiltered, D1*fill_dev,
	int rows1, int cols1,
	D2*depth_dev2, int rows2, int cols2,
	D2*depth_dev3, int rows3, int cols3)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= cols1 || y >= rows1)
		return;
	if (depth_dev1_midfiltered[cols1*y + x] > 0.001)
	{
		fill_dev[cols1*y + x] = depth_dev1_midfiltered[cols1*y + x];
		return;
	}
	
	int stage2_x = x / 4;
	int stage2_y = y / 4;
	if (stage2_x >= cols2 || stage2_y >= rows2)
		return;
	D1 stage2Value = depth_dev2[cols2*stage2_y + stage2_x].x;

	bool isFilledWithStage2 = false;
	if (stage2Value>0.001)
	{
		int stage2_xstart = stage2_x - 1;
		int stage2_xend = stage2_x + 2;
		int stage2_ystart = stage2_y - 1;
		int stage2_yend = stage2_y + 2;
		short roundCnt = 0;
		short validCnt = 0;
		for (int i = stage2_ystart; i < stage2_yend; i++)
		{
			for (int j = stage2_xstart; j < stage2_xend; j++)
			{
				if (j == stage2_x || i == stage2_y)
					continue;
				if (j < 0 || i < 0 || j >= cols2 || i >= rows2)
					continue;
				if (depth_dev2[cols2*i + j].x>0)validCnt++;
				roundCnt++;
			}
		}
		if (1.0*validCnt / roundCnt>0.7)
		{
			fill_dev[cols1*y + x] = depth_dev2[cols2*stage2_y + stage2_x].x;
			isFilledWithStage2 = true;
		}
	}
	if (isFilledWithStage2)
	{
		return;
	}
	//******
	else
	{
		fill_dev[cols1*y + x] = numeric_limits<float>::quiet_NaN();;
		return;
	}
	//******
	int stage3_x = x / 16;
	int stage3_y = y / 16;
	if (stage3_x >= cols3 || stage3_y >= rows3)
		return;
	D1 stage3Value = depth_dev3[cols3*stage3_y + stage3_x].x;
	if (stage3Value > 0)
	{
		int stage3_xstart = stage3_x - 1;
		int stage3_xend = stage3_x + 2;
		int stage3_ystart = stage3_y - 1;
		int stage3_yend = stage3_y + 2;
		short roundCnt = 0;
		short validCnt = 0;
		for (int i = stage3_ystart; i < stage3_yend; i++)
		{
			for (int j = stage3_xstart; j < stage3_xend; j++)
			{
				if (j == stage3_x || i == stage3_y)
					continue;
				if (j < 0 || i < 0 || j >= cols3 || i >= rows3)
					continue;
				if (depth_dev3[cols3*i + j].x > 0)validCnt++;
				roundCnt++;
			}
		}
		if (1.0*validCnt / roundCnt > 0.7)
		{
			fill_dev[cols1*y + x] = depth_dev3[cols3*stage3_y + stage3_x].x;
			return;
		}
		else
		{
			fill_dev[cols1*y + x] = numeric_limits<float>::quiet_NaN();;
		}
	}	
	else
	{
		fill_dev[cols1*y + x] = numeric_limits<float>::quiet_NaN();;
	}
}


#ifdef DOWNSAMPLE3TIMES
void  midfilter33AndFillHoles44_downsample3t(short*depth_dev1, int rows1, int cols1,
	short*depth_dev1_midfiltered, short*depth_dev1_filled,
	short2*depth_dev2, int rows2, int cols2,
	short2*depth_dev3, int rows3, int cols3,
	short2*depth_dev4, int rows4, int cols4)
{
	dim3 dimBlock(BLOCK_W_FILTER, BLOCK_H_FILTER);
	dim3 dimGrid((cols1 + dimBlock.x - 1) / dimBlock.x, (rows1 + dimBlock.y - 1) / dimBlock.y);
	medianFilter3AndDiscardNoisyKernel<short> << <dimGrid, dimBlock >> >(depth_dev1, depth_dev1_midfiltered, cols1, rows1);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	int rows_lmt1 = rows1 / 4 ;
	int cols_lmt1 = cols1 / 4 ;
	dim3 block_downSample1(32, 24);
	dim3 grid_downSample1(divUp(cols_lmt1, block_downSample1.x), divUp(rows_lmt1, block_downSample1.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<short,short2><<< grid_downSample1,block_downSample1 >>>(
		depth_dev1_midfiltered, rows1, cols1,
		rows_lmt1, cols_lmt1,
		depth_dev2, rows2, cols2);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	int rows_lmt2 = rows1 / 16;
	int cols_lmt2 = cols1 / 16;
	dim3 block_downSample2(32, 24);
	dim3 grid_downSample2(divUp(cols_lmt2, block_downSample2.x), divUp(rows_lmt2, block_downSample2.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<short, short2> <<< grid_downSample2, block_downSample2 >>>(depth_dev2, rows2, cols2,
		rows_lmt2, cols_lmt2,
		depth_dev3, rows3, cols3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	int rows_lmt3 = rows1 / 64;
	int cols_lmt3 = cols1 / 64;
	dim3 block_downSample3(32, 24);
	dim3 grid_downSample3(divUp(cols_lmt3, block_downSample3.x), divUp(rows_lmt3, block_downSample3.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<short, short2> <<< grid_downSample3, block_downSample3 >>>(depth_dev3, rows3, cols3,
		rows_lmt3, cols_lmt3,
		depth_dev4, rows4, cols4);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#else
void  midfilter33AndFillHoles44_downsample2t(short*depth_dev1, int rows1, int cols1,
	short*depth_dev1_midfiltered, short*depth_dev1_filled,
	short2*depth_dev2, int rows2, int cols2,
	short2*depth_dev3, int rows3, int cols3)
{
	dim3 dimBlock(BLOCK_W_FILTER, BLOCK_H_FILTER);
	dim3 dimGrid((cols1 + dimBlock.x - 1) / dimBlock.x, (rows1 + dimBlock.y - 1) / dimBlock.y);
	medianFilter3AndDiscardNoisyKernel<short> << <dimGrid, dimBlock >> >(depth_dev1, depth_dev1_midfiltered, cols1, rows1);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());


	//********************************
	return;
	//********************************

	int rows_lmt1 = rows1 / 4;
	int cols_lmt1 = cols1 / 4;
	dim3 block_downSample1(32, 24);
	dim3 grid_downSample1(divUp(cols_lmt1, block_downSample1.x), divUp(rows_lmt1, block_downSample1.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<short, short2> << < grid_downSample1, block_downSample1 >> >(
		depth_dev1_midfiltered, rows1, cols1,
		rows_lmt1, cols_lmt1,
		depth_dev2, rows2, cols2);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	int rows_lmt2 = rows1 / 16;
	int cols_lmt2 = cols1 / 16;
	dim3 block_downSample2(32, 24);
	dim3 grid_downSample2(divUp(cols_lmt2, block_downSample2.x), divUp(rows_lmt2, block_downSample2.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<short, short2> << < grid_downSample2, block_downSample2 >> >(depth_dev2, rows2, cols2,
		rows_lmt2, cols_lmt2,
		depth_dev3, rows3, cols3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	int rows_global = rows1 ;
	int cols_global = cols1 ;
	dim3 block_global(32, 24);
	dim3 grid_global(divUp(cols_global, block_global.x), divUp(rows_global, block_global.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	fillKernel<short, short2> << <grid_global, block_global >> >(
		depth_dev1_midfiltered, depth_dev1_filled,
		rows1, cols1,
		depth_dev2,  rows2, cols2,
		depth_dev3,  rows3, cols3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

#endif




void medfilter33_forOneDev(
	float*depth_dev1, int rows1, int cols1, 
	float*depth_dev1_midfiltered, float*depth_dev1_filled,
	float2*depth_dev2, int rows2, int cols2,
	float2*depth_dev3, int rows3, int cols3)
{
	dim3 dimBlock(BLOCK_W_FILTER, BLOCK_H_FILTER);
	dim3 dimGrid((cols1 + dimBlock.x - 1) / dimBlock.x, (rows1 + dimBlock.y - 1) / dimBlock.y);
	medianFilter3AndDiscardNoisyKernel<float> << <dimGrid, dimBlock >> >
		(depth_dev1, depth_dev1_midfiltered, cols1, rows1);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	//********************************
	int rows_lmt1 = rows1 / 4;
	int cols_lmt1 = cols1 / 4;
	dim3 block_downSample1(32, 24);
	dim3 grid_downSample1(divUp(cols_lmt1, block_downSample1.x), divUp(rows_lmt1, block_downSample1.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<float, float2> << < grid_downSample1, block_downSample1 >> >(
		depth_dev1_midfiltered, rows1, cols1,
		rows_lmt1, cols_lmt1,
		depth_dev2, rows2, cols2);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	//int rows_lmt2 = rows1 / 16;
	//int cols_lmt2 = cols1 / 16;
	//dim3 block_downSample2(32, 24);
	//dim3 grid_downSample2(divUp(cols_lmt2, block_downSample2.x), divUp(rows_lmt2, block_downSample2.y));
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());	
	//downSample44_2_Kernel<float, float2> << < grid_downSample2, block_downSample2 >> >
	//	(depth_dev2, rows2, cols2,
	//	rows_lmt2, cols_lmt2,
	//	depth_dev3, rows3, cols3);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	int rows_global = rows1;
	int cols_global = cols1;
	dim3 block_global(32, 24);
	dim3 grid_global(divUp(cols_global, block_global.x), divUp(rows_global, block_global.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	fillKernel<float, float2> << <grid_global, block_global >> >(
		depth_dev1_midfiltered, depth_dev1_filled,
		rows1, cols1,
		depth_dev2, rows2, cols2,
		depth_dev3, rows3, cols3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());


	//averageFilter5 << <grid_global, block_global >> >(
	//	depth_dev1_filled,
	//	depth_dev1,
	//	cols1, rows1);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());
	return;
}


#ifdef AVERAGE_DEEP_3
template<typename Dtype1, typename Dtype2> __global__
void combineAverageDeepKernel
(const Dtype1*avg0, const Dtype1*avg1, const Dtype1*avg2, Dtype2*out, const int rows, const int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= cols || v >= rows)
		return;
	int pos = (cols*v + u);
	int notEmptyCnt = 0;
	float sum = 0.;

	if (avg0[pos] > 10 )
	{
		notEmptyCnt++;
		sum += avg0[pos];
	}
	if (avg1[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg1[pos];
	}
	if (avg2[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg2[pos];
	}
	if (notEmptyCnt<2)
	{
		out[pos] = 0;
	}
	else
	{
		out[pos] = static_cast<Dtype2>(sum/ notEmptyCnt);
	}
}
template<>
void combineAverageDeep<short,short>(const short*avg0, const short*avg1, const short*avg2, short*out, const int rows, const int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	combineAverageDeepKernel<short, short> << <grid, block >> >(avg0, avg1, avg2, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
template<>
void combineAverageDeep<short, float>(const short*avg0, const short*avg1, const short*avg2, float*out, const int rows, const int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	combineAverageDeepKernel<short, float> << <grid, block >> >(avg0, avg1, avg2, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#elif AVERAGE_DEEP_5
template<typename Dtype1, typename Dtype2> __global__
void combineAverageDeepKernel
(const Dtype1*avg0, const Dtype1*avg1, const Dtype1*avg2, const Dtype1*avg3, const Dtype1*avg4,
	Dtype2*out, const int rows, const int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= cols || v >= rows)
		return;
	int pos = (cols*v + u);
	int notEmptyCnt = 0;
	float sum = 0.;

	if (avg0[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg0[pos];
	}
	if (avg1[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg1[pos];
	}
	if (avg2[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg2[pos];
	}
	if (avg3[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg3[pos];
	}
	if (avg4[pos] > 10)
	{
		notEmptyCnt++;
		sum += avg4[pos];
	}
	if (notEmptyCnt<4)
	{
		out[pos] = 0;
	}
	else
	{
		out[pos] = static_cast<Dtype2>(sum / notEmptyCnt);
	}
}
template<>
void combineAverageDeep<short,short>(const short*avg0, const short*avg1, const short*avg2, const short*avg3, const short*avg4,
	short*out, const int rows, const int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	combineAverageDeepKernel<short, short> << <grid, block >> >(avg0, avg1, avg2, avg3, avg4, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
template<>
void combineAverageDeep<short, float>(const short*avg0, const short*avg1, const short*avg2, const short*avg3, const short*avg4,
	float*out, const int rows, const int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	combineAverageDeepKernel<short, float> << <grid, block >> >(avg0, avg1, avg2, avg3, avg4, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#endif // AVERAGE_DEEP_3


#if AVERAGE_DEEP_3_UPDATA || AVERAGE_DEEP_5_UPDATA || AVERAGE_DEEP_15_UPDATA
struct combineAverageDeepUpdataParam
{
	enum
	{
		HEIGHT = 8,
		WIDTH = 8,
		UPDATA_THRESHOLD = 10,
	};
};
#endif
#ifdef AVERAGE_DEEP_3_UPDATA
template<typename Dtype1, typename Dtype2> __global__ void
combineAverage3DeepUpdataKernel
(const Dtype1*avg0, const Dtype1*avg1, const Dtype1*avg2,
	Dtype2*out, const int rows, const int cols)
{
	__shared__ Dtype1 window[combineAverageDeepUpdataParam::WIDTH*combineAverageDeepUpdataParam::HEIGHT][3];
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid = threadIdx.y*blockDim.y + threadIdx.x;
	if (x >= cols && y >= rows)
		return;
	int pos = (cols*y + x);
	window[tid][0] = avg0[pos];
	window[tid][1] = avg1[pos];
	window[tid][2] = avg2[pos];
	for (int i = 0; i < 2; i++)
	{
		int minIdx = i;
		for (int j = i+1; j < 3; j++)
		{
			if (window[tid][minIdx] > window[tid][j]) minIdx = j;
		}
		if (minIdx != i)
		{
			float temp = window[tid][i];
			window[tid][i] = window[tid][minIdx];
			window[tid][minIdx] = temp;
		}
	}
	int minDiffIdx = 0;
	float minDiff = numeric_limits<Dtype2>::max();
	for (int i = 0; i < 2; i++)
	{
		if (abs(window[tid][i]- window[tid][i+1])<minDiff)
		{
			minDiffIdx = i;
		}
	}
	if (abs(out[pos]- window[tid][minDiffIdx])>combineAverageDeepUpdataParam::UPDATA_THRESHOLD)
	{
		out[pos] = window[tid][minDiffIdx];
	}
	else
	{
		out[pos] = out[pos] *0.9+ window[tid][minDiffIdx]*.1;
	}
	return;
}
template<> void
combineAverageDeep<short,float>(const short*avg0, const short*avg1, const short*avg2,
	float*out, const int rows, const int cols)
{
	dim3 dimBlock(combineAverageDeepUpdataParam::WIDTH, combineAverageDeepUpdataParam::HEIGHT);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
	combineAverage3DeepUpdataKernel<short, float> << <dimGrid, dimBlock >> >(avg0, avg1, avg2, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
template<> void
combineAverageDeep<short, short>(const short*avg0, const short*avg1, const short*avg2,
	short*out, const int rows, const int cols)
{
	dim3 dimBlock(combineAverageDeepUpdataParam::WIDTH, combineAverageDeepUpdataParam::HEIGHT);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
	combineAverage3DeepUpdataKernel<short, short> << <dimGrid, dimBlock >> >(avg0, avg1, avg2, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#elif AVERAGE_DEEP_5_UPDATA
template<typename Dtype> __global__ void
combineAverage5DeepUpdataKernel
(const Dtype*avg0, const Dtype*avg1, const Dtype*avg2, const Dtype*avg3, const Dtype*avg4,
	float*out, const int rows, const int cols)
{
	__shared__ Dtype window[combineAverageDeepUpdataParam::WIDTH*combineAverageDeepUpdataParam::HEIGHT][5];
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid = threadIdx.y*blockDim.y + threadIdx.x;
	if (x >= cols && y >= rows)
		return;
	int pos = (cols*y + x);
	window[tid][0] = avg0[pos];
	window[tid][1] = avg1[pos];
	window[tid][2] = avg2[pos];
	window[tid][3] = avg3[pos];
	window[tid][4] = avg4[pos];
	int emptyCnt = 0;
	for (int i = 0; i < 5; i++)
	{
		if (window[tid][i] < 10)emptyCnt++;
	}
	if (emptyCnt>1)
	{
		out[pos] = 0;
	}
	for (int i = 0; i < 4; i++)
	{
		int minIdx = i;
		for (int j = i + 1; j < 5; j++)
		{
			if (window[tid][minIdx] > window[tid][j]) minIdx = j;
		}
		if (minIdx != i)
		{
			float temp = window[tid][i];
			window[tid][i] = window[tid][minIdx];
			window[tid][minIdx] = temp;
		}
	}
	int minDiffIdx = 0;
	float minDiff = numeric_limits<Dtype>::max();
	for (int i = 0; i < 4; i++)
	{
		if (abs(window[tid][i] - window[tid][i + 1])<minDiff)
		{
			minDiffIdx = i;
		}
	}
	if (abs(out[pos] - window[tid][minDiffIdx])>combineAverageDeepUpdataParam::UPDATA_THRESHOLD)
	{
		out[pos] = window[tid][minDiffIdx]*2*0.5;
	}
	else
	{
		out[pos] = static_cast<int>(out[pos] * 0.5 + window[tid][minDiffIdx] * .5)*2*0.5;
	}
	return;
}
template<> void
combineAverageDeep<short>(const short*avg0, const short*avg1, const short*avg2, const short*avg3, const short*avg4,
	float*out, const int rows, const int cols)
{
	dim3 dimBlock(combineAverageDeepUpdataParam::WIDTH, combineAverageDeepUpdataParam::HEIGHT);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
	combineAverage5DeepUpdataKernel<short> << <dimGrid, dimBlock >> >(avg0, avg1, avg2, avg3, avg4, out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#elif AVERAGE_DEEP_15_UPDATA
template<typename Dtype> __global__ void
combineAverage5DeepUpdataKernel
(const Dtype*avg15,	float*out, const int rows, const int cols)
{
	__shared__ Dtype window[combineAverageDeepUpdataParam::WIDTH*combineAverageDeepUpdataParam::HEIGHT][15];
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tid = threadIdx.y*blockDim.y + threadIdx.x;
	if (x >= cols && y >= rows)
		return;
	int pos = (cols*y + x);
	int interval = rows*cols;
	for (int i = 0; i < 15; i++)
	{
		window[tid][i] = avg15[pos+i*interval];
	}
	int emptyCnt = 0;
	for (int i = 0; i < 15; i++)
	{
		if (window[tid][i] < 10)emptyCnt++;
	}
	if (emptyCnt>2)
	{
		out[pos] = 0;
	}
	for (int i = 0; i < 14; i++)
	{
		int minIdx = i;
		for (int j = i + 1; j < 15; j++)
		{
			if (window[tid][minIdx] > window[tid][j]) minIdx = j;
		}
		if (minIdx != i)
		{
			float temp = window[tid][i];
			window[tid][i] = window[tid][minIdx];
			window[tid][minIdx] = temp;
		}
	}
	int minDiffIdx = 0;
	float minDiff = numeric_limits<Dtype>::max();
	for (int i = 0; i < 14; i++)
	{
		if (abs(window[tid][i] - window[tid][i + 1])<minDiff)
		{
			minDiffIdx = i;
		}
	}
	if (abs(out[pos] - window[tid][minDiffIdx])>combineAverageDeepUpdataParam::UPDATA_THRESHOLD)
	{
		out[pos] = window[tid][minDiffIdx];
	}
	else
	{
		out[pos] = out[pos] * 0.9 + window[tid][minDiffIdx] * .1;
	}
	return;
}
template<> void
combineAverageDeep<short, float>(const short*avg15, float*out, const int rows, const int cols)
{
	dim3 dimBlock(combineAverageDeepUpdataParam::WIDTH, combineAverageDeepUpdataParam::HEIGHT);
	dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
	combineAverage5DeepUpdataKernel<short> << <dimGrid, dimBlock >> >(avg15,  out, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
#endif // AVERAGE_DEEP_3_UPDATA
