#ifndef _FILTERS_H_
#define _FILTERS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_WIDTH_MEANBLUR		32
#define BLOCK_HEIGHT_MEANBLUR	32

const unsigned int BLOCK_W_FILTER = 8;
const unsigned int BLOCK_H_FILTER = 8;

template<typename Dtype>
__global__ void CudaMedianFilter3(Dtype * input, Dtype * output, unsigned int DATA_W, unsigned int DATA_H)
{
	__shared__ float window[BLOCK_W_FILTER*BLOCK_H_FILTER][9];

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int tid = threadIdx.y*blockDim.y + threadIdx.x;

	if (x >= DATA_W && y >= DATA_H)
		return;

	window[tid][0] = (y == 0 || x == 0) ? 0.0f : input[(y - 1)*DATA_W + x - 1];
	window[tid][1] = (y == 0) ? 0.0f : input[(y - 1)*DATA_W + x];
	window[tid][2] = (y == 0 || x == DATA_W - 1) ? 0.0f : input[(y - 1)*DATA_W + x + 1];
	window[tid][3] = (x == 0) ? 0.0f : input[y*DATA_W + x - 1];
	window[tid][4] = input[y*DATA_W + x];
	window[tid][5] = (x == DATA_W - 1) ? 0.0f : input[y*DATA_W + x + 1];
	window[tid][6] = (y == DATA_H - 1 || x == 0) ? 0.0f : input[(y + 1)*DATA_W + x - 1];
	window[tid][7] = (y == DATA_H - 1) ? 0.0f : input[(y + 1)*DATA_W + x];
	window[tid][8] = (y == DATA_H - 1 || x == DATA_W - 1) ? 0.0f : input[(y + 1)*DATA_W + x + 1];

	// Order elements (only half of them)
	for (unsigned int j = 0; j<5; ++j)
	{
		// Find position of minimum element
		int min = j;
		for (unsigned int l = j + 1; l<9; ++l)
			if (window[tid][l] < window[tid][min])
				min = l;

		// Put found minimum element in its place
		float temp = window[tid][j];
		window[tid][j] = window[tid][min];
		window[tid][min] = temp;
	}
	if (((x < 1) && (y < 1)) || ((x > DATA_W - 1) && (y < 1)) || ((x < 1) && (y > DATA_H - 1)) || ((x > DATA_W - 1) && (y > DATA_H - 1)))
	{
		output[y*DATA_W + x] = input[y*DATA_W + x];
	}
	else
		output[y*DATA_W + x] = window[tid][4];
}

template<typename Dtype>
__global__ void CudaMedianFilter5(Dtype * input, Dtype * output, unsigned int DATA_W, unsigned int DATA_H)
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

template<unsigned short RADIUS >
__global__ void kRadialBlur(unsigned char* img, unsigned width, unsigned height, size_t pitch)
{
	__shared__ unsigned char sh[BLOCK_HEIGHT_MEANBLUR + 2 * RADIUS][BLOCK_WIDTH_MEANBLUR + 2 * RADIUS];

	int g_x = blockDim.x*blockIdx.x + threadIdx.x;
	int g_y = blockDim.y*blockIdx.y + threadIdx.y;

	int pid_x = threadIdx.x + RADIUS;
	int pid_y = threadIdx.y + RADIUS;

	
	sh[pid_y][pid_x] = img[g_y*pitch + g_x];


	if ((threadIdx.x < RADIUS) && (g_x >= RADIUS))
	{
		sh[pid_y][pid_x - RADIUS] = img[g_y*pitch + g_x - RADIUS];

		if ((threadIdx.y < RADIUS) && (g_y >= RADIUS))
		{
			sh[pid_y - RADIUS][pid_x - RADIUS] = img[(g_y - RADIUS)*pitch + g_x - RADIUS];
		}
		if ((threadIdx.y >(BLOCK_HEIGHT_MEANBLUR - 1 - RADIUS)))
		{
			sh[pid_y + RADIUS][pid_x - RADIUS] = img[(g_y + RADIUS)*pitch + g_x - RADIUS];
		}
	}
	if ((threadIdx.x > (BLOCK_WIDTH_MEANBLUR - 1 - RADIUS)) && (g_x < (width - RADIUS)))
	{
		sh[pid_y][pid_x + RADIUS] = img[g_y*pitch + g_x + RADIUS];

		if ((threadIdx.y < RADIUS) && (g_y > RADIUS))
		{
			sh[pid_y - RADIUS][pid_x + RADIUS] = img[(g_y - RADIUS)*pitch + g_x + RADIUS];
		}
		if ((threadIdx.y >(BLOCK_HEIGHT_MEANBLUR - 1 - RADIUS)) && (g_y < (height - RADIUS)))
		{
			sh[pid_y + RADIUS][pid_x + RADIUS] = img[(g_y + RADIUS)*pitch + g_x + RADIUS];
		}
	}

	if ((threadIdx.y < RADIUS) && (g_y >= RADIUS))
	{
		sh[pid_y - RADIUS][pid_x] = img[(g_y - RADIUS)*pitch + g_x];
	}
	if ((threadIdx.y >(BLOCK_HEIGHT_MEANBLUR - 1 - RADIUS)) && (g_y < (height - RADIUS)))
	{
		sh[pid_y + RADIUS][pid_x] = img[(g_y + RADIUS)*pitch + g_x];
	}

	__syncthreads();
	unsigned val = 0;
	unsigned k = 0;
	for (int i = -RADIUS; i <= RADIUS; i++)
		for (int j = -RADIUS; j <= RADIUS; j++)
		{
			if (((g_x + j) < 0) || ((g_x + j) > (width - 1)))
				continue;
			if (((g_y + i) < 0) || ((g_y + i) > (height - 1)))
				continue;
			val += sh[pid_y + i][pid_x + j];
			k++;
		}
	val /= k;
	img[g_y*pitch + g_x] = (unsigned char)val;

}

template<typename Dtype>
__global__ void averageFilter_3c(const Dtype * input, Dtype * output, 
	const unsigned int DATA_W, const  unsigned int DATA_H,const int radiusK)
{
	//float window_0[25];
	//float window_1[25];
	//float window_2[25];
	float sum0 = 0.;
	float sum1 = 0.;
	float sum2 = 0.;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= DATA_W && y >= DATA_H)
		return;
	int pos = 3*(DATA_W*y + x);
	int availableCnt = 1;
	for (int i = y - radiusK; i < y + radiusK; i++)
	{
		for (int j = x - radiusK; j < x + radiusK; j++)
		{
			int thisPos = 3 * (DATA_W*i + j);
			if (i<0 || j<0 || j >= DATA_W || i >= DATA_H)
			{
				continue;
			}
			if (input[thisPos] == 0&& input[thisPos+1] == 0&& input[thisPos+2] == 0)
			{
				continue;
			}
			sum0 += input[thisPos];
			sum1 += input[thisPos + 1];
			sum2 += input[thisPos + 2];
			availableCnt++;
		}
	}


	if (availableCnt == 0)
	{
		output[pos] = numeric_limits<Dtype>::quiet_NaN();
		output[pos + 1] = numeric_limits<Dtype>::quiet_NaN();
		output[pos + 2] = numeric_limits<Dtype>::quiet_NaN();
	}
	else
	{
		output[pos] = sum0/ availableCnt;
		output[pos + 1] = sum1 / availableCnt;
		output[pos + 2] = sum2 / availableCnt;
	}
};


template<typename Dtype>
void CudaMedianFilter(Dtype ** pImage, int imageWidth, int imageHeight, int kernelSize)
{
	Dtype* pTmpImage = NULL;
	cudaMalloc((void**)&pTmpImage, imageWidth*imageHeight * sizeof(uchar));


	dim3 dimBlock(BLOCK_W_FILTER, BLOCK_H_FILTER);
	dim3 dimGrid((imageWidth + dimBlock.x - 1) / dimBlock.x, (imageHeight + dimBlock.y - 1) / dimBlock.y);

	Dtype *d_input;

	cudaMemcpy(d_input, *pImage, imageWidth*imageHeight * sizeof(uchar), cudaMemcpyHostToDevice);

	if (kernelSize == 3)
	{
		CudaMedianFilter3<Dtype> << <dimGrid, dimBlock >> >(d_input, pTmpImage, imageWidth, imageHeight);
	}
	else if (kernelSize == 5)
	{
		CudaMedianFilter5<Dtype> << <dimGrid, dimBlock >> >(d_input, pTmpImage, imageWidth, imageHeight);
	}

	cudaMemcpy(*pImage, pTmpImage, imageWidth*imageHeight * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaFree(pTmpImage);
}


//{
//	dim3 dGrid(width / BLOCK_WIDTH_MEANBLUR, height / BLOCK_HEIGHT_MEANBLUR);
//	dim3 dBlock(BLOCK_WIDTH_MEANBLUR, BLOCK_HEIGHT_MEANBLUR);
//
//	// execution of the version using global memory
//	cudaEventRecord(startEvent);
//	kRadialBlur<4> << < dGrid, dBlock >> > (d_img, width, height, pitch);
//	cudaThreadSynchronize();
//}


#endif // !_FILTERS_H_
