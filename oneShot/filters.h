#ifndef _FILTERS_H_
#define _FILTERS_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



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



#endif // !_FILTERS_H_
