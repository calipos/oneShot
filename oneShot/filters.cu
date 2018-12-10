#include "unreGpu.h"
#include "filters.h"


//暂时没有去除飞点，考虑了一下不合适
__global__ void medianFilter3AndDiscardNoisyKernel(short * input, short * output, unsigned int DATA_W, unsigned int DATA_H)
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
		
	for (unsigned int j = 0; j<5; ++j)
	{
		int min = j;
		for (unsigned int l = j + 1; l<9; ++l)
			if (window[tid][l] < window[tid][min])
				min = l;
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

__global__ void downSample44_2_Kernel(
	short*depth_dev1, int rows1, int cols1,
	int rows_lmt, int cols_lmt,
	short2*depth_dev2, int rows2, int cols2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols_lmt || y >= rows_lmt)
		return;
	int downsample2_sum = 0; int downsample2_idx = 0;
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
		depth_dev2[y*cols2 + x].x = static_cast<short>(1.0*downsample2_sum / downsample2_idx);
		depth_dev2[y*cols2 + x].y = downsample2_idx;
	}
	
}

__global__ void downSample44_2_Kernel(
	short2*depth_dev1, int rows1, int cols1,
	int rows_lmt, int cols_lmt,
	short2*depth_dev2, int rows2, int cols2)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols_lmt || y >= rows_lmt)
		return;
	int downsample2_sum = 0; int downsample2_idx = 0;
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
		depth_dev2[y*cols2 + x].x = static_cast<short>(1.0*downsample2_sum / downsample2_idx);
		depth_dev2[y*cols2 + x].y = downsample2_idx;
	}

}

__global__ void fillKernel(
	short*depth_dev1_midfiltered, short*fill_dev,
	int rows1, int cols1,
	short2*depth_dev2, int rows2, int cols2,
	short2*depth_dev3, int rows3, int cols3)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= cols1 || y >= rows1)
		return;
	if (depth_dev1_midfiltered[cols1*y + x] > 1)
	{
		fill_dev[cols1*y + x] = depth_dev1_midfiltered[cols1*y + x];
		return;
	}
	
	int stage2_x = x / 4;
	int stage2_y = y / 4;
	if (stage2_x >= cols2 || stage2_y >= rows2)
		return;
	short stage2Value = depth_dev2[cols2*stage2_y + stage2_x].x;

	bool isFilledWithStage2 = false;
	if (stage2Value>0)
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
	int stage3_x = x / 16;
	int stage3_y = y / 16;
	if (stage3_x >= cols3 || stage3_y >= rows3)
		return;
	short stage3Value = depth_dev3[cols3*stage3_y + stage3_x].x;
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
	medianFilter3AndDiscardNoisyKernel << <dimGrid, dimBlock >> >(depth_dev1, depth_dev1_midfiltered, cols1, rows1);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	int rows_lmt1 = rows1 / 4 ;
	int cols_lmt1 = cols1 / 4 ;
	dim3 block_downSample1(32, 24);
	dim3 grid_downSample1(divUp(cols_lmt1, block_downSample1.x), divUp(rows_lmt1, block_downSample1.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	downSample44_2_Kernel<<< grid_downSample1,block_downSample1 >>>(
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
	downSample44_2_Kernel<<< grid_downSample2, block_downSample2 >>>(depth_dev2, rows2, cols2,
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
	downSample44_2_Kernel<<< grid_downSample3, block_downSample3 >>>(depth_dev3, rows3, cols3,
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
	medianFilter3AndDiscardNoisyKernel << <dimGrid, dimBlock >> >(depth_dev1, depth_dev1_midfiltered, cols1, rows1);
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
	downSample44_2_Kernel << < grid_downSample1, block_downSample1 >> >(
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
	downSample44_2_Kernel << < grid_downSample2, block_downSample2 >> >(depth_dev2, rows2, cols2,
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
	fillKernel << <grid_global, block_global >> >(
		depth_dev1_midfiltered, depth_dev1_filled,
		rows1, cols1,
		depth_dev2,  rows2, cols2,
		depth_dev3,  rows3, cols3);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

#endif