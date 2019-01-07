#include"unreGpu.h"

__global__
void combineNmap2RgbKernel1(
	unsigned char*rgb, float*nmap,
	unsigned char*rgbOut,
	int colorRows, int colorCols
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= colorCols || y >= colorRows)
		return;
	int pos_ = colorCols*y + x;
	float n_r = -0.58*nmap[3 * pos_ + 2];
	float n_g = -0.58*nmap[3 * pos_ + 2];
	float n_b = -0.58*nmap[3 * pos_ + 2];
	float test_in = max(0.,n_r + n_g + n_b);

	//int newR = rgb[3 * pos_ + 0] * test_in;
	//int newG = rgb[3 * pos_ + 1] * test_in;
	//int newB = rgb[3 * pos_ + 2] * test_in;
	//rgbOut[3 * pos_] = max(0, min(255, newR));
	//rgbOut[3 * pos_ + 1] = max(0, min(255, newG));
	//rgbOut[3 * pos_ + 2] = max(0, min(255, newB));

	int newR = 100* test_in;
	int newG = newR;
	int newB = 120 * test_in;
	rgbOut[3 * pos_] = max(0, min(255, newB));
	rgbOut[3 * pos_+1] = max(0, min(255, newG));
	rgbOut[3 * pos_+2] = max(0, min(255, newR));
}


void combineNmap2Rgb(
	unsigned char*rgb, float*nmap,
	unsigned char*rgbOut,
	int colorRows, int colorCols)
{
	dim3 block_global(32, 24);
	dim3 grid_global(divUp(colorCols, block_global.x), divUp(colorRows, block_global.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	combineNmap2RgbKernel1 << <grid_global, block_global >> > (rgb, nmap,rgbOut,colorRows, colorCols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}