#include "unreGpu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

template<>
float* creatGpuData<float>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(float));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(float));
	}
	cudaSafeCall(cudaGetLastError());
	return (float*)gpudata;
}
template<>
short* creatGpuData<short>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(short));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(short));
	}
	cudaSafeCall(cudaGetLastError());
	return (short*)gpudata;
}
template<>
unsigned short* creatGpuData<unsigned short>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(unsigned short));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(unsigned short));
	}
	cudaSafeCall(cudaGetLastError());
	return (unsigned short*)gpudata;
}


__global__ void
scaleDepth(const unsigned short* depth, float* scaled, const int rows,const int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols || y >= rows)
		return;

	int Dp = depth[y*cols+x];

	float xl = (x - intr_cx) / intr_fx;
	float yl = (y - intr_cy) / intr_fy;
	float lambda = sqrtf(xl * xl + yl * yl + 1);

	scaled[y*cols + x] = Dp * lambda / 1000.f; //meters
}


__global__ void
tsdf23(const float* depthScaled, short* volume, int rows, int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy,
	const float tranc_dist, const double* R, const double* t, float3 &cell_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= VOLUME_X || y >= VOLUME_Y)
		return;

	float v_g_x = (x + 0.5f) * VOLUME_SIZE_X - t[0];
	float v_g_y = (y + 0.5f) * VOLUME_SIZE_Y - t[1];
	float v_g_z = (0 + 0.5f) * VOLUME_SIZE_Z - t[2];

	float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

	float v_x = (R[0] * v_g_x + R[1] * v_g_y + R[2] * v_g_z) * intr_fx;
	float v_y = (R[3] * v_g_x + R[4] * v_g_y + R[5] * v_g_z) * intr_fy;
	float v_z = (R[6] * v_g_x + R[7] * v_g_y + R[8] * v_g_z);

	float z_scaled = 0;

	float Rcurr_inv_0_z_scaled = R[2] * VOLUME_SIZE_Z * intr_fx;
	float Rcurr_inv_1_z_scaled = R[5] * VOLUME_SIZE_Z * intr_fy;

	float tranc_dist_inv = 1.0f / tranc_dist;
	
	short2* pos = 0;
	int elem_step = 0;
	//short2* pos = volume.ptr(y) + x;
	//int elem_step = volume.step * VOLUME_Y / sizeof(short2);

	//#pragma unroll
	for (int z = 0; z < VOLUME_Z;
		++z,
		v_g_z += VOLUME_SIZE_Z,
		z_scaled += VOLUME_SIZE_Z,
		v_x += Rcurr_inv_0_z_scaled,
		v_y += Rcurr_inv_1_z_scaled,
		pos += elem_step)
	{
		float inv_z = 1.0f / (v_z + R[8] * z_scaled);
		if (inv_z < 0)
			continue;

		// project to current cam
		int2 coo =
		{
			__float2int_rn(v_x * inv_z + intr_cx),
			__float2int_rn(v_y * inv_z + intr_cy)
		};

		if (coo.x >= 0 && coo.y >= 0 && coo.x < cols && coo.y < rows)         //6
		{
			float Dp_scaled = depthScaled[coo.y*cols+coo.x]; //meters

			float sdf = Dp_scaled - sqrtf(v_g_z * v_g_z + v_g_part_norm);

			if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
			{
				float tsdf = fmin(1.0f, sdf * tranc_dist_inv);

				//float tsdf_prev;
				//int weight_prev;
				//unpack_tsdf(*pos, tsdf_prev, weight_prev);
				//const int Wrk = 1;
				//float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
				//int weight_new = min(weight_prev + Wrk, Tsdf::MAX_WEIGHT);
				//pack_tsdf(tsdf_new, weight_new, *pos);
			}
		}
	}      
}      


void integrateTsdfVolume(const unsigned short* depth_raw, int rows, int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy,
	const double* R, const double* t, const float tranc_dist, short* volume, float* depthRawScaled)
{

	depthRawScaled = creatGpuData<float>(rows*cols);

	dim3 block_scale(32, 8);
	dim3 grid_scale(divUp(cols, block_scale.x), divUp(rows, block_scale.y));

	//scales depth along ray and converts mm -> meters. 
	scaleDepth << <grid_scale, block_scale >> >(depth_raw, depthRawScaled, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy);
	cudaSafeCall(cudaGetLastError());

	float3 cell_size;
	cell_size.x = VOLUME_SIZE_X / VOLUME_X;
	cell_size.y = VOLUME_SIZE_Y / VOLUME_Y;
	cell_size.z = VOLUME_SIZE_Z / VOLUME_Z;

	//dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
	dim3 block(16, 16);
	dim3 grid(divUp(VOLUME_X, block.x), divUp(VOLUME_Y, block.y));

	tsdf23 << <grid, block >> >(depthRawScaled, volume, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy,
		tranc_dist,R, t, cell_size);
	//tsdf23normal_hack<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}
