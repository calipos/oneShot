
#include "unreGpu.h"




short2*volume = NULL;
struct Tsdf
{
	enum
	{
		CTA_SIZE_X = 32, CTA_SIZE_Y = 8,
		MAX_WEIGHT = 1 << 7
	};
};

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
short2* creatGpuData<short2>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(short2));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(short2));
	}
	cudaSafeCall(cudaGetLastError());
	return (short2*)gpudata;
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
template<>
float2* creatGpuData<float2>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(float2));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(float2));
	}
	cudaSafeCall(cudaGetLastError());
	return (float2*)gpudata;
}
template<>
float3* creatGpuData<float3>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(float3));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(float3));
	}
	cudaSafeCall(cudaGetLastError());
	return (float3*)gpudata;
}


__device__ __forceinline__ void
pack_tsdf(float tsdf, int weight, short2& value)
{
	int fixedp = max(-DIVISOR, min(DIVISOR, __float2int_rz(tsdf * DIVISOR)));
	//int fixedp = __float2int_rz(tsdf * DIVISOR);
	value = make_short2(fixedp, weight);
}

__device__ __forceinline__ void
pack_tsdf(float tsdf, float weight, short2& value)
{
	short fixedp = max(-DIVISOR, min(DIVISOR, __float2int_rz(tsdf * DIVISOR)));
	short fixeweight = __float2int_rz(weight * DIVISOR);
	value = make_short2(fixedp, fixeweight);
}

__device__ __forceinline__ void
pack_tsdf(short tsdf, short weight, short2& value)
{
	value = make_short2(tsdf, weight);
}

__device__ __forceinline__ void
unpack_tsdf(short2 value, float& tsdf, int& weight)
{
	weight = value.y;
	tsdf = __int2float_rn(value.x) / DIVISOR;   //*/ * INV_DIV;
}

__device__ __forceinline__ float
unpack_tsdf(short2 value)
{
	return static_cast<float>(value.x) / DIVISOR;    //*/ * INV_DIV;
}

int initVolu(short*&depth_dev, float*&scaledDepth, float3*&dev_vmap,
	short*&depth_midfiltered, short*&depth_filled,
	short2*&depth_2, short2*&depth_3,
	int depthRows, int depthCols)
{
	depth_dev = creatGpuData<short>(depthRows*depthCols);//体素的空间
	scaledDepth = creatGpuData<float>(depthRows*depthCols);//计算体素的中间会重scale depth
	dev_vmap = creatGpuData<float3>(depthRows*depthCols, true);//用以接受从体素模型中扫出的点云
	volume = creatGpuData<short2>(VOLUME_X*VOLUME_Y*VOLUME_Z);

	depth_midfiltered = creatGpuData<short>(depthRows*depthCols);
	depth_filled = creatGpuData<short>(depthRows*depthCols);
	int downsample_h2 = depthRows / 4;
	int downsample_w2 = depthCols / 4;
	depth_2 = creatGpuData<short2>(downsample_h2*downsample_w2);
	int downsample_h3 = depthRows / 16;
	int downsample_w3 = depthCols / 16;
	depth_3 = creatGpuData<short2>(downsample_h3*downsample_w3);

	return 0;
}


__global__ void
scaleDepth(const short* depth, float* scaled, const int rows, const int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= cols || y >= rows)
		return;



	int Dp = depth[y*cols + x];
	float xl = (x - intr_cx) / intr_fx;
	float yl = (y - intr_cy) / intr_fy;
	float lambda = sqrtf(xl * xl + yl * yl + 1);
	scaled[y*cols + x] = Dp * lambda / 1000.f; //meters

}


__global__ void
tsdf23(const float* depthScaled, short2* volume, int rows, int cols,
	const float intr_cx, const float intr_cy, const float intr_fx, const float intr_fy,
	const float tranc_dist, const Mat33 R, const float3 t, float3 cell_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= VOLUME_X || y >= VOLUME_Y)
		return;

	float v_g_x = (x + 0.5f) * cell_size.x + t.x;
	float v_g_y = (y + 0.5f) * cell_size.y + t.y;
	float v_g_z = (0 + 0.5f) * cell_size.z + t.z;

	float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

	float v_x = (R.data[0].x * v_g_x + R.data[0].y * v_g_y + R.data[0].z * v_g_z) * intr_fx;
	float v_y = (R.data[1].x * v_g_x + R.data[1].y * v_g_y + R.data[1].z * v_g_z) * intr_fy;
	float v_z = (R.data[2].x * v_g_x + R.data[2].y * v_g_y + R.data[2].z * v_g_z);

	float z_scaled = 0;

	float Rcurr_inv_0_z_scaled = R.data[0].z * cell_size.z * intr_fx;
	float Rcurr_inv_1_z_scaled = R.data[1].z * cell_size.z * intr_fy;

	float tranc_dist_inv = 1.0f / tranc_dist;

	short2* pos = volume + y*VOLUME_X + x;
	int elem_step = VOLUME_X * VOLUME_Y;

	//#pragma unroll
	for (int z = 0; z < VOLUME_Z;
		++z,
		v_g_z += cell_size.z,
		z_scaled += cell_size.z,
		v_x += Rcurr_inv_0_z_scaled,
		v_y += Rcurr_inv_1_z_scaled,
		pos += elem_step)
	{
		float inv_z = 1.0f / (v_z + R.data[2].z * z_scaled);
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
			float Dp_scaled = depthScaled[coo.y*cols + coo.x]; //meters

			float sdf = Dp_scaled - sqrtf(v_g_z * v_g_z + v_g_part_norm);

			if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
			{
				float tsdf = fmin(1.0f, sdf * tranc_dist_inv);
				//read and unpack
				float tsdf_prev;
				int weight_prev;
				unpack_tsdf(*pos, tsdf_prev, weight_prev);

				const int Wrk = 1;

				float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
				int weight_new = min(weight_prev + Wrk, Tsdf::MAX_WEIGHT);

				//pack_tsdf(tsdf_new, weight_new, *pos);
				pack_tsdf(tsdf_new, weight_new, *pos);
			}
		}
	}
}


__global__ void
self_volume_kernel(const float* depth_raw, short2* volume, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	float tranc_dist, Mat33 R, float3 t, float3 cameraPos, float3 cell_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= VOLUME_X || y >= VOLUME_Y)
		return;

	float v_g_x = (x + 0.5f) * cell_size.x;
	float v_g_y = (y + 0.5f) * cell_size.y;
	float v_g_z = (0.5f) * cell_size.z;
	float v_x = (R.data[0].x * v_g_x + R.data[0].y * v_g_y + R.data[0].z * v_g_z) + t.x;
	float v_y = (R.data[1].x * v_g_x + R.data[1].y * v_g_y + R.data[1].z * v_g_z) + t.y;
	float v_z = (R.data[2].x * v_g_x + R.data[2].y * v_g_y + R.data[2].z * v_g_z) + t.z;
	float vx_inCamera = v_x;
	float vy_inCamera = v_y;
	float vz_inCamera = v_z;
	float vx_inCamera_increase = R.data[0].z * cell_size.z;
	float vy_inCamera_increase = R.data[1].z * cell_size.z;
	float vz_inCamera_increase = R.data[2].z * cell_size.z;

	v_x = (v_x)* intr_cx;
	v_y = (v_y)* intr_cy;

	float z_scaled = 0;

	float Rcurr_inv_0_z_scaled = R.data[0].z * cell_size.z * intr_fx;
	float Rcurr_inv_1_z_scaled = R.data[1].z * cell_size.z * intr_fy;

	float tranc_dist_inv = 1.0f / tranc_dist;

	int elem_step = VOLUME_X * VOLUME_Y;
	short2* pos = volume + y*VOLUME_X + x;


	//#pragma unroll
	for (int z = 0; z <VOLUME_Z;
		++z,
		v_g_z += cell_size.z,
		z_scaled += R.data[2].z*cell_size.z,
		v_x += Rcurr_inv_0_z_scaled,
		v_y += Rcurr_inv_1_z_scaled,
		vx_inCamera += vx_inCamera_increase,
		vy_inCamera += vy_inCamera_increase,
		vz_inCamera += vz_inCamera_increase,
		pos += elem_step)
	{
		float this_v_z = v_z + z_scaled;
		float inv_z = 1.0f / (this_v_z);
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
			//v_part_norm += this_z*this_z;
			float distance_sqr = vx_inCamera*vx_inCamera + vy_inCamera*vy_inCamera + vz_inCamera*vz_inCamera;
			float radius = sqrtf((coo.x - cols / 2)*(coo.x - cols / 2) + (coo.y - rows / 2)*(coo.y - rows / 2));
			//float weight = radius/ distance;
			float weight = 1.f / radius;
			float Dp_scaled = depth_raw[coo.y*cols + coo.x]; //mm
			float distance = sqrtf(distance_sqr);

			float sdf = Dp_scaled - distance;

			if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
			{
				pack_tsdf(fmin(1.f, sdf*tranc_dist_inv), weight, *pos);
				//pos->x = 32000;
				//pos->y = 32000;
				//pack_tsdf(short(sdf), short(0), *pos);
			}
		}
		else
		{
			pack_tsdf(1.f, 0, *pos);
		}
	}
}

__global__ void
self_volume_kernel2(const float* depth_raw, short2* volume, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	float tranc_dist, Mat33 R, float3 t, float3 cell_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= VOLUME_X || y >= VOLUME_Y)
		return;

	float v_g_x = (x + 0.5f) * cell_size.x;
	float v_g_y = (y + 0.5f) * cell_size.y;
	float v_g_z = (0.5f) * cell_size.z;

	float v_x = (R.data[0].x * v_g_x + R.data[0].y * v_g_y + R.data[0].z * v_g_z) + t.x;
	float v_y = (R.data[1].x * v_g_x + R.data[1].y * v_g_y + R.data[1].z * v_g_z) + t.y;
	float v_z = (R.data[2].x * v_g_x + R.data[2].y * v_g_y + R.data[2].z * v_g_z) + t.z;

	float v_part_norm = v_x * v_x + v_y * v_y;

	v_x = (v_x)* intr_cx;
	v_y = (v_y)* intr_cy;

	float z_scaled = 0;

	float Rcurr_inv_0_z_scaled = R.data[0].z * cell_size.z * intr_fx;
	float Rcurr_inv_1_z_scaled = R.data[1].z * cell_size.z * intr_fy;

	float tranc_dist_inv = 1.0f / tranc_dist;

	int elem_step = VOLUME_X * VOLUME_Y;
	short2* pos = volume + y*VOLUME_X + x;


	//#pragma unroll
	for (int z = 0; z <VOLUME_Z;
		++z,
		v_g_z -= cell_size.z,
		z_scaled -= R.data[2].z*cell_size.z,
		v_x -= Rcurr_inv_0_z_scaled,
		v_y -= Rcurr_inv_1_z_scaled,
		pos += elem_step)
	{
		float this_v_z = v_z - z_scaled;
		float inv_z = 1.0f / (this_v_z);
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
			//v_part_norm += this_z*this_z;
			float distance_sqr = v_part_norm + this_v_z*this_v_z;
			float radius = sqrtf((coo.x - cols / 2)*(coo.x - cols / 2) + (coo.y - rows / 2)*(coo.y - rows / 2));
			//float weight = radius/ distance;
			float weight = 1.f / radius;
			float Dp_scaled = depth_raw[coo.y*cols + coo.x]; //mm
			float distance = sqrtf(distance_sqr);

			float sdf = Dp_scaled - distance;

			if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
			{
				pack_tsdf(fmin(1.f, sdf*tranc_dist_inv), weight, *pos);
				//pos->x = 32000;
				//pos->y = 32000;
				//pack_tsdf(short(sdf), short(0), *pos);
			}
		}
		else
		{
			pack_tsdf(1.f, 0, *pos);
		}
	}
}




void integrateTsdfVolume(const short* depth_raw, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R, float3 t, float3 cameraPos, float tranc_dist, short2* volume, float*&depthRawScaled)
{


	dim3 block_scale(32, 8);
	dim3 grid_scale(divUp(cols, block_scale.x), divUp(rows, block_scale.y));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	//scales depth along ray and converts mm -> meters. 
	scaleDepth << <grid_scale, block_scale >> >(depth_raw, depthRawScaled, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	float3 cell_size;
	cell_size.x = 1.*VOLUME_SIZE_X / VOLUME_X;
	cell_size.y = 1.*VOLUME_SIZE_Y / VOLUME_Y;
	cell_size.z = 1.*VOLUME_SIZE_Z / VOLUME_Z;

	cudaMemset(volume, 0, VOLUME_X*VOLUME_Y*VOLUME_Z * sizeof(short2));
	//dim3 block(Tsdf::CTA_SIZE_X, Tsdf::CTA_SIZE_Y);
	dim3 block(16, 16);
	dim3 grid(divUp(VOLUME_X, block.x), divUp(VOLUME_Y, block.y));
	self_volume_kernel << <grid, block >> >(depthRawScaled, volume, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy,
		tranc_dist, R, t, cameraPos, cell_size);
	//tsdf23 << <grid, block >> >(depthRawScaled, volume, rows, cols,
	//	intr_cx, intr_cy, intr_fx, intr_fy,
	//	tranc_dist,R, t, cell_size);
	//tsdf23normal_hack<<<grid, block>>>(depthScaled, volume, tranc_dist, Rcurr_inv, tcurr, intr, cell_size);

	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}


