#include"unreGpu.h"

//inline float rsqrtf(float x)
//{
//	return 1.0f / sqrtf(x);
//}

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
};

template<> struct numeric_limits<short>
{
	__device__ __forceinline__ static short
		max() { return SHRT_MAX; };
};


__device__ __forceinline__ float3
normalized(const float3& v)
{
	return v * rsqrt(dot(v, v));
}

		struct RayCaster
		{
			enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };
		};

		__device__ __forceinline__ float
			getMinTime(const float3& origin, const float3& dir)
		{
			float txmin = ((dir.x > 0 ? 0.f : VOLUME_SIZE_X) - origin.x) / dir.x;
			float tymin = ((dir.y > 0 ? 0.f : VOLUME_SIZE_Y) - origin.y) / dir.y;
			float tzmin = ((dir.z > 0 ? 0.f : VOLUME_SIZE_Z) - origin.z) / dir.z;
			return fmax(fmax(txmin, tymin), tzmin);
		}

		__device__ __forceinline__ float
			getMaxTime(const float3& origin, const float3& dir)
		{
			float txmax = ((dir.x > 0 ? VOLUME_SIZE_X : 0.f) - origin.x) / dir.x;
			float tymax = ((dir.y > 0 ? VOLUME_SIZE_Y : 0.f) - origin.y) / dir.y;
			float tzmax = ((dir.z > 0 ? VOLUME_SIZE_Z : 0.f) - origin.z) / dir.z;
			return fmin(fmin(txmax, tymax), tzmax);
		}

		__device__ __forceinline__ float3
			get_ray_next(int x, int y,
				const float intr_cx, const float intr_cy, 
				const float intr_fx, const float intr_fy)
		{
			float3 ray_next;
			ray_next.x = (x - intr_cx) / intr_fx;
			ray_next.y = (y - intr_cy) / intr_fy;
			ray_next.z = 1;
			return ray_next;
		}

		__device__ __forceinline__ float
			readTsdf(const short2* volume,int x, int y, int z)
		{
			return static_cast<float>(volume[z*VOLUME_X*VOLUME_Y + y*VOLUME_X + x].x) / DIVISOR;
		}
		__device__ __forceinline__ float
			readTsdfWeight(const short2* volume, int x, int y, int z)
		{
			return static_cast<float>(volume[z*VOLUME_X*VOLUME_Y + y*VOLUME_X + x].y) / DIVISOR;
		}

		__device__ __forceinline__ int3
			getVoxel(float3 point) 
		{
			int vx = __float2int_rd(point.x / VOLUME_SIZE_X);        // round to negative infinity
			int vy = __float2int_rd(point.y / VOLUME_SIZE_Y);
			int vz = __float2int_rd(point.z / VOLUME_SIZE_Z);
			return make_int3(vx, vy, vz);
		}

		__device__ __forceinline__ bool
			checkInds(const int3& g)
		{
			return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_Y && g.z < VOLUME_Z);
		}

		

		__device__ __forceinline__ float
			interpolateTrilineary(const short2* volume,const float3& point)
		{
			int3 g = getVoxel(point);

			if (g.x <= 0 || g.x >= VOLUME_X - 1)
				return numeric_limits<float>::quiet_NaN();

			if (g.y <= 0 || g.y >= VOLUME_Y - 1)
				return numeric_limits<float>::quiet_NaN();

			if (g.z <= 0 || g.z >= VOLUME_Z - 1)
				return numeric_limits<float>::quiet_NaN();

			float vx = (g.x + 0.5f) * VOLUME_SIZE_X;
			float vy = (g.y + 0.5f) * VOLUME_SIZE_Y;
			float vz = (g.z + 0.5f) * VOLUME_SIZE_Z;

			g.x = (point.x < vx) ? (g.x - 1) : g.x;
			g.y = (point.y < vy) ? (g.y - 1) : g.y;
			g.z = (point.z < vz) ? (g.z - 1) : g.z;

			float a = (point.x - (g.x + 0.5f) * VOLUME_SIZE_X) / VOLUME_SIZE_X;
			float b = (point.y - (g.y + 0.5f) * VOLUME_SIZE_Y) / VOLUME_SIZE_Y;
			float c = (point.z - (g.z + 0.5f) * VOLUME_SIZE_Z) / VOLUME_SIZE_Z;

			float res = readTsdf(volume, g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
				readTsdf(volume, g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
				readTsdf(volume, g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
				readTsdf(volume, g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
				readTsdf(volume, g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
				readTsdf(volume, g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
				readTsdf(volume, g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
				readTsdf(volume, g.x + 1, g.y + 1, g.z + 1) * a * b * c;
			return res;
		}

		__device__ __forceinline__ float
			interpolateTrilineary(const short2* volume, const float3& origin, const float3& dir, float time)
		{
			return interpolateTrilineary(volume, origin + dir * time);
		}

		__global__  void
			rayCastKernel(const short2* volume, float3* vmap, int rows, int cols,
				const float intr_cx, const float intr_cy, 
				const float intr_fx, const float intr_fy,
				const Mat33 R_, const float3 t_, const float tranc_dist)
		{
			int x = threadIdx.x + blockIdx.x * RayCaster::CTA_SIZE_X;
			int y = threadIdx.y + blockIdx.y * RayCaster::CTA_SIZE_Y;

			if (x >= cols || y >= rows)
				return;

			
			float3 ray_start = t_;
			float3 ray_next = R_ * get_ray_next(x, y, intr_cx, intr_cy,
				intr_fx, intr_fy) + t_;

			float3 ray_dir = normalized(ray_next - ray_start);

			//ensure that it isn't a degenerate case
			ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
			ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
			ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

			// computer time when entry and exit volume
			float time_start_volume = getMinTime(ray_start, ray_dir);
			float time_exit_volume = getMaxTime(ray_start, ray_dir);

			const float min_dist = 0.f;         //in meters
			time_start_volume = fmax(time_start_volume, min_dist);
			if (time_start_volume >= time_exit_volume)
				return;

			float time_curr = time_start_volume;
			int3 g = getVoxel(ray_start + ray_dir * time_curr);
			g.x = max(0, min(g.x, VOLUME_X - 1));
			g.y = max(0, min(g.y, VOLUME_Y - 1));
			g.z = max(0, min(g.z, VOLUME_Z - 1));

			float tsdf = readTsdf(volume, g.x, g.y, g.z);
			//infinite loop guard
			const float max_time = 3 * (VOLUME_SIZE_X + VOLUME_SIZE_Y + VOLUME_SIZE_Z);

			float time_step = 0.8*tranc_dist;
			for (; time_curr < max_time; time_curr += time_step)
			{
				float tsdf_prev = tsdf;

				int3 g = getVoxel(ray_start + ray_dir * (time_curr + time_step));
				if (!checkInds(g))
					break;

				tsdf = readTsdf(volume, g.x, g.y, g.z);

				if (tsdf_prev < 0.f && tsdf > 0.f)
					break;
				
				if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
				{
					float Ftdt = interpolateTrilineary(volume, ray_start, ray_dir, time_curr + time_step);
					if (isnan(Ftdt))
						break;

					float Ft = interpolateTrilineary(volume, ray_start, ray_dir, time_curr);
					if (isnan(Ft))
						break;

					float Ts = time_curr - time_step * Ft / (Ftdt - Ft);

					float3 vetex_found = ray_start + ray_dir * Ts;

					vmap[y*cols + x].x = vetex_found.x;
					vmap[y*cols + x].y = vetex_found.y;
					vmap[y*cols + x].z = vetex_found.z;

				}
				
			}          
		}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
raycast(const short2* volume, float3* vmap, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R_, float3 t_, float tranc_dist)
{

	dim3 block(RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

	rayCastKernel << <grid, block >> >(volume, vmap, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy,
		R_, t_, tranc_dist);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

