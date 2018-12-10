#include"unreGpu.h"

__device__ __forceinline__ float3
normalized(const float3& v)
{
	return v * rsqrt(dot(v, v));
}

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



struct RayCaster
{
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };
};

__device__ __forceinline__ float
	getMinTime(const float3& origin, const float3& dir)
{
	float txmin = 0.0;
	if (origin.x<0 && dir.x > 0)
	{
		txmin = (- origin.x) / dir.x;
	}
	else if (origin.x>0 && dir.x < 0)
	{
		txmin = (VOLUME_SIZE_X-origin.x) / dir.x;
	}
	float tymin = 0.0;
	if (origin.y<0 && dir.y > 0)
	{
		tymin = ( - origin.y) / dir.y;
	}
	else if (origin.y>0 && dir.y < 0)
	{
		tymin = (VOLUME_SIZE_X -origin.y) / dir.y;
	}
	float tzmin = 0.0;
	if (origin.z<0 && dir.z > 0)
	{
		tzmin = (VOLUME_SIZE_Z - origin.z) / dir.z; 
	}
	else if (origin.z>0 && dir.z < 0)
	{
		tzmin = -origin.z / dir.z;
	}
	return fmax(fmax(txmin, tymin), tzmin);
}

__device__ __forceinline__ float
	getMaxTime(const float3& origin, const float3& dir)
{
	float txmax = 0.0;
	if (origin.x<0 && dir.x > 0)
	{
		txmax = (VOLUME_SIZE_X -origin.x) / dir.x;
	}
	else if (origin.x>0 && dir.x < 0)
	{
		txmax = - origin.x / dir.x;
	}
	float tymax = 0.0;
	if (origin.y<0 && dir.y > 0)
	{
		tymax = (VOLUME_SIZE_Y -origin.y) / dir.y;
	}
	else if (origin.y>0 && dir.y < 0)
	{
		tymax =  - origin.y / dir.y;
	}
	float tzmax = 0.0;
	if (origin.z<0 && dir.z > 0)
	{
		tzmax = -origin.z / dir.z; 
	}
	else if (origin.z>0 && dir.z < 0)
	{
		tzmax = -(VOLUME_SIZE_Z + origin.z) / dir.z;
	}
	return fmin(fmin(txmax, tymax), tzmax);
}

__device__ __forceinline__ float3
	get_ray_next(int x, int y,
		const float intr_cx, const float intr_cy, 
		const float intr_fx, const float intr_fy)
{
	float3 ray_next;
	ray_next.x = 0.5*(x - intr_cx) / intr_fx;
	ray_next.y = 0.5*(y - intr_cy) / intr_fy;
	ray_next.z = 0.5;
	return ray_next;
}

__device__ __forceinline__ float
	readTsdf(const short2* volume,int x, int y, int z)
{
	return static_cast<float>(volume[z*VOLUME_X*VOLUME_Y + y*VOLUME_X + x].x*1.f) / DIVISOR;
}
__device__ __forceinline__ float
	readTsdfWeight(const short2* volume, int x, int y, int z)
{
	return static_cast<float>(volume[z*VOLUME_X*VOLUME_Y + y*VOLUME_X + x].y*1.f) / DIVISOR;
}

__device__ __forceinline__ int3
	getVoxel(float3 point, float3 cell_size)
{
	int vx = __float2int_rd(point.x / cell_size.x);        // round to negative infinity
	int vy = __float2int_rd(point.y / cell_size.y);
	int vz = __float2int_rd(point.z / cell_size.z);
	return make_int3(vx, vy, -vz);
}

__device__ __forceinline__ bool
	checkInds(const int3& g)
{
	return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_Y && g.z < VOLUME_Z);
}

		

__device__ __forceinline__ float
	interpolateTrilineary(const short2* volume,const float3& point, float3 cell_size)
{
	int3 g = getVoxel(point, cell_size);

	if (g.x <= 0 || g.x >= VOLUME_X - 1)
		return numeric_limits<float>::quiet_NaN();

	if (g.y <= 0 || g.y >= VOLUME_Y - 1)
		return numeric_limits<float>::quiet_NaN();

	if (g.z <= 0 || g.z >= VOLUME_Z - 1)
		return numeric_limits<float>::quiet_NaN();

	float vx = (g.x + 0.5f) * (cell_size.x);
	float vy = (g.y + 0.5f) * (cell_size.y);
	float vz = -(g.z + 0.5f) * (cell_size.z);

	g.x = (point.x < vx) ? (g.x - 1) : g.x;
	g.y = (point.y < vy) ? (g.y - 1) : g.y;
	g.z = (point.z > vz) ? (g.z - 1) : g.z;

	float a = (point.x - (g.x + 0.5f) * cell_size.x) / cell_size.x;
	float b = (point.y - (g.y + 0.5f) * cell_size.y) / cell_size.y;
	float c = (-point.z - (g.z + 0.5f) * cell_size.z) / cell_size.z;
			
	float res = 
		readTsdf(volume, g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
		readTsdf(volume, g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
		readTsdf(volume, g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b       * (1 - c) +
		readTsdf(volume, g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b       * c +
		readTsdf(volume, g.x + 1, g.y + 0, g.z + 0) * a       * (1 - b) * (1 - c) +
		readTsdf(volume, g.x + 1, g.y + 0, g.z + 1) * a       * (1 - b) * c +
		readTsdf(volume, g.x + 1, g.y + 1, g.z + 0) * a       * b       * (1 - c) +
		readTsdf(volume, g.x + 1, g.y + 1, g.z + 1) * a       * b       * c;
	return res;
}

__device__ __forceinline__ float
	interpolateTrilineary(const short2* volume, const float3& origin, const float3& dir, float time, float3 cell_size)
{
	return interpolateTrilineary(volume, origin + dir * time, cell_size);
}

////用光线追踪的想法来扫描体素得到点云，此处的点云的角度并不是large-slace，所以t_必须在体素外！！
__global__  void
	rayCastPointKernel(const short2* volume, float3* vmap, int rows, int cols,
		const float intr_cx, const float intr_cy, 
		const float intr_fx, const float intr_fy,
		const Mat33 R_inv, const float3 t_, const float tranc_dist, float3 cell_size)
{
	int x = threadIdx.x + blockIdx.x * RayCaster::CTA_SIZE_X;
	int y = threadIdx.y + blockIdx.y * RayCaster::CTA_SIZE_Y;
			
	if (x >= cols || y >= rows)
		return;		
			
	float3 ray_start = t_;
	//float3 ray_next = R_ * get_ray_next(x, y, intr_cx, intr_cy, intr_fx, intr_fy) + t_;
	float3 ray_next = R_inv * (get_ray_next(x, y, intr_cx, intr_cy, intr_fx, intr_fy) - t_);
	//if (ray_next.z > ray_start.z&&ray_start.z>0)
	//{
	//	ray_next = ray_next*-1.;		
	//}
	//if (ray_next.z < ray_start.z&&ray_start.z<0)
	//{
	//	ray_next = ray_next*-1.;
	//}

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
	int3 g = getVoxel(ray_start + ray_dir * time_curr, cell_size);
	g.x = max(0, min(g.x, VOLUME_X - 1));
	g.y = max(0, min(g.y, VOLUME_Y - 1));
	g.z = max(0, min(g.z, VOLUME_Z - 1));

	float tsdf = readTsdf(volume, g.x, g.y, g.z);
	//infinite loop guard
	const float max_time = 3 * (VOLUME_SIZE_X + VOLUME_SIZE_Y + VOLUME_SIZE_Z);

	float time_step = 0.8*tranc_dist;
	vmap[y*cols + x].x = numeric_limits<float>::quiet_NaN();
	for (; time_curr < max_time; time_curr += time_step)
	{
		
		float tsdf_prev = tsdf;

		int3 g = getVoxel(ray_start + ray_dir * (time_curr + time_step),cell_size);
		
		if (!checkInds(g))
			break;
		
		tsdf = readTsdf(volume, g.x, g.y, g.z);
		
		if (tsdf_prev < 0.f && tsdf > 0.f)
			break;
		
		if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
		{
			//vmap[y*cols + x].x = g.x;
			//vmap[y*cols + x].y = g.y;
			//vmap[y*cols + x].z = g.z;
			//break;
			//float3 vetex_found_ = ray_start + ray_dir * time_curr;
			//vmap[y*cols + x].x = vetex_found_.x;
			//vmap[y*cols + x].y = vetex_found_.y;
			//vmap[y*cols + x].z = vetex_found_.z;
			//break;//用这种直接返回的没有那么多的噪点，但是用下面的就会有很多噪点！！！

			float Ftdt = interpolateTrilineary(volume, ray_start, ray_dir, time_curr + time_step, cell_size);					
			if (isnan(Ftdt))
				break;
			float Ft = interpolateTrilineary(volume, ray_start, ray_dir, time_curr, cell_size);
			if (isnan(Ftdt))
				break;

			float Ts = time_curr - time_step * Ft / (Ftdt - Ft);

			float3 vetex_found = ray_start + ray_dir * Ts;
					
			vmap[y*cols + x].x = vetex_found.x;
			vmap[y*cols + x].y = vetex_found.y;
			vmap[y*cols + x].z = vetex_found.z;
			break;
		}				
	}          
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////用光线追踪的想法来扫描体素得到点云，此处的点云的角度并不是large-slace，所以t_必须在体素外！！
void
raycastPoint(const short2* volume, float3* vmap, int rows, int cols,
	float intr_cx, float intr_cy, float intr_fx, float intr_fy,
	Mat33 R_inv, float3 t_, float tranc_dist)
{

	dim3 block(RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
	dim3 grid(divUp(cols, block.x), divUp(rows, block.y));

	float3 cell_size;
	cell_size.x = 1.*VOLUME_SIZE_X / VOLUME_X;
	cell_size.y = 1.*VOLUME_SIZE_Y / VOLUME_Y;
	cell_size.z = 1.*VOLUME_SIZE_Z / VOLUME_Z;

	rayCastPointKernel << <grid, block >> >(volume, vmap, rows, cols,
		intr_cx, intr_cy, intr_fx, intr_fy,
		R_inv, t_, tranc_dist, cell_size);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

