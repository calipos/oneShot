#include"unreGpu.h"
#include"filters.h"

template <class T>
__device__ __host__ __forceinline__ void swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}

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



__device__ __host__ __forceinline__ float3
cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}
__device__ __forceinline__ void
computeRoots2(const float& b, const float& c, float3& roots)
{
	roots.x = 0.f;
	float d = b * b - 4.f * c;
	if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}

__device__ __forceinline__ void
computeRoots3(float c0, float c1, float c2, float3& roots)
{
	if (fabsf(c0) < numeric_limits<float>::epsilon())// one root is 0 -> quadratic equation
	{
		computeRoots2(c2, c1, roots);
	}
	else
	{
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// Construct the parameters used in classifying the roots of the equation
		// and in solving the equation for the roots in closed form.
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;

		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// Compute the eigenvalues by solving for the roots of the polynomial.
		float rho = sqrtf(-a_over_3);
		float theta = atan2f(sqrtf(-q), half_b)*s_inv3;
		float cos_theta = __cosf(theta);
		float sin_theta = __sinf(theta);
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// Sort in increasing order.
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z)
		{
			swap(roots.y, roots.z);

			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}
		if (roots.x <= 0) // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
			computeRoots2(c2, c1, roots);
	}
}



struct Eigen33
{
public:
	template<int Rows>
	struct MiniMat
	{
		float3 data[Rows];
		__device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
		__device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
	};
	typedef MiniMat<3> Mat33;
	typedef MiniMat<4> Mat43;


	static __forceinline__ __device__ float3
		unitOrthogonal(const float3& src)
	{
		float3 perp;
		/* Let us compute the crossed product of *this with a vector
		* that is not too close to being colinear to *this.
		*/

		/* unless the x and y coords are both close to zero, we can
		* simply take ( -y, x, 0 ) and normalize it.
		*/
		if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
		{
			float invnm = rsqrtf(src.x*src.x + src.y*src.y);
			perp.x = -src.y * invnm;
			perp.y = src.x * invnm;
			perp.z = 0.0f;
		}
		/* if both x and y are close to zero, then the vector is close
		* to the z-axis, so it's far from colinear to the x-axis for instance.
		* So we take the crossed product with (1,0,0) and normalize it.
		*/
		else
		{
			float invnm = rsqrtf(src.z * src.z + src.y * src.y);
			perp.x = 0.0f;
			perp.y = -src.z * invnm;
			perp.z = src.y * invnm;
		}

		return perp;
	}

	__device__ __forceinline__
		Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}
	__device__ __forceinline__ void
		compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
	{
		// Scale the matrix so its entries are in [-1,1].  The scaling is applied
		// only when at least one matrix entry has magnitude larger than 1.

		float max01 = fmaxf(fabsf(mat_pkg[0]), fabsf(mat_pkg[1]));
		float max23 = fmaxf(fabsf(mat_pkg[2]), fabsf(mat_pkg[3]));
		float max45 = fmaxf(fabsf(mat_pkg[4]), fabsf(mat_pkg[5]));
		float m0123 = fmaxf(max01, max23);
		float scale = fmaxf(max45, m0123);

		if (scale <= numeric_limits<float>::min())
			scale = 1.f;

		mat_pkg[0] /= scale;
		mat_pkg[1] /= scale;
		mat_pkg[2] /= scale;
		mat_pkg[3] /= scale;
		mat_pkg[4] /= scale;
		mat_pkg[5] /= scale;

		// The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
		// eigenvalues are the roots to this equation, all guaranteed to be
		// real-valued, because the matrix is symmetric.
		float c0 = m00() * m11() * m22()
			+ 2.f * m01() * m02() * m12()
			- m00() * m12() * m12()
			- m11() * m02() * m02()
			- m22() * m01() * m01();
		float c1 = m00() * m11() -
			m01() * m01() +
			m00() * m22() -
			m02() * m02() +
			m11() * m22() -
			m12() * m12();
		float c2 = m00() + m11() + m22();

		computeRoots3(c0, c1, c2, evals);

		if (evals.z - evals.x <= numeric_limits<float>::epsilon())
		{
			evecs[0] = make_float3(1.f, 0.f, 0.f);
			evecs[1] = make_float3(0.f, 1.f, 0.f);
			evecs[2] = make_float3(0.f, 0.f, 1.f);
		}
		else if (evals.y - evals.x <= numeric_limits<float>::epsilon())
		{
			// first and second equal                
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[2]);
			evecs[0] = cross(evecs[1], evecs[2]);
		}
		else if (evals.z - evals.y <= numeric_limits<float>::epsilon())
		{
			// second and third equal                                    
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[0]);
			evecs[2] = cross(evecs[0], evecs[1]);
		}
		else
		{

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			float mmax[3];

			unsigned int min_el = 2;
			unsigned int max_el = 2;
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[2] = len1;
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[2] = len2;
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				mmax[2] = len3;
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot(vec_tmp[0], vec_tmp[0]);
			len2 = dot(vec_tmp[1], vec_tmp[1]);
			len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				mmax[1] = len1;
				evecs[1] = vec_tmp[0] * rsqrtf(len1);
				min_el = len1 <= mmax[min_el] ? 1 : min_el;
				max_el = len1 > mmax[max_el] ? 1 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[1] = len2;
				evecs[1] = vec_tmp[1] * rsqrtf(len2);
				min_el = len2 <= mmax[min_el] ? 1 : min_el;
				max_el = len2 > mmax[max_el] ? 1 : max_el;
			}
			else
			{
				mmax[1] = len3;
				evecs[1] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 1 : min_el;
				max_el = len3 > mmax[max_el] ? 1 : max_el;
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot(vec_tmp[0], vec_tmp[0]);
			len2 = dot(vec_tmp[1], vec_tmp[1]);
			len3 = dot(vec_tmp[2], vec_tmp[2]);


			if (len1 >= len2 && len1 >= len3)
			{
				mmax[0] = len1;
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[0] = len2;
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else
			{
				mmax[0] = len3;
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}

			unsigned mid_el = 3 - min_el - max_el;
			evecs[min_el] = normalized(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
			evecs[mid_el] = normalized(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));
		}
		// Rescale back to the original size.
		evals = evals* scale;
	}
private:
	volatile float* mat_pkg;

	__device__  __forceinline__ float m00() const { return mat_pkg[0]; }
	__device__  __forceinline__ float m01() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m02() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m10() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m11() const { return mat_pkg[3]; }
	__device__  __forceinline__ float m12() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m20() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m21() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m22() const { return mat_pkg[5]; }

	__device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
	__device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
	__device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

	__device__  __forceinline__ static bool isMuchSmallerThan(float x, float y)
	{
		// copied from <eigen>/include/Eigen/src/Core/NumTraits.h
		const float prec_sqr = numeric_limits<float>::epsilon() * numeric_limits<float>::epsilon();
		return x * x <= prec_sqr * y * y;
	}
};


template<typename Dtype>
struct NmapConfig
{
	enum
	{
		kx = 19,
		ky = 19,
		STEP = 1,
	};
	static const int naiveScopeK = 10;
#define NmapConfig_Restriction (0.01)
};
//template<typename Dtype> const float  NmapConfig<Dtype>::restriction= 5. ;提示该变量没有定义在device

template <typename Dtype>
__global__ void computeVmapKernel(const float* depth, float* vmap, Dtype fx_inv, Dtype fy_inv, Dtype cx, Dtype cy, const int rows, const int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	int pos_ = v*cols + u;
	int plane_cnt = cols*rows;
	if (u < cols && v < rows)
	{
		float z = depth[pos_];;//already meter / 1000.f; // load and convert: mm -> meters

		if (z >0.01)
		{
			float vx = z * (u - cx) * fx_inv;
			float vy = z * (v - cy) * fy_inv;
			float vz = z;
			
			vx = static_cast<float>((int)(vx * 1000)) / 1000;
			vy = static_cast<float>((int)(vy * 1000)) / 1000;

			//vmap[pos_] = vx;
			//vmap[pos_ + plane_cnt] = vy;
			//vmap[pos_ + 2 * plane_cnt] = vz;
			vmap[3*pos_] = vx;
			vmap[3*pos_ + 1] = vy;
			vmap[3*pos_ + 2] = vz;
		}
		else
		{
			vmap[3*pos_] = numeric_limits<float>::quiet_NaN();
			vmap[3*pos_+1] = 0;
			vmap[3*pos_+2] = 0;
		}

	}
}


template<>
int createVMap<double>(const float*dataIn, float*dataOut, const double fx, const double fy, const double cx, const double cy, const int rows, const int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	computeVmapKernel<double> << <grid, block >> >(dataIn, dataOut, 1. / fx, 1. / fy, cx, cy, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}


template <typename Dtype>
__global__ void	computeNmapNaiveKernel(const Dtype*vmap, Dtype*nmap, int rows, int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= cols - NmapConfig<float>::naiveScopeK || v >= rows - NmapConfig<float>::naiveScopeK
		|| u < NmapConfig<float>::naiveScopeK || v < NmapConfig<float>::naiveScopeK)
		return;
	int plane_cnt = cols*rows;
	int pos_ = cols*v + u;

	int u0 = u - NmapConfig<float>::naiveScopeK;
	int v0 = v - NmapConfig<float>::naiveScopeK;
	int u1 = u + NmapConfig<float>::naiveScopeK;
	int v1 = v + NmapConfig<float>::naiveScopeK;
	if (isnan(vmap[3 * (cols*v0 + u0)])|| isnan(vmap[3 * (cols*v1 + u1)]))
		return;
	
	float3 diff;
	diff.x = vmap[3 * (cols*v0 + u0)] - vmap[3 * (cols*v1 + u1)];
	diff.y = vmap[3 * (cols*v0 + u0)+1] - vmap[3 * (cols*v1 + u1)+1];
	diff.z = -vmap[3 * (cols*v0 + u0)+2] + vmap[3 * (cols*v1 + u1)+2];
	float3 n = normalized(diff);
	
	nmap[3 * pos_] = n.x;
	nmap[3 * pos_ + 1] = n.y;
	nmap[3 * pos_ + 2] = n.z;
}


template <typename Dtype>
__global__ void	computeNmapRestrictedKernel(const Dtype*vmap, Dtype*nmap, int rows, int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= cols || v >= rows)
		return;
	int plane_cnt = cols*rows;
	int pos_ = cols*v + u;

	if (isnan(vmap[3 * pos_]))
		return;
	int ty = min(v - NmapConfig<Dtype>::ky / 2 + NmapConfig<Dtype>::ky, rows - 1);
	int tx = min(u - NmapConfig<Dtype>::kx / 2 + NmapConfig<Dtype>::kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	Dtype anchor = vmap[3 * pos_ + 2];
	for (int cy = max(v - NmapConfig<Dtype>::ky / 2, 0); cy < ty; cy += NmapConfig<Dtype>::STEP)
		for (int cx = max(u - NmapConfig<Dtype>::kx / 2, 0); cx < tx; cx += NmapConfig<Dtype>::STEP)
		{
			int pos_2 = cols*cy + cx;
			float v_x = vmap[3 * pos_2];
			if (!isnan(v_x) && abs(anchor -vmap[3 * pos_2 + 2]) <= NmapConfig_Restriction)
			{
				centroid.x += v_x;
				centroid.y += vmap[3 * pos_2 + 1];
				centroid.z += vmap[3 * pos_2 + 2];
				++counter;
			}
		}

	if (counter < NmapConfig<Dtype>::kx * NmapConfig<Dtype>::ky * 3 / 4)
		return;

	centroid = centroid*(1.f / counter);

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - NmapConfig<Dtype>::ky / 2, 0); cy < ty; cy += NmapConfig<Dtype>::STEP)
		for (int cx = max(u - NmapConfig<Dtype>::kx / 2, 0); cx < tx; cx += NmapConfig<Dtype>::STEP)
		{
			int pos_3 = cols*cy + cx;
			float3 v;
			v.x = vmap[3 * pos_3];
			if (isnan(v.x) || abs(anchor - vmap[3 * pos_3 + 2]) > NmapConfig_Restriction)
				continue;
			v.y = vmap[3 * pos_3 + 1];
			v.z = vmap[3 * pos_3 + 2];

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}

	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized(evecs[0]);

	u = threadIdx.x + blockIdx.x * blockDim.x;
	v = threadIdx.y + blockIdx.y * blockDim.y;

	nmap[3 * pos_] = n.z > 0 ? n.x : -n.x;
	nmap[3 * pos_ + 1] = n.z > 0 ? n.y : -n.y;
	float nz2 = n.z > 0 ? -n.z : n.z;
	nmap[3 * pos_ + 2] = nz2;
}


template <typename Dtype>
__global__ void	computeNmapKernelEigen(const Dtype*vmap, Dtype*nmap, int rows, int cols)
{
	int u = threadIdx.x + blockIdx.x * blockDim.x;
	int v = threadIdx.y + blockIdx.y * blockDim.y;
	if (u >= cols || v >= rows)
		return;
	int plane_cnt = cols*rows;
	int pos_ = cols*v + u;
	
	if (isnan(vmap[3*pos_]))
		return;
	int ty = min(v - NmapConfig<Dtype>::ky / 2 + NmapConfig<Dtype>::ky, rows - 1);
	int tx = min(u - NmapConfig<Dtype>::kx / 2 + NmapConfig<Dtype>::kx, cols - 1);

	float3 centroid = make_float3(0.f, 0.f, 0.f);
	int counter = 0;
	
	for (int cy = max(v - NmapConfig<Dtype>::ky / 2, 0); cy < ty; cy += NmapConfig<Dtype>::STEP)
		for (int cx = max(u - NmapConfig<Dtype>::kx / 2, 0); cx < tx; cx += NmapConfig<Dtype>::STEP)
		{
			int pos_2 = cols*cy + cx;
			float v_x = vmap[3*pos_2];
			if (!isnan(v_x))
			{
				centroid.x += v_x;
				centroid.y += vmap[3*pos_2 + 1];
				centroid.z += vmap[3*pos_2 + 2];
				++counter;
			}
		}

	if (counter < NmapConfig<Dtype>::kx * NmapConfig<Dtype>::ky * 3 / 4)
		return;

	centroid = centroid*( 1.f / counter);

	float cov[] = { 0, 0, 0, 0, 0, 0 };

	for (int cy = max(v - NmapConfig<Dtype>::ky / 2, 0); cy < ty; cy += NmapConfig<Dtype>::STEP)
		for (int cx = max(u - NmapConfig<Dtype>::kx / 2, 0); cx < tx; cx += NmapConfig<Dtype>::STEP)
		{
			int pos_3 = cols*cy + cx;
			float3 v;
			v.x = vmap[3*pos_3];
			if (isnan(v.x))
				continue;

			v.y = vmap[3*pos_3 + 1];
			v.z = vmap[3*pos_3 + 2];

			float3 d = v - centroid;

			cov[0] += d.x * d.x;               //cov (0, 0)
			cov[1] += d.x * d.y;               //cov (0, 1)
			cov[2] += d.x * d.z;               //cov (0, 2)
			cov[3] += d.y * d.y;               //cov (1, 1)
			cov[4] += d.y * d.z;               //cov (1, 2)
			cov[5] += d.z * d.z;               //cov (2, 2)
		}
	
	typedef Eigen33::Mat33 Mat33;
	Eigen33 eigen33(cov);

	Mat33 tmp;
	Mat33 vec_tmp;
	Mat33 evecs;
	float3 evals;
	eigen33.compute(tmp, vec_tmp, evecs, evals);

	float3 n = normalized(evecs[0]);

	u = threadIdx.x + blockIdx.x * blockDim.x;
	v = threadIdx.y + blockIdx.y * blockDim.y;
	
	nmap[3*pos_] = n.z > 0 ? n.x: -n.x;
	nmap[3*pos_ + 1] = n.z > 0 ? n.y: -n.y;
	float nz2 = n.z > 0 ? n.z : -n.z;


	//float nx2 = static_cast<int>(n.x * 20 + 0.5)*0.05;
	//float ny2 = static_cast<int>(n.y * 20 + 0.5)*0.05;	
	//float nz2 = static_cast<int>(nz * 20 + 0.5)*0.05;
	//nmap[3 * pos_] = nx2;
	//nmap[3 * pos_ + 1] = ny2;
	nmap[3 * pos_ + 2] = nz2;
}

template<>//计算vmap，vmap必须提前申请空间，其中vmap和namp的高是depthImage高的3倍：CHW的关系
int computeNormalsEigen<float>(const float*vmap, float*nmap, float*nmap_average, int rows, int cols)
{
	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);
	cudaMemset(nmap, 0, 3* rows* cols * sizeof(float));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	//computeNmapKernelEigen<float> << <grid, block >> > (vmap, nmap, rows, cols);
	//computeNmapNaiveKernel<float> << <grid, block >> > (vmap, nmap, rows, cols);
	computeNmapRestrictedKernel<float> << <grid, block >> > (vmap, nmap, rows, cols);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	//averageFilter_3c<float> << <grid, block >> >(nmap, nmap_average, cols, rows, 5);
	//cudaSafeCall(cudaGetLastError());
	//cudaSafeCall(cudaDeviceSynchronize());

	return 0;
}


template <typename Dtype>
__global__ void	tranformMapsKernel(const int rows, const int cols, const int plane_cnt,
	const Dtype* vmap_src, const Dtype* nmap_src,
	const Mat33 Rmat, const float3 tvec,
	Dtype* vmap_dst, Dtype* nmap_dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float qnan = numeric_limits<float>::quiet_NaN();
	if (x < cols && y < rows)
	{
		int pos_ = y*cols + x;
		//vertices
		float3 vsrc, vdst = make_float3(
			numeric_limits<float>::quiet_NaN(), 
			numeric_limits<float>::quiet_NaN(), 
			numeric_limits<float>::quiet_NaN());
		vsrc.x = vmap_src[pos_];

		if (!isnan(vsrc.x))
		{

			vsrc.y = vmap_src[pos_ + plane_cnt];
			vsrc.z = vmap_src[pos_ + 2 * plane_cnt];

			vdst = Rmat * vsrc + tvec;

			vmap_dst[pos_ + plane_cnt] = vdst.y;
			vmap_dst[pos_ + 2 * plane_cnt] = vdst.z;
		}

		vmap_dst[pos_] = vdst.x;

		//normals
		float3 nsrc, ndst = make_float3(qnan, qnan, qnan);
		nsrc.x = nmap_src[pos_];

		if (!isnan(nsrc.x))
		{
			nsrc.y = nmap_src[pos_ + plane_cnt];
			nsrc.z = nmap_src[pos_ + 2 * plane_cnt];

			ndst = Rmat * nsrc;

			nmap_dst[pos_ + plane_cnt] = ndst.y;
			nmap_dst[pos_ + 2 * plane_cnt] = ndst.z;
		}

		nmap_dst[pos_] = ndst.x;
	}
}

template<>//计算vmap_dst，nmap_dst必须提前申请空间
int tranformMaps<float>(const float* vmap_src, const float* nmap_src, const float*Rmat_, const float*tvec_, float* vmap_dst, float* nmap_dst, const int& rows, const int& cols)
{

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	float3 device_t = make_float3(tvec_[0], tvec_[1], tvec_[2]);
	Mat33 device_R_inv(make_float3(Rmat_[0], Rmat_[1], Rmat_[2]), make_float3(Rmat_[3], Rmat_[4], Rmat_[5]), make_float3(Rmat_[6], Rmat_[7], Rmat_[8]));
	int plane_cnt = rows*cols;
	tranformMapsKernel << <grid, block >> >(rows, cols, plane_cnt,
		vmap_src, nmap_src,
		device_R_inv, device_t,
		vmap_dst, nmap_dst);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}

template <typename Dtype>
__global__ void	tranformVMapsKernel(const int rows, const int cols, const int plane_cnt,
	const Dtype* vmap_src,
	const Mat33 Rmat, const float3 tvec,
	Dtype* vmap_dst)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	const float qnan = numeric_limits<float>::quiet_NaN();
	if (x < cols && y < rows)
	{
		int pos_ = y*cols + x;
		//vertices
		float3 vsrc, vdst = make_float3(qnan, qnan, qnan);
		vsrc.x = vmap_src[pos_];
		if (!isnan(vsrc.x))
		{
			vsrc.y = vmap_src[pos_ + plane_cnt];
			vsrc.z = vmap_src[pos_ + 2 * plane_cnt];
			vdst = Rmat * vsrc + tvec;
			vmap_dst[pos_ + plane_cnt] = vdst.y;
			vmap_dst[pos_ + 2 * plane_cnt] = vdst.z;
		}
		vmap_dst[pos_] = vdst.x;
	}
}


template<>//计算vmap_dst，nmap_dst必须提前申请空间
int tranformMaps<float>(const float* vmap_src, const float*Rmat_, const float*tvec_, float* vmap_dst, const int& rows, const int& cols)
{

	dim3 block(32, 8);
	dim3 grid(1, 1, 1);
	grid.x = divUp(cols, block.x);
	grid.y = divUp(rows, block.y);

	float3 device_t = make_float3(tvec_[0], tvec_[1], tvec_[2]);
	Mat33 device_R_inv(make_float3(Rmat_[0], Rmat_[1], Rmat_[2]), make_float3(Rmat_[3], Rmat_[4], Rmat_[5]), make_float3(Rmat_[6], Rmat_[7], Rmat_[8]));
	int plane_cnt = rows*cols;
	tranformVMapsKernel << <grid, block >> >(rows, cols, plane_cnt, vmap_src, device_R_inv, device_t, vmap_dst);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}


