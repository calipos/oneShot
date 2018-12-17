#ifdef COMPILE_TEST
#include "opencv2/opencv.hpp"

struct float3
{
	float x;
	float y;
	float z;
};
void invert3x3(const float3 * src, float3 * dst)
{
	float det;
	dst[0].x = +src[1].y * src[2].z - src[1].z * src[2].y;
	dst[0].y = -src[0].y * src[2].z + src[0].z * src[2].y;
	dst[0].z = +src[0].y * src[1].z - src[0].z * src[1].y;
	dst[1].x = -src[1].x * src[2].z + src[1].z * src[2].x;
	dst[1].y = +src[0].x * src[2].z - src[0].z * src[2].x;
	dst[1].z = -src[0].x * src[1].z + src[0].z * src[1].x;
	dst[2].x = +src[1].x * src[2].y - src[1].y * src[2].x;
	dst[2].y = -src[0].x * src[2].y + src[0].y * src[2].x;
	dst[2].z = +src[0].x * src[1].y - src[0].y * src[1].x;
	det = src[0].x * dst[0].x + src[0].y * dst[1].x + src[0].z * dst[2].x;
	det = 1.0f / det;
	dst[0].x *= det;
	dst[0].y *= det;
	dst[0].z *= det;
	dst[1].x *= det;
	dst[1].y *= det;
	dst[1].z *= det;
	dst[2].x *= det;
	dst[2].y *= det;
	dst[2].z *= det;
}
inline void invert3x3(const double * src, double * dst)
{
	float det;

	/* Compute adjoint: */

	dst[0] = +src[4] * src[8] - src[5] * src[7];
	dst[1] = -src[1] * src[8] + src[2] * src[7];
	dst[2] = +src[1] * src[5] - src[2] * src[4];
	dst[3] = -src[3] * src[8] + src[5] * src[6];
	dst[4] = +src[0] * src[8] - src[2] * src[6];
	dst[5] = -src[0] * src[5] + src[2] * src[3];
	dst[6] = +src[3] * src[7] - src[4] * src[6];
	dst[7] = -src[0] * src[7] + src[1] * src[6];
	dst[8] = +src[0] * src[4] - src[1] * src[3];

	/* Compute determinant: */

	det = src[0] * dst[0] + src[1] * dst[3] + src[2] * dst[6];

	/* Multiply adjoint with reciprocal of determinant: */

	det = 1.0f / det;

	dst[0] *= det;
	dst[1] *= det;
	dst[2] *= det;
	dst[3] *= det;
	dst[4] *= det;
	dst[5] *= det;
	dst[6] *= det;
	dst[7] *= det;
	dst[8] *= det;
}
int TEST_inverse()
{
	cv::Mat a = cv::Mat(3, 3, CV_64FC1);
	{
		a.ptr<double>(0)[0] = 0.998972376591411;
		a.ptr<double>(0)[1] = -0.01070089643556875;
		a.ptr<double>(0)[2] = -0.044041816751621;
		a.ptr<double>(1)[0] = 0.01616702812429442;
		a.ptr<double>(1)[1] = 0.9919373078916188;
		a.ptr<double>(1)[2] = 0.12569409061032314;
		a.ptr<double>(2)[0] = 0.042341681697074879;
		a.ptr<double>(2)[1] = -0.1262769497105591;
		a.ptr<double>(2)[2] = 0.9910909715878056;
	}
	cv::Mat b = a.inv();
	float3 a1, a2, a3;
	{
		a1.x = 1;
		a1.y = 22;
		a1.z = 33;
		a2.x = 14;
		a2.y = 25;
		a2.z = 36;
		a3.x = 17;
		a3.y = 28;
		a3.z = 39;
	}
	float3 b1, b2, b3,b33[3];
	float3 input[3];
	input[0] = a1;
	input[1] = a2;
	input[2] = a3;
	//invert3x3(input,b33);
	double A[9] = { 0.998972376591411, -0.01070089643556875, -0.044041816751621, 0.01616702812429442, 0.9919373078916188, 0.12569409061032314, 0.042341681697074879, -0.1262769497105591, 0.9910909715878056 };
	double B[9];
	invert3x3(A, B);
	return 0;
}

#endif