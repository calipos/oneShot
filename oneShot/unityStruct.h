#ifndef _UNITY_STRUCT_H_
#define _UNITY_STRUCT_H_

#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
//#include "math_functions.h"
#include "device_functions.h"
#define SAMPLE_H (3)
#define SAMPLE_W (3)

int initUnityData(const int h,const int w);
int sampleUnityData(const float*vmap,
	const float*nmap,
	const unsigned char*rgb,
	double4 intr,
	const int h, const int w, const int ori_w);
int device2Host(float*host_points, float*host_norms, float*host_rgba);
#endif // !_UNITY_STRUCT_H_
