
#include<time.h>
#include "unreGpu.h"

__device__ __forceinline__ float length(float2 v)
{
	return sqrtf(dot(v, v));
}
__device__ __forceinline__ float length(float3 v)
{
	return sqrtf(dot(v, v));
}
__device__ __forceinline__ float length(float4 v)
{
	return sqrtf(dot(v, v));
}

template<typename Dtype>
struct CollisionParam
{
#define AVAILABLESPACE_X_START  (-.5)
#define AVAILABLESPACE_X_END  (.5)
#define AVAILABLESPACE_Y_START  (-.5)
#define AVAILABLESPACE_Y_END  (.5)
#define AVAILABLESPACE_Z_START  (.2)
#define AVAILABLESPACE_Z_END  (1.2)

#define EMITTER_Z  AVAILABLESPACE_Z_START;
#define EMITTER_X_START  (-0.25)
#define EMITTER_X_END  (0.25)
#define EMITTER_Y_START  (-0.25)
#define EMITTER_Y_END  (0.25)

#define BALL_RADIUS  (0.02)
#define BALL_DIAMETER  (0.04)//2* BALL_RADIUS;
#define BALL_MAX_INIT_VEL  (0.05)
	static const int BALL_NUMBER_IN_ROW = int((-EMITTER_X_START - BALL_RADIUS) / BALL_DIAMETER)
		+ int((EMITTER_X_END - BALL_RADIUS) / BALL_DIAMETER) + 1;
	static const int BALL_NUMBER_IN_COL = (int((-EMITTER_Y_START - BALL_RADIUS) / BALL_DIAMETER)
		+ int((EMITTER_Y_END - BALL_RADIUS) / BALL_DIAMETER) + 1);
	static const int BALL_NUMBER = (int((-EMITTER_X_START - BALL_RADIUS) / BALL_DIAMETER)
		+ int((EMITTER_X_END - BALL_RADIUS) / BALL_DIAMETER) + 1)
		*(int((-EMITTER_Y_START - BALL_RADIUS) / BALL_DIAMETER)
			+ int((EMITTER_Y_END - BALL_RADIUS) / BALL_DIAMETER) + 1);
	const static float BALL_POS_X_START;
	const static float BALL_POS_Y_START;

	
#define COLLISIONPARAM_Gravity  (10.f)
#define COLLISIONPARAM_Spring  (0.5f)
#define COLLISIONPARAM_Damping  (0.02f)
#define COLLISIONPARAM_Shear  (0.1f)
#define COLLISIONPARAM_Attraction  (0.0f)
};
const float CollisionParam<float>::BALL_POS_X_START = EMITTER_X_START +
BALL_RADIUS*(int((-EMITTER_X_START - BALL_RADIUS) / BALL_DIAMETER) * 2 + 2);
const float CollisionParam<float>::BALL_POS_Y_START = EMITTER_Y_START +
BALL_RADIUS*(int((-EMITTER_Y_START - BALL_RADIUS) / BALL_DIAMETER) * 2 + 2);

template<typename Dtype>
struct RandSoruce
{
#define eachBallRandTableSize (20)
#define RandSoruceCnt (CollisionParam<float>::BALL_NUMBER*eachBallRandTableSize)
	int *randInt_dev{NULL};
	float *randFloat_dev{ NULL };
	int *randInt_host{ NULL };
	float *randFloat_host{ NULL };
	int *eachBallRandIndex_dev{NULL};
	int curIdx;
	RandSoruce()
	{
		curIdx = 0;
		srand((int)time(0));
		randInt_host = new int[RandSoruceCnt];
		randFloat_host = new float[RandSoruceCnt];
		randInt_dev = creatGpuData<int>(RandSoruceCnt);
		randFloat_dev = creatGpuData<float>(RandSoruceCnt);
		eachBallRandIndex_dev = creatGpuData<int>(CollisionParam<float>::BALL_NUMBER, true);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
		for (size_t i = 0; i < RandSoruceCnt; i++)
		{
			randInt_host[i] = rand() % 1000;
			randFloat_host[i] = rand() % 1000*0.001f;
		}
		cudaMemcpy(randInt_dev, randInt_host, sizeof(int)*RandSoruceCnt, cudaMemcpyHostToDevice);
		cudaMemcpy(randFloat_dev, randFloat_host, sizeof(float)*RandSoruceCnt, cudaMemcpyHostToDevice);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
		delete[]randInt_host;
		delete[]randFloat_host;
	}

	__device__  __forceinline__ float3 getFloat3(int ballIdx)
	{
		int &thisRandIdx = eachBallRandIndex_dev[ballIdx];
		thisRandIdx = (thisRandIdx + 1) % eachBallRandTableSize;
		float &v0 = randFloat_dev[eachBallRandTableSize*ballIdx + thisRandIdx];
		thisRandIdx = (thisRandIdx + 1) % eachBallRandTableSize;
		float &v1 = randFloat_dev[eachBallRandTableSize*ballIdx + thisRandIdx];
		thisRandIdx = (thisRandIdx + 1) % eachBallRandTableSize;
		float &v2 = randFloat_dev[eachBallRandTableSize*ballIdx + thisRandIdx];
		return make_float3(v0, v1, v2);
	}

};
RandSoruce<float> randTable;

struct tennisStat
{
	int idx;
	int inWhichBucket;
	float3 pos;
	float3 velocity;
	float3 force;	
	uchar3 rgb;
	int3 hash0;//第一个表示hash里面有几个值了
	int3 hash1;
	int3 hash2;
	
	//float mass;  假设质量都是1单位
	//void init() 
	//{
	//	pos.x =
	//	pos.z = EMITTER_Z;
	//	
	//	int rr = rand();
	//}
	//void collideWith(tennisStat&other)
	//{
	//	float3 collisionDirect;
	//	collisionDirect.x = pos.x - other.pos.x;
	//	collisionDirect.y = pos.x - other.pos.y;
	//	collisionDirect.z = pos.x - other.pos.z;
	//}
};

template<>
tennisStat* creatGpuData<tennisStat>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(tennisStat));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(tennisStat));
	}
	cudaSafeCall(cudaGetLastError());
	return (tennisStat*)gpudata;
}

template<typename>
struct BucketHashParam
{
	enum 
	{
		bucketHash_H_cnt = static_cast<int>((AVAILABLESPACE_Y_END - AVAILABLESPACE_Y_START)*1. / BALL_DIAMETER / 3.0) + 1, 
		bucketHash_W_cnt = static_cast<int>((AVAILABLESPACE_X_END - AVAILABLESPACE_X_START)*1. / BALL_DIAMETER / 3.0) + 1,
		bucketHash_Deep_cnt = static_cast<int>((AVAILABLESPACE_Z_END - AVAILABLESPACE_Z_START)*1. / BALL_DIAMETER / 3.0) + 1,
		maxCntInEachBucket = bucketHash_Deep_cnt*9,
		eachBucketIntOffset = bucketHash_Deep_cnt * (9+1),
		bucketNum = bucketHash_H_cnt*bucketHash_W_cnt,
	};
	const static float bucketHash_H_size;
	const static float bucketHash_W_size;
	int* bucketHashData{NULL};
	BucketHashParam()
	{
		////第一个值还是装this bucket里面存在的值
		bucketHashData = creatGpuData<int>(bucketHash_H_cnt*bucketHash_W_cnt*(bucketHash_Deep_cnt*9+1));
	}
};
const float BucketHashParam<float>::bucketHash_H_size = BALL_DIAMETER*3;
const float BucketHashParam<float>::bucketHash_W_size = BALL_DIAMETER * 3;
BucketHashParam<float> buckets;

tennisStat*tennis = NULL;
//int*

__global__ void initBall(tennisStat*data,const int ballNum,
						const float x_strat, const float y_strat,
						RandSoruce<float>*randS)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= ballNum)
		return;
	data[index].pos.x = x_strat + index%CollisionParam<float>::BALL_NUMBER_IN_ROW*BALL_DIAMETER;
	data[index].pos.x = y_strat + index/ CollisionParam<float>::BALL_NUMBER_IN_ROW*BALL_DIAMETER;
	data[index].pos.z = EMITTER_Z;
	data[index].force = make_float3(.0f, .0f, .0f);
	data[index].idx = index;
	data[index].rgb = make_uchar3(0,255,0);

	int thisBallRandIdx = randS->eachBallRandIndex_dev[index];
	float randF0 = randS->randFloat_dev[thisBallRandIdx];//x vel
	thisBallRandIdx = (thisBallRandIdx + 1) % eachBallRandTableSize;
	float randF1 = randS->randFloat_dev[thisBallRandIdx];//y vel
	thisBallRandIdx = (thisBallRandIdx + 1) % eachBallRandTableSize;
	float randF2 = randS->randFloat_dev[thisBallRandIdx];//z vel
	randF2 = randF2 > 0 ? randF2 : -randF2;
	thisBallRandIdx = (thisBallRandIdx + 1) % eachBallRandTableSize;
	randS->eachBallRandIndex_dev[index] = thisBallRandIdx;
	data[index].velocity = make_float3(randF0, randF1, randF2)*BALL_MAX_INIT_VEL;
}

int initTennisBalls()
{
	int ballNum = CollisionParam<float>::BALL_NUMBER;
	tennis = creatGpuData<tennisStat>(ballNum);
	dim3 blockSize(256);
	dim3 gridSize((ballNum + blockSize.x - 1) / blockSize.x);
	initBall<<<gridSize , blockSize >>>(tennis, ballNum, 
		CollisionParam<float>::BALL_POS_X_START, CollisionParam<float>::BALL_POS_Y_START,
		&randTable);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}


__global__ void putBallsInBucketsKernel(tennisStat*tennisBall, const int ballNum, BucketHashParam<float>*buckets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= buckets->bucketNum)
		return;
	float thisBucket_XStart;
	float thisBucket_XEnd;
	float thisBucket_YStart;
	float thisBucket_XEnd;

	for (int i = 0; i < ballNum; i++)
	{
		if
	}
}
int putBallsInBuckets()
{
	int bucketNum = buckets.bucketNum;
	int ballNum = CollisionParam<float>::BALL_NUMBER;
	dim3 blockSize(256);
	dim3 gridSize((bucketNum + blockSize.x - 1) / blockSize.x);
	putBallsInBucketsKernel << <gridSize, blockSize >> > (tennis, ballNum, buckets);
	return 0;
}




// collide two spheres using DEM method
__device__
float3 collideSpheres(float3 posA, float3 posB,
	float3 velA, float3 velB,
	float radiusA, float radiusB,
	float attraction)
{
	// calculate relative position
	float3 relPos = posB - posA;

	float dist = length(relPos);
	float collideDist = radiusA + radiusB;

	float3 force = make_float3(0.0f,.0f,.0f);

	if (dist < collideDist)
	{
		float3 norm = relPos / dist;

		// relative velocity
		float3 relVel = velB - velA;

		// relative tangential velocity
		float3 tanVel = relVel - (norm * dot(relVel, norm));

		// spring force
		force = norm *(-COLLISIONPARAM_Spring)*(collideDist - dist) ;
		// dashpot (damping) force
		force += relVel*COLLISIONPARAM_Damping;
		// tangential shear force
		force += tanVel*COLLISIONPARAM_Shear;
		// attraction
		force += relPos*attraction;
	}

	return force;
}
