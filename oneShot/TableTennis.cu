
#include<time.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "unreGpu.h"

#define __SHOW__

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
#define AVAILABLESPACE_X_START  (-.6)
#define AVAILABLESPACE_X_END  (.6)
#define AVAILABLESPACE_Y_START  (-.6)
#define AVAILABLESPACE_Y_END  (.6)
#define AVAILABLESPACE_Z_START  (0)
#define AVAILABLESPACE_Z_END  (1.2)

#define WORLD_X_START AVAILABLESPACE_X_START
#define WORLD_Y_START AVAILABLESPACE_Y_START
#define WORLD_Z_START AVAILABLESPACE_Z_START


#define BALL_RADIUS  (0.02)
#define BALL_DIAMETER  (0.04)//2* BALL_RADIUS;
#define BALL_LAYERS (10)
	static const int OBJECT_TRUCT_Z = (AVAILABLESPACE_Z_START + BALL_LAYERS*BALL_DIAMETER);
#define BALL_MAX_INIT_VEL  (0.1)

#define CELL_UNIT_SIZE BALL_DIAMETER
#define GRID_X_SIZE (64)
#define GRID_Y_SIZE (64)
#define GRID_Z_SIZE (64)
	static const int gridNum = GRID_X_SIZE*GRID_Y_SIZE*GRID_Z_SIZE;

	static const int BALL_NUMBER_IN_ROW = int((-AVAILABLESPACE_X_START - BALL_RADIUS) / BALL_DIAMETER)
		+ int((AVAILABLESPACE_X_END - BALL_RADIUS) / BALL_DIAMETER) + 1;
	static const int BALL_NUMBER_IN_COL = (int((-AVAILABLESPACE_Y_START - BALL_RADIUS) / BALL_DIAMETER)
		+ int((AVAILABLESPACE_Y_END - BALL_RADIUS) / BALL_DIAMETER) + 1);
	static const int BALL_NUMBER = BALL_NUMBER_IN_ROW * BALL_NUMBER_IN_COL*BALL_LAYERS;
	static const int BALL_NUMBER_INLAYER = BALL_NUMBER_IN_ROW * BALL_NUMBER_IN_COL;
	const static float BALL_POS_X_START;
	const static float BALL_POS_Y_START;

#define COLLISIONPARAM_Gravity  (0.001f)
#define COLLISIONPARAM_Spring  (0.2f)
#define COLLISIONPARAM_Damping  (0.1f)//global
#define COLLISIONPARAM_BoundryDamping  (-0.3f)
#define COLLISIONPARAM_Shear  (0.1f)
#define COLLISIONPARAM_Attraction  (0.0f)
};
const float CollisionParam<float>::BALL_POS_X_START = -BALL_RADIUS*(int((-AVAILABLESPACE_X_START - BALL_RADIUS) / BALL_DIAMETER) * 2);
const float CollisionParam<float>::BALL_POS_Y_START = -BALL_RADIUS*(int((-AVAILABLESPACE_Y_START - BALL_RADIUS) / BALL_DIAMETER) * 2);


struct RandSoruce
{
#define eachBallRandTableSize (10)
	static const int  RandSoruceCnt = (CollisionParam<float>::BALL_NUMBER*eachBallRandTableSize);
	int *randInt_dev{ NULL };
	float *randFloat_dev{ NULL };
	int *randInt_host{ NULL };
	float *randFloat_host{ NULL };
	int curIdx;
	RandSoruce()
	{
		curIdx = 0;
		srand((int)time(0));
		randInt_host = new int[RandSoruceCnt];
		randFloat_host = new float[RandSoruceCnt];
		randInt_dev = creatGpuData<int>(RandSoruceCnt);
		randFloat_dev = creatGpuData<float>(RandSoruceCnt);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
		for (size_t i = 0; i < RandSoruceCnt; i++)
		{
			randInt_host[i] = rand() % 1000;
			randFloat_host[i] = rand() % 100 * 0.01f;
		}
		cudaMemcpy(randInt_dev, randInt_host, sizeof(int)*RandSoruceCnt, cudaMemcpyHostToDevice);
		cudaMemcpy(randFloat_dev, randFloat_host, sizeof(float)*RandSoruceCnt, cudaMemcpyHostToDevice);
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
		delete[]randInt_host;
		delete[]randFloat_host;
	}

};
RandSoruce* randTable=NULL;

struct tennisStat
{
	float3 *pos;
	float3 *vel;
	float3 *pos_sorted;
	float3 *vel_sorted;
	uchar3 *rgb;
	uchar3 *rgb_sorted;
	tennisStat(bool fore_zeros = false)
	{
		cudaMalloc((void**)&pos, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
		cudaMalloc((void**)&vel, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
		cudaMalloc((void**)&pos_sorted, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
		cudaMalloc((void**)&vel_sorted, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
		cudaMalloc((void**)&rgb, CollisionParam<float>::BALL_NUMBER * sizeof(uchar3));
		cudaMalloc((void**)&rgb_sorted, CollisionParam<float>::BALL_NUMBER * sizeof(uchar3));
		if (fore_zeros)
		{
			cudaMemset(pos, 0, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
			cudaMemset(vel, 0, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
			cudaMemset(pos_sorted, 0, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
			cudaMemset(vel_sorted, 0, CollisionParam<float>::BALL_NUMBER * sizeof(float3));
			cudaMemset(rgb, 0, CollisionParam<float>::BALL_NUMBER * sizeof(char3));
			cudaMemset(rgb_sorted, 0, CollisionParam<float>::BALL_NUMBER * sizeof(char3));
		}
		cudaSafeCall(cudaGetLastError());
		cudaSafeCall(cudaDeviceSynchronize());
	}
};
tennisStat* tennis = NULL;

template<>
unsigned int* creatGpuData<unsigned int>(const int elemCnt, bool fore_zeros)
{
	void*gpudata = NULL;
	cudaMalloc((void**)&gpudata, elemCnt * sizeof(unsigned int));
	if (fore_zeros)
	{
		cudaMemset(gpudata, 0, elemCnt * sizeof(unsigned int));
	}
	cudaSafeCall(cudaGetLastError());
	return (unsigned int*)gpudata;
}

struct integrate_functor
{
	float deltaTime;

	__host__ __device__
		integrate_functor(float delta_time) : deltaTime(delta_time) {}

	template <typename Tuple>
	__device__
		void operator()(Tuple t)
	{
		float3 &pos = thrust::get<0>(t);
		float3 &vel = thrust::get<1>(t);
		//float3 pos = make_float3(posData.x, posData.y, posData.z);
		//float3 vel = make_float3(velData.x, velData.y, velData.z);

		//vel += (COLLISIONPARAM_Gravity * deltaTime);
		vel.y+= (COLLISIONPARAM_Gravity * deltaTime);
		//vel *= COLLISIONPARAM_Damping;

		// new position = old position + velocity * deltaTime
		pos += vel * deltaTime;

		// set this to zero to disable collisions with cube sides
#if 1

		if (pos.x > AVAILABLESPACE_X_END - BALL_RADIUS)
		{
			pos.x = AVAILABLESPACE_X_END - BALL_RADIUS;
			vel.x *= COLLISIONPARAM_BoundryDamping;
		}

		if (pos.x < AVAILABLESPACE_X_START + BALL_RADIUS)
		{
			pos.x = AVAILABLESPACE_X_START + BALL_RADIUS;
			vel.x *= COLLISIONPARAM_BoundryDamping;
		}

		if (pos.y > AVAILABLESPACE_Y_END - BALL_RADIUS)
		{
			pos.y = AVAILABLESPACE_Y_END - BALL_RADIUS;
			vel.y *= COLLISIONPARAM_BoundryDamping;
		}
		

		if (pos.z > AVAILABLESPACE_Z_END - BALL_RADIUS)
		{
			pos.z = AVAILABLESPACE_Z_END - BALL_RADIUS;
			vel.z *= COLLISIONPARAM_BoundryDamping;
		}

		if (pos.z < AVAILABLESPACE_Z_START + BALL_RADIUS)
		{
			pos.z = AVAILABLESPACE_Z_START + BALL_RADIUS;
			vel.z *= COLLISIONPARAM_BoundryDamping;
		}

#endif
		if (pos.y < AVAILABLESPACE_Y_START + BALL_RADIUS)
		{
			pos.y = AVAILABLESPACE_Y_START + BALL_RADIUS;
			vel.y *= COLLISIONPARAM_BoundryDamping;
		}

		// store new position and velocity
		//thrust::get<0>(t) = make_float4(pos, posData.w);
		//thrust::get<1>(t) = make_float4(vel, velData.w);
	}
};

__device__ int3 calcGridPos(const float3& p)
{
	int3 gridPos;
	gridPos.x = floor((p.x - WORLD_X_START) / CELL_UNIT_SIZE);
	gridPos.y = floor((p.y - WORLD_Y_START) / CELL_UNIT_SIZE);
	gridPos.z = floor((p.z - WORLD_Z_START) / CELL_UNIT_SIZE);
	return gridPos;
}
__device__ unsigned int calcGridHash(int3 &gridPos)
{
	gridPos.x = gridPos.x & (GRID_X_SIZE - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (GRID_Y_SIZE - 1);  // And the head number has been truct
	gridPos.z = gridPos.z & (GRID_Z_SIZE - 1);
	return (gridPos.z*GRID_Y_SIZE*GRID_X_SIZE) + (gridPos.y*GRID_X_SIZE) + gridPos.x;
}
void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n + numThreads - 1) / numThreads;
}

unsigned int *gridParticleHash = NULL;
unsigned int *gridParticleIndex = NULL;
unsigned int *cellStart = NULL;
unsigned int *cellEnd = NULL;
//int*

__global__ void initBall(float3*pos, 
						float3*vel,
						uchar3*rgb,
						const int ballNum,
						const float x_strat, const float y_strat,
						const int rows, const int cols,
						float*randFloat, int*randInt)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= ballNum)
		return;
	int num_in_layer = rows*cols;

	(pos)[index].z = AVAILABLESPACE_Z_START+index / num_in_layer*BALL_DIAMETER;
	int posXY = index % num_in_layer;
	pos[index].x = x_strat + posXY % CollisionParam<float>::BALL_NUMBER_IN_ROW*BALL_DIAMETER;
	pos[index].y = y_strat + posXY / CollisionParam<float>::BALL_NUMBER_IN_ROW*BALL_DIAMETER;
	
	int thisBallRandIdx0 = ( index) % RandSoruce::RandSoruceCnt;
	int thisBallRandIdx1 = ( index+1) % RandSoruce::RandSoruceCnt;
	int thisBallRandIdx2 = ( index+2) % RandSoruce::RandSoruceCnt;
	float &randF0 = randFloat[thisBallRandIdx0];//x vel
	float &randF1 = randFloat[thisBallRandIdx1];//y vel
	float &randF2 = randFloat[thisBallRandIdx2];//z vel	
	randF2 = randF2 > 0 ? randF2 : -randF2;
	vel[index] = make_float3(randF0, randF1, randF2)*BALL_MAX_INIT_VEL;

	//unsigned char randI0 = (randInt[thisBallRandIdx0] % 5+5)*27;//
	//unsigned char randI1 = (randInt[thisBallRandIdx1] % 5+5)* 27;//
	//unsigned char randI2 = (randInt[thisBallRandIdx2] % 5+5)* 27;//
	//rgb[index] = make_uchar3(randI0, randI1, randI2);
	rgb[index] = make_uchar3(0, 255,0 );
}

int initTennisBalls()
{
	int ballNum = CollisionParam<float>::BALL_NUMBER;
	if (randTable==NULL)
	{
		randTable = new	RandSoruce();
	}
	if (tennis==NULL)
	{
		tennis = new	tennisStat();
	}
	if (gridParticleHash==NULL)
	{
		gridParticleHash = creatGpuData<unsigned int>(ballNum);
	}
	if (gridParticleIndex==NULL)
	{
		gridParticleIndex = creatGpuData<unsigned int>(ballNum);
	}
	if (cellStart == NULL)
	{
		cellStart = creatGpuData<unsigned int>(CollisionParam<float>::gridNum);
	}
	if (cellEnd == NULL)
	{
		cellEnd = creatGpuData<unsigned int>(CollisionParam<float>::gridNum);
	}	
	dim3 blockSize(256);
	dim3 gridSize((ballNum + blockSize.x - 1) / blockSize.x);
	initBall << <gridSize, blockSize >> >(tennis->pos, tennis->vel, tennis->rgb, 
		ballNum,
		CollisionParam<float>::BALL_POS_X_START, CollisionParam<float>::BALL_POS_Y_START,
		CollisionParam<float>::BALL_NUMBER_IN_COL, CollisionParam<float>::BALL_NUMBER_IN_ROW,
		randTable->randFloat_dev, randTable->randInt_dev);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return ballNum;
}

void transVNmap2cells(const float*vmap, const float*nmap, float objectCells)
{}

void action(float deltaTime) {}

void integrateSystem(float deltaTime)
{
	thrust::device_ptr<float3> d_pos3(tennis->pos);
	thrust::device_ptr<float3> d_vel3(tennis->vel);
	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(d_pos3, d_vel3)),
		thrust::make_zip_iterator(thrust::make_tuple(d_pos3 + CollisionParam<float>::BALL_NUMBER, d_vel3 + CollisionParam<float>::BALL_NUMBER)),
		integrate_functor(deltaTime));
}


__global__
void calcHashD(unsigned int   *gridParticleHash,  // output
	unsigned int   *gridParticleIndex, // output
	float3 *pos,               // input: positions
	unsigned int    numParticles)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	if (index >= numParticles) return;

	float3& p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(p);
	unsigned int hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;
}

void calcHash()
{
	unsigned int numThreads, numBlocks;
	computeGridSize(CollisionParam<float>::BALL_NUMBER, 256, numBlocks, numThreads);
	// execute the kernel
	calcHashD << < numBlocks, numThreads >> >(gridParticleHash,
		gridParticleIndex,
		tennis->pos,
		CollisionParam<float>::BALL_NUMBER);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}

void sortParticles()
{
	thrust::sort_by_key(
		thrust::device_ptr<unsigned int>(gridParticleHash),
		thrust::device_ptr<unsigned int>(gridParticleHash + CollisionParam<float>::BALL_NUMBER),
		thrust::device_ptr<unsigned int>(gridParticleIndex));
}


__global__
void reorderDataAndFindCellStartD(unsigned int   *cellStart,        // output: cell start index
	unsigned int   *cellEnd,          // output: cell end index
	float3 *sortedPos,        // output: sorted positions
	float3 *sortedVel,        // output: sorted velocities
	uchar3 *sortedRgb,
	unsigned int   *gridParticleHash, // input: sorted grid hashes
	unsigned int   *gridParticleIndex,// input: sorted particle indices
	float3 *oldPos,           // input: sorted position array
	float3 *oldVel,           // input: sorted velocity array
	uchar3 *oldRgb,
	unsigned int    numParticles)
{
	extern __shared__ unsigned int sharedHash[];    // blockSize + 1 elements
	unsigned int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	unsigned int hash;

	if (index < numParticles)
	{
		hash = gridParticleHash[index];
		sharedHash[threadIdx.x + 1] = hash;
		if (index > 0 && threadIdx.x == 0)
		{
			sharedHash[0] = gridParticleHash[index - 1];
		}
	}
	__syncthreads();

	if (index < numParticles)
	{
		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}
		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}
		unsigned int sortedIndex = gridParticleIndex[index];
		float3& pos = oldPos[sortedIndex];       // macro does either global read or texture fetch
		float3& vel = oldVel[sortedIndex];       // see particles_kernel.cuh
		uchar3& rgb = oldRgb[sortedIndex];       // see particles_kernel.cuh
		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedRgb[index] = rgb;
	}
}

void reorderDataAndFindCellStart()
{
	unsigned int numThreads, numBlocks;
	computeGridSize(CollisionParam<float>::BALL_NUMBER, 256, numBlocks, numThreads);
	int gridNum = CollisionParam<float>::gridNum;
	cudaMemset(cellStart, 0xffffffff, gridNum * sizeof(unsigned int));
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	unsigned int smemSize = sizeof(unsigned int)*(numThreads + 1);
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(
		cellStart,
		cellEnd,
		tennis->pos_sorted,
		tennis->vel_sorted,
		tennis->rgb_sorted,
		gridParticleHash,
		gridParticleIndex,
		tennis->pos,
		tennis->vel,
		tennis->rgb,
		CollisionParam<float>::BALL_NUMBER);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}



__device__
float3 collideSpheres(float3 posA, float3 posB,
	float3 velA, float3 velB,
	float radiusA, float radiusB,
	float attraction)
{
	float3 relPos = posB - posA;
	float dist = length(relPos);
	float collideDist = radiusA + radiusB;
	float3 force = make_float3(0.f, 0.f, 0.f);
	if (dist < collideDist)
	{
		float3 norm = relPos / dist;
		float3 relVel = velB - velA;
		float3 tanVel = relVel - (norm*dot(relVel, norm));
		// spring force
		force = norm*(collideDist - dist)*(-COLLISIONPARAM_Spring);
		// dashpot (damping) force
		force += relVel*COLLISIONPARAM_Damping;
		// tangential shear force
		force += tanVel*COLLISIONPARAM_Shear;
		// attraction
		force += relPos*COLLISIONPARAM_Attraction;
	}
	return force;
}



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
	unsigned int    index,
	float3  pos,
	float3  vel,
	float3 *oldPos,
	float3 *oldVel,
	unsigned int   *cellStart,
	unsigned int   *cellEnd)
{
	unsigned int gridHash = calcGridHash(gridPos);
	unsigned int startIndex = cellStart[gridHash];
	float3 force = make_float3(0.f, 0.f, 0.f);
	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		unsigned int &endIndex = cellEnd[gridHash];
		for (unsigned int j = startIndex; j<endIndex; j++)
		{
			if (j != index)                // check not colliding with self
			{
				float3 pos2 = oldPos[j];
				float3 vel2 = oldVel[j];
				force += collideSpheres(pos, pos2, vel, vel2, BALL_RADIUS, BALL_RADIUS, COLLISIONPARAM_Attraction);
			}
		}
	}
	return force;
}


__global__
void collideD(float3 *newVel,               // output: new velocity
	float3 *oldPos,               // input: sorted positions
	float3 *oldVel,               // input: sorted velocities
	unsigned int   *gridParticleIndex,    // input: sorted particle indices
	unsigned int   *cellStart,
	unsigned int   *cellEnd,
	unsigned int    numParticles)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= numParticles) return;

	float3 &pos = oldPos[index];
	float3 &vel = oldVel[index];

	// get address in grid
	int3 gridPos = calcGridPos(pos);

	// examine neighbouring cells
	float3 force = make_float3(0.f,0.f,0.f);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				int3 neighbourPos = gridPos + make_int3(x, y, z);
				force += collideCell(neighbourPos, index, pos, vel, oldPos, oldVel, cellStart, cellEnd);
			}
		}
	}

	// collide with cursor sphere
	//force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);

	// write new velocity back to original unsorted location
	unsigned int originalIndex = gridParticleIndex[index];
	newVel[originalIndex] = vel+ force;
}

void collide()
{
	// thread per particle
	unsigned int numThreads, numBlocks;
	computeGridSize(CollisionParam<float>::BALL_NUMBER, 64, numBlocks, numThreads);
	collideD << < numBlocks, numThreads >> >(
		tennis->vel,
		tennis->pos_sorted,
		tennis->vel_sorted,
		gridParticleIndex,
		cellStart,
		cellEnd,
		CollisionParam<float>::BALL_NUMBER);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
}


int loopProc(const float deltaTime,float*pos_host, float*vel_host,unsigned char*rgb_host)
{
	integrateSystem(deltaTime);
	calcHash();
	sortParticles();
	reorderDataAndFindCellStart();
	collide();
#ifdef __SHOW__
	if (pos_host != NULL&&vel_host != NULL&&rgb_host != NULL)
	{
		cudaMemcpy(pos_host, tennis->pos, 3 * CollisionParam<float>::BALL_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(vel_host, tennis->vel, 3 * CollisionParam<float>::BALL_NUMBER * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(rgb_host, tennis->rgb, 3 * CollisionParam<float>::BALL_NUMBER * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
#endif // __SHOW__
	
	
	return 0;
}


#define GenerMeshData
#ifdef GenerMeshData
float*pos_dev = NULL;
float*norm_dev = NULL;
unsigned int*triIdx_dev = NULL;
unsigned char*rgb_dev = NULL;

float*pos_dev_template = NULL;
float*norm_dev_template = NULL;
unsigned int*triIdx_dev_template = NULL;
int initMeshData_dev(const int posTemplateNum, const int normTemplateNum)
{
	pos_dev_template = creatGpuData<float>(3 * posTemplateNum);
	norm_dev_template = creatGpuData<float>(3 * normTemplateNum);
	triIdx_dev_template = creatGpuData<unsigned int>(3 * normTemplateNum);

	pos_dev = creatGpuData<float>(3 * posTemplateNum*CollisionParam<float>::BALL_NUMBER);
	norm_dev = creatGpuData<float>(3 * normTemplateNum*CollisionParam<float>::BALL_NUMBER);
	triIdx_dev = creatGpuData<unsigned int>(3 * normTemplateNum*CollisionParam<unsigned int>::BALL_NUMBER);
	rgb_dev = creatGpuData<unsigned char>(3 * normTemplateNum*CollisionParam<unsigned char>::BALL_NUMBER);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return CollisionParam<float>::BALL_NUMBER;
}

__global__ void getMeshPosDataKernel(const int posTemplateNum, const int ballNum,
	float* pos_dev,float3*tennis_pos, float* pos_dev_template)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= ballNum)
		return;
	float thisBallX = tennis_pos[index].x;
	float thisBallY = tennis_pos[index].y;
	float thisBallZ = tennis_pos[index].z;
	int startIdx = posTemplateNum*index*3;
	for (int i = 0; i < posTemplateNum; i++)
	{
		int this_index = startIdx + i;
		pos_dev[3 * this_index] = pos_dev_template[3 * i] + thisBallX;
		pos_dev[3 * this_index + 1] = pos_dev_template[3 * i + 1] + thisBallY;
		pos_dev[3 * this_index + 2] = pos_dev_template[3 * i + 2] + thisBallZ;
	}	
}
__global__ void getMeshNormTriRGBDataKernel(const int normTemplateNum, const int ballNum,
	float*norm_dev, float*norm_dev_template,
	unsigned int*triIdx_dev, unsigned int*triIdx_dev_template,
	unsigned char*rgb_dev, uchar3*tennis_rgb)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index >= ballNum)
		return;
	int startIdx = normTemplateNum*index;
	for (int i = 0; i < normTemplateNum; i++)
	{
		int this_index = startIdx + i;
		norm_dev[3 * this_index] = norm_dev_template[3 * i];
		norm_dev[3 * this_index + 1] = norm_dev_template[3 * i + 1];
		norm_dev[3 * this_index + 2] = norm_dev_template[3 * i + 2];
		triIdx_dev[3 * this_index] = triIdx_dev_template[3 * i]+ startIdx;
		triIdx_dev[3 * this_index + 1] = triIdx_dev_template[3 * i + 1]+ startIdx;
		triIdx_dev[3 * this_index + 2] = triIdx_dev_template[3 * i + 2]+ startIdx;
		rgb_dev[3 * this_index] = tennis_rgb[startIdx].x;
		rgb_dev[3 * this_index + 1] = tennis_rgb[startIdx].y;
		rgb_dev[3 * this_index + 2] = tennis_rgb[startIdx].z;
	}
}


int getMeshData(float*pos_host, float*norm_host, unsigned int*triIdx_host,unsigned char*rgb_host,
	float*pos_host_template, float*norm_host_template, unsigned int*triIdx_host_template,
	const int posTemplateNum, const int normTemplateNum)
{	
	cudaMemcpy(pos_dev_template, pos_host_template, 3 * posTemplateNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(norm_dev_template, norm_host_template, 3 * normTemplateNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(triIdx_dev_template, triIdx_host_template, 3 * normTemplateNum * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	dim3 blockSize(256);
	dim3 gridSize((CollisionParam<float>::BALL_NUMBER + blockSize.x - 1) / blockSize.x);
	getMeshPosDataKernel << <gridSize, blockSize >> > (posTemplateNum, CollisionParam<float>::BALL_NUMBER,
		pos_dev,tennis->pos, pos_dev_template);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	getMeshNormTriRGBDataKernel << <gridSize, blockSize >> > (normTemplateNum, CollisionParam<float>::BALL_NUMBER,
		norm_dev, norm_dev_template,
		triIdx_dev, triIdx_dev_template,
		rgb_dev, tennis->rgb);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());

	cudaMemcpy(pos_host, pos_dev, 3 * posTemplateNum * CollisionParam<float>::BALL_NUMBER*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(norm_host, norm_dev, 3 * normTemplateNum *CollisionParam<float>::BALL_NUMBER* sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(triIdx_host, triIdx_dev, 3 * normTemplateNum *CollisionParam<float>::BALL_NUMBER * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgb_host, rgb_dev, 3 * normTemplateNum*CollisionParam<float>::BALL_NUMBER * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaSafeCall(cudaGetLastError());
	cudaSafeCall(cudaDeviceSynchronize());
	return 0;
}
#endif // GenerMeshData


