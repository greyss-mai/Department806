#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 16

#define BLOCK_SIZE 16

__global__ void matMult(float * a, float * b, int n, float * c) {
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;
	int aBegin = n * BLOCK_SIZE * by;
	int aEnd = aBegin + n - 1;
	int bBegin = BLOCK_SIZE * bx;
	int aStep = BLOCK_SIZE, bStep = BLOCK_SIZE * n;
	float sum = 0.0f;

	float as[BLOCK_SIZE][BLOCK_SIZE];
	float bs[BLOCK_SIZE][BLOCK_SIZE];

	for (int i = 0; i < BLOCK_SIZE; i++){
		for (int j = 0; j < BLOCK_SIZE; j++){
			for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep){
				as[j][i] = a[ia + n * j + i];
				bs[j][i] = b[ib + n * j + i];
				//__syncthreads(); 	// Synchronize to make sure the matrices are loaded 	
			}
		}
	}

	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep){
		for (int k = 0; k < BLOCK_SIZE; k++)
			sum += as[ty][k] * bs[k][tx];
		//__syncthreads(); 	// Synchronize to make sure submatrices not needed
	}

	c[n * BLOCK_SIZE * by + BLOCK_SIZE * bx + n * ty + tx] = sum;
}



int main(int argc, char *  argv[])
{
	int         numBytes = N * N * sizeof (float);
	float     * adev, *bdev, *cdev;
	dim3        threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3        blocks(N / threads.x, N / threads.y);

	//Generate matricies
	float * a = new float[N*N];
	float * b = new float[N*N];
	float * c = new float[N*N];

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			a[i*N + j] = 1;
			b[i*N + j] = 2;
		}
	}

	cudaMalloc((void**)&adev, numBytes);	// allocate DRAM
	cudaMalloc((void**)&bdev, numBytes);	// allocate DRAM
	cudaMalloc((void**)&cdev, numBytes);	// allocate DRAM
	// copy from CPU to DRAM
	cudaMemcpy(adev, a, numBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(bdev, b, numBytes, cudaMemcpyHostToDevice);

	matMult << <blocks, threads >> > (adev, bdev, N, cdev);

	cudaThreadSynchronize();
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%.5f ", c[i*N + j]);
		}
		printf("\n");
	}

	// free GPU memory
	cudaFree(adev);
	cudaFree(bdev);
	cudaFree(cdev);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	return 0;
}
