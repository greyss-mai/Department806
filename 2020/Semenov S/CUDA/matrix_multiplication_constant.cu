#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define N 16

#define BLOCK_SIZE 16

// simulation parameters in constant memory
__constant__ float d_a[N*N];
__constant__ float d_b[N*N];

__global__ void matMult(int n, float * c)
{
	int   bx = blockIdx.x;
	int   by = blockIdx.y;
	int   tx = threadIdx.x;
	int   ty = threadIdx.y;
	float sum = 0.0f;
	int   ia = n * BLOCK_SIZE * by + n * ty;
	int   ib = BLOCK_SIZE * bx + tx;
	int   ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	for (int k = 0; k < n; k++)
		sum += d_a[ia + k] * d_b[ib + k*n];

	c[ic + n * ty + tx] = sum;
}


int main(int argc, char *  argv[])
{
	int         numBytes = N * N * sizeof (float);
	float     * adev, *bdev, *cdev;
	dim3        threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3        blocks(N / threads.x, N / threads.y);

	//Generate matricies
	float * h_a = new float[N*N];
	float * h_b = new float[N*N];
	float * c = new float[N*N];

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			h_a[i*N + j] = 1;
			h_b[i*N + j] = 2;
		}
	}

	cudaMalloc((void**)&cdev, numBytes);	// allocate DRAM

	// copy matricies to constant memory
	cudaMemcpyToSymbol(d_a, h_a, numBytes);
	cudaMemcpyToSymbol(d_b, h_b, numBytes);

	matMult <<<blocks, threads >>> (N, cdev);

	cudaThreadSynchronize();
	cudaMemcpy(c, cdev, numBytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%.5f ", c[i*N + j]);
		}
		printf("\n");
	}

	// free GPU memory
	cudaFree(cdev);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	return 0;
}
