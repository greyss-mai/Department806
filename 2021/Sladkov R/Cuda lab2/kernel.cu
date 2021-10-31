#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#define N 100000

__global__ void kernel(int* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int answ = 1;

	for (int i = 2; i < idx; i++) {
		if (idx % i == 0) {
			answ += i;
		}
	}

	if (answ == idx) { 
		out[idx] = answ;
	}
	else
	{
		out[idx] = NULL;
	}
}

int main()
{

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int* out = new int[N];
	int* dev;

	cudaMalloc((void**)&dev, N * sizeof(float));
	cudaThreadSynchronize();

	dim3 dimThreads(N / 8, 1);
	dim3 dimBlocks(N / dimThreads.x, 1);

	kernel << <dimBlocks, dimThreads >> > (dev);

	cudaMemcpy(out, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("N = %d\n\nGPU compute time: %.10f milliseconds\n", N, gpuTime);

	//for (int i = 0; i < N; i++)
		//if(out[i]) printf("%d ", out[i]);

	cudaFree(dev);
	cudaDeviceReset();

	double start2;
	double end2;
	start2 = omp_get_wtime();

	float* a = new float[N];

	for (int i = 0; i < N; i++) {
		int answ = 1;

		for (int j = 2; i < i; j++) {
			if (i % j == 0) {
				answ += j;
			}
		}

		if (answ == i) a[i] == answ;
	}

	end2 = omp_get_wtime();

	printf("\n\nCPU compute time: %f milliseconds\n", (end2 - start2) * 1000);

	return 0;
}
