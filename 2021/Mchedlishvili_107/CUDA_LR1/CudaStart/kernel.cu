#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>

#define N 4096
#define PI 3.1415926

__global__ void kernel(float* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//foreach N * 2 intervals from 0 to 2 PI
	float x = 6.0f * (float) PI * (float) idx / (float) N;
	//Calclulte
	out[idx] = sinf(x) / x;
}

int main()
{
	std::cout << "<-------------------------------GPU-------------------------------->\n";

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	float* out = new float[N];
	float* dev;

	//Allocate memory on device
	cudaMalloc((void**)&dev, N * sizeof(float));
	cudaThreadSynchronize();
	
	dim3 dimThreads(4, 1);
	dim3 dimBlocks(N / dimThreads.x, 1);

	//Execute method on GPU
	kernel <<<dimBlocks, dimThreads>>> (dev);

	//copy data from GPU to CPU(host)
	cudaMemcpy(out, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("DEVICE GPU compute time: %.10f milliseconds\n\n", gpuTime);

	//Free and reset GPU
	cudaFree(dev);
	cudaDeviceReset();

	for (int i = 0; i < N; i++) {
		printf("a[%i] = %.5f\n", i, out[i]);
	}

	std::cout << "<-------------------------------CPU-------------------------------->\n";
	
	double start2;
	double end2;
	start2 = omp_get_wtime();

	float* a = new float[N];

	for (int i = 0; i < N; i++) {
		float x = 6.0f * (float)PI * (float)i / (float)N;
		a[i] = sinf(x) / x;
	}

	end2 = omp_get_wtime();

	printf("CPU compute time: %f milliseconds\n\n", (end2 - start2) * 1000);

	return 0;
}

