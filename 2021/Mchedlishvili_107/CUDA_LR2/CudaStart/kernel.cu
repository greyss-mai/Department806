#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <omp.h>
#include <iostream>
#include <fstream>

__constant__ float c_H;
__constant__ float c_N;
__constant__ float c_Pi;

__device__ float Function(float x) {
	return sinf(x) / x;
}

__global__ void kernel(float* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//foreach N * 2 intervals from 0 to 2 PI
	float x = 6.0f * c_Pi * (float) idx / c_N;
	//Calclalte
	out[idx] = (Function(x + c_H) - Function(x - c_H)) / (2 * c_H);
}

__host__ float Function1(float x) {
	return sinf(x) / x;
}

int main()
{
	float H = 0.01f;
	float N = 32768;
	float PI = 3.1415926;

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

	cudaMemcpyToSymbol(c_H, &H, sizeof(float));
	cudaMemcpyToSymbol(c_N, &N, sizeof(float));
	cudaMemcpyToSymbol(c_Pi, &PI, sizeof(float));

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
		//printf("a[%i] = %.5f\n", i, out[i]);
	}

	std::cout << "<-------------------------------CPU-------------------------------->\n";
	
	double start2;
	double end2;
	start2 = omp_get_wtime();

	float* a = new float[N];

	for (int i = 0; i < N; i++) {
		float x = 6.0f * (float)PI * (float)i / (float)N;
		a[i] = (Function1(x + H) - Function1(x - H)) / (2 * H);
	}

	end2 = omp_get_wtime();

	printf("CPU compute time: %f milliseconds\n\n", (end2 - start2) * 1000);

	std::fstream f;
	f.open("D://test.txt", std::fstream::in | std::fstream::out);
	f << out[0];

	for (int i = 1; i < N; i++) {
		f << ", " << out[i];
	}
	f.close();
	return 0;
}

