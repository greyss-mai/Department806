#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <chrono>

cudaError_t addWithCuda(long double *c, unsigned int size, double *x);


__device__ long double fact(int n) {
	long double f = 1;
	for (int i = 2; i <= n; i++) {
		f *= i;
	}
	return f;
}

__device__ long double power(double a, double b) {
	long double c = long double(a);
	for (int i = 1; i < b; i++) {
		c *= a;
	}
	return c;
}

__global__ void addKernel(long double *c, const double *x)
{
	int i = threadIdx.x;
	c[i] = power(x[0], i) / fact(i);
}

void calculateE_gpu(double *x, int arraySize) {
	long double *c = new long double[arraySize];
	auto begin = std::chrono::steady_clock::now();

	cudaError_t cudaStatus = addWithCuda(c, arraySize, x);

	auto end = std::chrono::steady_clock::now();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");

	}


	long double e = 0;
	for (int i = 0; i < arraySize; i++) {
		e += c[i];
	}
	printf("With GPU(%i components)e^%lf = %.16lf \n", arraySize, x[0], e);
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	printf("Time= %i nanoseconds", elapsed_ms.count());
}

long double fact_cpu(int n) {
	long double f = 1;
	for (int i = 2; i <= n; i++) {
		f *= i;
	}
	return f;
}

long double power_cpu(double a, double b) {
	long double c = long double(a);
	for (int i = 1; i < b; i++) {
		c *= a;
	}
	return c;
}

void calculateE_cpu(double x, int number) {
	auto begin = std::chrono::steady_clock::now();


	long double e = 1;
	for (int i = 1; i < number; i++) {
		e += power_cpu(x, i) / fact_cpu(i);
	}

	auto end = std::chrono::steady_clock::now();

	printf("With CPU(%i components)e^%lf = %.16lf \n", number, x, e);
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	printf("Time= %i nanoseconds", elapsed_ms.count());
}

int main()
{
	cudaError_t cudaStatus;
	int arraySize = 8;
	double x[1] = { 1 };
	//long double c[arraySize] = { 0 };

	std::cout << "Enter X: ";
	std::cin >> x[0];


	for (int j = arraySize; j < 2500; j *= 2) {
		calculateE_gpu(x, j);
		printf("\n");
		calculateE_cpu(x[0], j);
		printf("\n");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(long double *c, unsigned int size, double *x)
{

	long double *dev_c = 0;
	double *dev_x = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(long double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_x, x, sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/*cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}*/

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size >>> (dev_c, dev_x);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(long double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_x);

	return cudaStatus;
}

