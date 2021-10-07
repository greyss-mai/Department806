#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <iostream>

__constant__ unsigned long long globalN[1];

__global__ void kernel_isPerfectNumber(bool* arr);
__device__ bool isPerfectNumber(unsigned  long long number);
bool isPerfectNumber_(unsigned long long number);
void host_isPerfectNumber(bool* arr, unsigned long long size);
cudaError_t CUDA_get_perfect_numbers(bool* arr, unsigned long long threads, unsigned long long blocks, unsigned long long N);

int main()
{

	unsigned long long threads = 512;
	unsigned long long blocks = 32;
	unsigned long long size = threads * blocks;
	unsigned long long N = size;
	bool* array_cpu = (bool*)malloc(size * sizeof(bool));
	bool* array_gpu = (bool*)malloc(size * sizeof(bool));

	unsigned int start_time;
	unsigned int end_time;
	unsigned int search_time;

	start_time = clock();
	host_isPerfectNumber(array_cpu, N);
	end_time = clock();
	search_time = end_time - start_time;
	std::cout << "CPU runtime: ";
	std::cout << search_time / 1000.0 << std::endl;


	start_time = clock();
	CUDA_get_perfect_numbers(array_gpu, threads, blocks, N);
	end_time = clock();
	search_time = end_time - start_time;
	std::cout << "GPU runtime: ";
	std::cout << search_time / 1000.0 << std::endl;

	std::cout << "Perfect numbers:\n";
	for (unsigned long long i = 1; i < N; i++)
	{		
		if (array_gpu[i])
			std::cout << i << std::endl;
	}

	return 0;
}

//CPU
void host_isPerfectNumber(bool* arr, unsigned long long size)
{
	for (unsigned long long i = 0; i < size; i++)
	{
		if (isPerfectNumber_(i))
			arr[i] = true;
		else
			arr[i] = false;
	}
}

bool isPerfectNumber_(unsigned long long number)
{
	unsigned long long i = 1, sum = 0;
	while (i < number)
	{
		if (number % i == 0)
			sum += i;
		i++;
	}

	if (sum == number)
		return true;
	else
		return false;
}


//GPU

cudaError_t CUDA_get_perfect_numbers(bool* arr, unsigned long long threads, unsigned long long blocks, unsigned long long N)
{
	unsigned long long size = threads * blocks;

	bool* dev_arr = nullptr;

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpyToSymbol(globalN, &N, sizeof(unsigned long long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	kernel_isPerfectNumber << <blocks, threads >> > (dev_arr);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(arr, dev_arr, size * sizeof(bool), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(dev_arr);
	return cudaStatus;
}

__global__ void kernel_isPerfectNumber(bool* arr)
{
	unsigned long long i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i >= globalN[0])
	{
		arr[i] = false;
		return;
	}

	if (isPerfectNumber(i))
		arr[i] = true;
	else
		arr[i] = false;
}

__device__ bool isPerfectNumber(unsigned long long number)
{
	unsigned long long i = 1, sum = 0;
	while (i < number)
	{
		if (number % i == 0)
			sum = sum + i;
		i++;
	}

	if (sum == number)
		return true;
	else
		return false;
}