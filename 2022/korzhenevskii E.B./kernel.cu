
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef int number;

const int SIZE = 100000;
const int N = 512;

void fillArray(number*& input);
void printArray(number* input);

__global__ void invert(number* input, number* output)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < SIZE)
	{
		output[SIZE - tid - 1] = input[tid];
	}
}
number main(number argc, char** argv)
{

	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	if (deviceCount < 1) {
		printf("No CUDA capable device found.\n");
		return 1;
	}

	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device name: %s\n", deviceProp.name);
		printf("Total global memory: %lu\n", deviceProp.totalGlobalMem);
		printf("Clock rate: %d\n", deviceProp.clockRate);
		printf("Max grid size: x = %d, y = %d, z = %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("Compute capability: %d.%d\n\n", deviceProp.major, deviceProp.minor);
	}
	number* input = (number*)malloc(SIZE * sizeof(number));
	number* output = (number*)malloc(SIZE * sizeof(number));
	fillArray(input);

	number* dev_input = 0;
	number* dev_output = 0;
	cudaMalloc((void**)&dev_input, SIZE * sizeof(number));
	cudaMalloc((void**)&dev_output, SIZE * sizeof(number));

	cudaMemcpy(dev_input, input, SIZE * sizeof(number), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	invert << <(N + 512 - 1), N >> > (dev_input, dev_output);

	cudaMemcpy(output, dev_output, SIZE * sizeof(number), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);

	printArray(output);

	printf("GPU Time: %f ms\n", elapsed);

	cudaFree(dev_input);
	cudaFree(dev_output);
	free(input);
	free(output);

	return 0;
}
void fillArray(number*& input)
{
	for (int i = 0; i < SIZE; ++i)
	{
		*(input + i) = i + 1;
	}
}
void printArray(number* input)
{
	if (SIZE < 10)
	{
		return;
	}

	for (int i = 0; i < 10; ++i)
	{
		printf("%d ", *(input + i));
	}

	printf(" ... ");

	int offset = SIZE - 11;
	for (int i = offset; i < SIZE; ++i)
	{
		printf("%d ", *(input + i));
	}

	printf("\n");
}

