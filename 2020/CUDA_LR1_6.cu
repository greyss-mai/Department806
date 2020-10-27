#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>


#define N 65535


__global__ void add(int *a)
{
	/*int tid = blockIdx.x;
	if (tid < N)
		a[tid] = tid + 1;*/

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		a[tid] = tid + 1;
		tid += blockDim.x * gridDim.x;
	}
}


int main(void)
{
	int a[N];
	int *dev_a;

	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	cudaEvent_t start, stop;
	float gpuTime = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cudaMalloc((void**)&dev_a, N * sizeof(int));

	add<<<blocksPerGrid, threadsPerBlock >>>(dev_a);

	cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaThreadSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);


	for (int i = 0; i < N; ++i)
		printf("%d\t", a[i]);

	printf("\n===================   GPU    ===================\n");
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	printf("DEVICE GPU compute time: %.2f milliseconds\n\n", gpuTime);

	cudaFree(dev_a);


	// CPU
	int start2, time2;
	start2 = clock();

	int a2[N];

	for (int i = 0; i < N; ++i)
		a2[i] = i + 1;
		
	time2 = clock() - start2;

	//for (int i = 0; i < N; ++i)
		//printf("%d\t", a2[i]);

	double time_CPU = time2 / 2.0;

	printf("\n===================   CPU    ===================\n");
	printf("CPU compute time: %f milliseconds\n\n", time_CPU);

	return 0;
}