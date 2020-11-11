
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu() {
	printf("Cuda lab1 \n");
}

int main() {
	int n=10;

//GPU  
  cudaEvent_t start, stop;
  float gpuTime =0.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start,0);
  print_from_gpu<<<10,1>>>();
  cudaEventRecord(stop,0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuTime, start, stop);
  printf("time on GPU = %.4f ms\n", gpuTime);

	cudaDeviceSynchronize();
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

//CPU
  clock_t c_start, c_stop;  
  clock_t begin = clock();

	for (int i = 0; i < n; i++)
	{
		printf("Cuda lab1 \n");
	}

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;

	printf("time on CPU = %.4f ms\n", time_spent);


	return 0;
}
