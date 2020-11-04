
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>


__global__ void print_from_gpu() {
	printf("Cuda lab1 \n");
}

int main() {
	print_from_gpu<<<2,5>>>();
	cudaDeviceSynchronize();
	return 0;
}
