//A simple hello world program
#include <stdio.h>

#include "cuda_profiler_api.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void __global__ mykernel() {
	__shared__ int a;

	a = 10;

	int b[16], c[16], d[16] ;
	b[threadIdx.x] = threadIdx.x;
	c[threadIdx.x] = 2*threadIdx.x;
	d[threadIdx.x] = b[threadIdx.x] + c[threadIdx.x];
}

int main(){
	mykernel <<<4, 16>>> ();
    printf("Hello Wolrd!");

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
    return 0;
}
