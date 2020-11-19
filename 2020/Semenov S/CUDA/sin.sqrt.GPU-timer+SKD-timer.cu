#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "math.h"

// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#define	N	(1024*1024)		

__global__ void kernel(float * data)
{
	int   idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = 2.0f * 3.1415926f * (float)idx / (float)N;

	data[idx] = sinf(sqrtf(x));
}

int main(int argc, char *  argv[])
{
	float * a = new float [N];
	
	float elapsedTimeInMsGPU = 0.0f;
	float elapsedTimeInMsCPU = 0.0f;
	StopWatchInterface *timer = NULL;


	//GPU restart
	cudaDeviceReset();

	//Entry point to mesure time
	cudaEvent_t start, stop;
	//GPU timer
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	//SDK timer
	sdkCreateTimer(&timer);

	//Start the timer
	checkCudaErrors(cudaEventRecord(start, 0));
	sdkStartTimer(&timer);

	float * dev = NULL;
	
	cudaMalloc((void**)&dev, N * sizeof (float));

	kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);

	cudaMemcpy(a, dev, N * sizeof (float), cudaMemcpyDeviceToHost);
	cudaFree(dev);

	//Stop the timer
	checkCudaErrors(cudaEventRecord(stop, 0));
	sdkStopTimer(&timer);

	// make sure GPU has finished copying
	checkCudaErrors(cudaDeviceSynchronize());
	
	//Finish point to mesure time
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMsGPU, start, stop));
	elapsedTimeInMsCPU = sdkGetTimerValue(&timer);

	printf("Execution time in ms via CPU timer %f\n", elapsedTimeInMsCPU);
	printf("Execution time in ms via GPU timer %f\n", elapsedTimeInMsGPU);
	//for (int idx = 0; idx < N; idx++)  printf("a[%d] = %.5f\n", idx, a[idx]);

	return 0;
}
