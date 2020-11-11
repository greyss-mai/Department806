#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include <iostream>
#include <numeric>
// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
using namespace std;

__global__ void sum(int* input)
{
	const int tid = threadIdx.x;
	auto step_size = 1;
	int number_of_threads = blockDim.x;
	while (number_of_threads > 0)
	{
		if (tid < number_of_threads) 
		{
			const auto fst = tid * step_size * 2;
			const auto snd = fst + step_size;
			input[fst] += input[snd];
		}
		step_size <<= 1; 
		number_of_threads >>= 1;
	}
	  __syncthreads();
}

int main()
{

    tryAgain: // это лейбл
	
    srand(time(NULL));          //зерно рандома
    int i,n;                    //для цикла
	int *h;
	printf("Input array size: ");
    scanf("%d",&n);             //задаем размер
    //int h[n];
	h = (int*)malloc(n * sizeof(int));
    
    for(i=0;i<n;i++)            //запоняем рандомом
	{
		h[i]=rand()%1699999+1699995;
		// cout << " " << h[i] << endl;
	}
        

	const auto count = n;
	const int size = count * sizeof(int);

	int* d;

    auto elapsedTimeInMsGPU = 0.0f;
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

	//start timer
	checkCudaErrors(cudaEventRecord(start, 0));
	sdkStartTimer(&timer);	

	cudaMalloc(&d, size);
	cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);

	sum <<<1, count / 2 >>>(d);

	int result;
	cudaMemcpy(&result, d, sizeof(int), cudaMemcpyDeviceToHost);
	
	//Stop the timer
	checkCudaErrors(cudaEventRecord(stop, 0));
		sdkStopTimer(&timer);
		elapsedTimeInMsCPU = sdkGetTimerValue(&timer);
	

	// make sure GPU has finished copying
	checkCudaErrors(cudaDeviceSynchronize());

	//Finish point to mesure time
	checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMsGPU, start, stop));
	
	printf("Execution time in ms via GPU timer %f\n", elapsedTimeInMsGPU);

	cout << "Sum(GPU) is " << result << endl;

	result = 0;
	for (int i = 0; i < count; i++)
		result= result+h[i];

	printf("Execution time in ms via CPU timer %f\n", elapsedTimeInMsCPU);

	cout << "Sum(CPU) is " << result << endl;

	
	getchar();

	cudaFree(d);
	delete[] h;

	goto tryAgain; // а это оператор goto
	
	return 0;
}