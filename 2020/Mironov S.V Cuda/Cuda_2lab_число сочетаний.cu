#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <chrono>

cudaError_t addWithCuda(int *inter,double *f, int size);


__global__ void addKernel(int *inter, double *fact)
{

	__shared__ double fact_gpu[4*3];
	int i = threadIdx.x;
	int num = int(i/4);
  	int i1 = i%4;
  	fact_gpu[i]=1;

  	for(int j=inter[num*5+i1]; j<inter[num*5+i1+1]; j++){
      		fact_gpu[i]*=j;
  	}
	__syncthreads();
  	fact[i]=fact_gpu[i];
}

void calculate_gpu(int n, int k, int arraySize) {
	
  	int intervals[5*3]; //5 интервала для n, 5 для k, 5 для n-k
  	double facts[(5-1)*3];

  	intervals[0] = 2; //начало
  	intervals[2] = int((2 + n) / 2); //середина
  	intervals[1] = int((2 + intervals[2]) / 2);
  	intervals[3] = int((intervals[2] + n) / 2);
  	intervals[4] = n+1;//конец
  
  	intervals[0+arraySize] = 2; //начало
  	intervals[2+arraySize] = int((2 + k) / 2); //середина
  	intervals[1+arraySize] = int((2 + intervals[2+arraySize]) / 2);
  	intervals[3+arraySize] = int((intervals[2+arraySize] + k) / 2);
  	intervals[4+arraySize] = k+1;//конец
 
  	intervals[0+2*arraySize] = 2; //начало
  	intervals[2+2*arraySize] = int((2 + n-k) / 2); //середина
  	intervals[1+2*arraySize] = int((2 + intervals[2+2*arraySize]) / 2);
  	intervals[3+2*arraySize] = int((intervals[2+2*arraySize] + n-k) / 2);
  	intervals[4+2*arraySize] = n-k+1;//конец


  	auto begin = std::chrono::steady_clock::now();
	cudaError_t cudaStatus = addWithCuda(intervals, facts, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");

	}
  	auto end = std::chrono::steady_clock::now();
  	auto elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	printf("Time= %i nanoseconds \n", elapsed_ms.count());

  	double nf = facts[0]*facts[1]*facts[2]*facts[3];
  	double kf = facts[0+4]*facts[1+4]*facts[2+4]*facts[3+4];
  	double nkf= facts[0+8]*facts[1+8]*facts[2+8]*facts[3+8];
  	double ans = nf/(kf*nkf);
  	printf("With GPU C (n=%i, k=%i) = %g \n", n, k, ans);
}

double fact_cpu(int n) {
	double f=1;

	for (int i = 2; i <= n; i++) {
		f *= (double)i;
    
	}
	return f;
}


void calculate_cpu(int n, int k) {
	auto begin = std::chrono::steady_clock::now();
  	double ans=fact_cpu(n)/(fact_cpu(k)*fact_cpu(n-k));
	auto end = std::chrono::steady_clock::now();

  	auto elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
	printf("Time= %i nanoseconds \n", elapsed_ms.count());
	printf("With CPU C (n=%i, k=%i) = %g \n", n, k, ans);
	
}

int main()
{
	cudaError_t cudaStatus;
  	const int arraySize=5;
  	int n = 12;
  	int k = 4;

	for (int j = 0; j < 7; j++) {
    		calculate_cpu(n,k);
		printf("\n");
		calculate_gpu(n, k, arraySize);
	  	printf("\n");

    		n*=2;
    		k*=2;
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
cudaError_t addWithCuda(int *inter,double *f, int size)
{

	int *dev_inter = 0;
	double *dev_fact = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_inter, size* 3 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_fact, (size-1)*3*sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_inter, inter, size*3*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
 

	cudaStatus = cudaMemcpy(dev_fact, f, (size-1)* 3 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
 

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, (size-1)*3 >>> (dev_inter, dev_fact);
 


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
	cudaStatus = cudaMemcpy(f, dev_fact, (size-1)* 3 * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_inter);
	cudaFree(dev_fact);

	return cudaStatus;
}