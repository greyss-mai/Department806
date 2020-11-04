
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
using namespace std;

#define N 20

cudaError_t sqrtWithCuda(float* res, const float *a, unsigned int size);

__global__ void addKernel(float* res, float *a)
{
    int idx = threadIdx.x;
    res[idx] = sqrt(a[idx]);
}


void printArr(float* a, int size)
{
	for (int i = 0; i < size; i++)
	{
		cout.width(4);
        cout << a[i] << " ";
    }
    cout << "\n\n";
}


void createArr(float* a, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] = rand();
    }
}


int main()
{
    int count;
    float *a = new float [N];
    float* res = new float[N];
    cudaGetDeviceCount(&count);

    createArr(a, N);
    cout << "New Array: ";
    printArr(a, N);

    cudaError_t cudaStatus = sqrtWithCuda(res, a, N);
	
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sqrtWithCuda failed!");
        return 1;
    }

    cout << "Result: ";
    printArr(res, N);
	
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	// Installed devices
    for (int i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("  Device number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device total global memory: %zd\n", prop.totalGlobalMem);
        printf("  Device total constant memory: %zd\n", prop.totalConstMem);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
    return 0;
}


cudaError_t sqrtWithCuda(float* res, const float* a, unsigned int size)
{
    float* dev_a = 0;
    float* dev_res = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_res, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);

    addKernel << <1, size >> > (dev_res, dev_a);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(res, dev_res, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_res);
    cudaFree(dev_a);

    return cudaStatus;
}
