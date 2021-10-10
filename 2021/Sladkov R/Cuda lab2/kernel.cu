
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <omp.h>
#include <stdio.h>

cudaError_t addWithCuda(unsigned int size);

__global__ void addKernel()
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int answ = 1;

    for (int j = 2; j <= i / 2; j++) {
        if (i % j == 0) {
            answ += j;
        }
    }

    if (i == answ && i != 1)
        printf("%d\n", i);
}

int main()
{
    const int arraySize = 100000;
    
    printf("N is %d\n\n", arraySize);

    cudaError_t cudaStatus = addWithCuda(arraySize);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
  
    double start;
    double stop;
    start = omp_get_wtime(); 
    
    for (int i = 2; i < arraySize; i++)
    {
        unsigned int answ = 1;

        for (int j = 2; j <= i / 2; j++) {
            if (i % j == 0) {
                answ += j;
            }
        }

        if (i == answ)
            printf("%d\n", i);
    }

    stop = omp_get_wtime();

    printf("Timing CPU Events %.10f", (stop - start) * 1000);

    return 0;
}

cudaError_t addWithCuda(unsigned int size)
{
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);

    addKernel<<<(size + 1023) / 1024, 1024>>>();

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

    cudaEventRecord(stop, 0);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("Timing CUDA Events %.10f\n\n", gpuTime);
 
Error:
   
    return cudaStatus;
}
