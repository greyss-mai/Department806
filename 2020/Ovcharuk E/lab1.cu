

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <iterator>

cudaError_t addWithCuda(int* a, unsigned int size);

__global__ void addKernel(int* a)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] = a[i] * a[i];
}

int main()
{
    const int matrixSize = 512;
    
    int* matrix = (int*)malloc((matrixSize * matrixSize) * sizeof(int));
   
   // printf("Original matrix\n");
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        matrix[i] = rand();
        /*printf("%d ", matrix[i]);
        if ((i + 1) % matrixSize == 0) {
            printf("\n");
        }*/
    }
    //printf("\n");

    int* newMatrix = (int*)malloc((matrixSize * matrixSize) * sizeof(int));
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        newMatrix[i] = matrix[i];
    }

    //printf("CPU calculated\n");
    auto beginCPU = std::chrono::steady_clock::now();
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        newMatrix[i] = newMatrix[i] * newMatrix[i];
        /*printf("%d ", newMatrix[i]);
        if ((i + 1) % matrixSize == 0) {
            printf("\n");
        }*/
    }
    auto endCPU = std::chrono::steady_clock::now();
    //printf("\n");

    auto elapsed_msCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - beginCPU);

    std::cout << "The time CPU: " << elapsed_msCPU.count() << " ns\n";

    auto begin = std::chrono::steady_clock::now();
    // Add vectors in parallel.
    
    cudaError_t cudaStatus = addWithCuda(matrix, matrixSize*matrixSize);
    auto end = std::chrono::steady_clock::now();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    //printf("\n");

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    std::cout << "The time CUDA: " << elapsed_ms.count() << " ns\n";
    //printf("\n");

    /*printf("GPU calculated\n");
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        printf("%d ", matrix[i]);
        if ((i + 1) % matrixSize == 0) {
            printf("\n");
        }
    }*/



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    free(matrix);
    free(newMatrix);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* a, unsigned int size)
{
    int* dev_a = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 threads(256);
    dim3 blocks(size / 256);


    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<blocks, threads>>> (dev_a);

    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

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

Error:
    cudaFree(dev_a);

    return cudaStatus;
}
