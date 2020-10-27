
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_profiler_api.h>
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono> 
using namespace std;
using namespace std::chrono;

#define testSize 10

cudaError_t multiplyWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void multiplyKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

void printArray(int* x, int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        cout.width(4);
        cout << left << x[i];
    }
    cout << "\n\n";
}

void fillArray(int* x, int arraySize) {
    for (int i = 0; i < arraySize; i++) {
        x[i] = rand() % 10;
    }
}

int main()
{
    //Test iteration
    int* a = new int[testSize];
    int* b = new int[testSize];
    int* c = new int[testSize];

    //Fill arrays with random digits
    srand(time(NULL));
    fillArray(a, testSize);
    fillArray(b, testSize);

    //Output arrays
    cout << "Array A:\n\n";
    printArray(a, testSize);
    cout << "Array B:\n\n";
    printArray(b, testSize);

    // Multiply vectors in parallel
    cudaError_t cudaStatus = multiplyWithCuda(c, a, b, testSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multiplyWithCuda failed!");
        return 1;
    }

    cout << "Array C:\n\n";
    printArray(c, testSize);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    cout << "-----Comparing test for CPU and GPU-----\n\n";

    for (int size = 100; size <= 51200; size *= 2) {
        a = new int[size];
        b = new int[size];
        c = new int[size];

        cout << size << " elements:\t";

        fillArray(a, size);
        fillArray(b, size);

        //---------------GPU----------------//

        // Multiply vectors in parallel
        cudaError_t cudaStatus = multiplyWithCuda(c, a, b, size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "multiplyWithCuda failed!");
            return 1;
        }

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }

        //---------------CPU----------------//

        auto start = high_resolution_clock::now();

        for (int i = 0; i < size; i++)
            c[i] = a[i] * b[i];

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        cout << "CPU: " << duration.count() << " nanoseconds\n";

        delete[] a;
        delete[] b;
        delete[] c;
    }

    //system("pause");
    return 0;
}


cudaError_t multiplyWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    auto start = high_resolution_clock::now();

    multiplyKernel <<<1, size>>> (dev_c, dev_a, dev_b);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);
    
    if (size != testSize)
        cout << "GPU: " << duration.count() << " nanoseconds;\t";

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
