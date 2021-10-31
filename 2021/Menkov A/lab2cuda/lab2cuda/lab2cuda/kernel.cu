
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <chrono> 
using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 128
#define N 16384

//----------CPU functions----------

bool isPrimeCPU(int x) {

    bool flag = true;

    if (x == 0 || x == 1)
        flag = false;

    else
        for (int i = 2; i <= x / 2; i++) {
            if (x % i == 0) {
                flag = false;
                break;
            }
        }

    return flag;
}

int powerCPU(int x, int n) {

    int result = x;

    for (int i = 1; i < n; i++) {
        result *= x;
    }

    return result;
}

bool isMersenneCPU(int x) {

    int p;

    for (int i = 0; ; i++) {
        p = powerCPU(2, i);

        if (p > x + 1)
            return false;

        else if ((p == x + 1) && isPrimeCPU(i))
            return true;
    }
}

//----------GPU functions----------

__device__ bool isPrime(int x) {

    bool flag = true;

    if (x == 0 || x == 1)
        flag = false;

    else
        for (int i = 2; i <= x / 2; i++) {
            if (x % i == 0) {
                flag = false;
                break;
            }
        }

    return flag;
}

__device__ int power(int x, int n) {

    int result = x;

    for (int i = 1; i < n; i++) {
        result *= x;
    }

    return result;
}

__device__ bool isMersenne(int x) {

    int p;

    for (int i = 0; ; i++) {
        p = power(2, i);

        if (p > x + 1)
            return false;

        else if ((p == x + 1) && isPrime(i))
            return true;
    }
}

__global__ void mersenneKernel(bool* a) {

    int i = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    __shared__ bool sharedA[N];

    __syncthreads();

    if (isMersenne(i))
        sharedA[i] = true;

    __syncthreads();
    a[i] = sharedA[i];
}

void showArray(bool* a, int n) {
    for (int i = 0; i < n; i++)
        if (a[i])
            cout << i << " ";
}


int main() {

    bool a[N] = { false };
    cout << N << " elements:\n";

    bool* dev_a = 0;
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(bool));

    auto start = high_resolution_clock::now();
    mersenneKernel << <N / BLOCK_SIZE, BLOCK_SIZE >> > (dev_a);

    cudaStatus = cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(a, dev_a, N * sizeof(bool), cudaMemcpyDeviceToHost);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);

    cudaFree(dev_a);
    cudaStatus = cudaDeviceReset();

    cout << "(GPU) Mersenne's numbers in range [0; " << N << "]: ";
    showArray(a, N);
    cout << "\n(GPU) Elapsed time: " << duration.count() << " nanoseconds\n";

    start = high_resolution_clock::now();

    for (int i = 0; i < N; i++)
        if (isMersenneCPU(i))
            a[i] = true;

    end = high_resolution_clock::now();
    duration = duration_cast<nanoseconds>(end - start);

    cout << "(CPU) Mersenne's numbers in range [0; " << N << "]: ";
    showArray(a, N);
    cout << "\n(CPU) Elapsed time: " << duration.count() << " nanoseconds\n";

    return 0;
}
