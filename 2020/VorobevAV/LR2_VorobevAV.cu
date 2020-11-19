#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>
#include <cmath>

#define N (1024)

__global__ void kernel(float *dev)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (N % idx == 0) {
    dev[idx] = (float) idx;
    }
}

int main (int argc, char *argv[])
{
//------------------— GPU PART —----------------—
    float arr [N];
    float *dev = NULL;

    cudaMalloc(&dev, N * sizeof(float));

    kernel<<<2, 512>>> (dev);

    cudaMemcpy(&arr, dev, N * sizeof(float), cudaMemcpyDeviceToHost);   

    for (int idx = 0; idx < N; idx++) 
    {
        if (arr[idx] != 0) {
            printf("%f ", arr[idx]);
        } 
    }
 
    cudaFree(dev);

    return 0;
}