#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <ctime>
#include <cmath>

#define	N (1024 * 1024)		

__global__ void kernel(float* data)
{
    int   idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 2.0f * 3.1415926f * (float)idx / (float)N;
    data[idx] = sinf(x) / x;
}

int main(int argc, char* argv[])
{
    // GPU 
    float* a = new float[N];
    float* dev = NULL;

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMalloc((void**)&dev, N * sizeof(float));
    kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);
    cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GPU compute TIME: %.2f msec\n\n", gpuTime);
    cudaFree(dev);

    //CPU
    int start2, time2;
    float* data2 = new float[N];

    start2 = clock();

    for (int idx2 = 0; idx2 < N; idx2++)
    {
        float x2 = 2.0f * 3.1415926f * (float)idx2 / (float)N;
        data2[idx2] = sinf(x2);
    }

    time2 = clock() - start2;
    double time_CPU = time2;

    printf("CPU compute TIME:  %.2f msec\n\n", time_CPU);

    //PRINT VALUES
    printf("Values:\n");

    int idx = 0;
    idx = 0;
    printf("\nPoint (zero):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 12;
    printf("\nPoint (Pi/6):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 8;
    printf("\nPoint (Pi/4):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 6;
    printf("\nPoint (Pi/3):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 4;
    printf("\nPoint (Pi/2):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 12 + N / 4;
    printf("\nPoint (2Pi/3):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 8 + N / 4;
    printf("\nPoint (3Pi/4):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 6 + N / 4;
    printf("\nPoint (5Pi/6):\n[%d] = %.3f\n", idx, a[idx]);

    idx = N / 4 + N / 4;
    printf("\nPoint (Pi):\n[%d] = %.3f\n", idx, a[idx]);

    return 0;
}
