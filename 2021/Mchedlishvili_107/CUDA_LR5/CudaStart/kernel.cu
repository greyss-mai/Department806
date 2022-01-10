#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <iostream>

#define THREADS 512
#define BLOCKS 1
#define NUM_VALS THREADS*BLOCKS

void array_fill(float* arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) 
    {
        arr[i] = (float)rand() / (float)RAND_MAX;
    }
}

__global__ void bitonic_sort_step(float* dev_values, int j, int k)
{
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ixj = i ^ j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) 
    {
        if ((i & k) == 0) //if ascends
        {
            if (dev_values[i] > dev_values[ixj]) 
            {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        else //if descends
        {
            if (dev_values[i] < dev_values[ixj]) 
            {
                float temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

int main(void)
{
    clock_t start, stop;

    float* values = (float*)malloc(NUM_VALS * sizeof(float));
    array_fill(values, NUM_VALS);

    start = clock();

    float* dev_values;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    int j, k;
    /* Major step */
    for (k = 2; k <= NUM_VALS; k <<= 1) 
    {
        /* Minor step */
        for (j = k >> 1; j > 0; j = j >> 1) 
        {
            bitonic_sort_step <<<blocks, threads>>>(dev_values, j, k);
        }
    }
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);

    stop = clock();

    for (int i = 0; i < NUM_VALS; i++) 
    {
        std::cout << values[i] << std::endl;
    }

    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
}