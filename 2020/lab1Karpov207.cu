#include <iostream>  
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_functions.h"  
#include <helper_cuda.h>       
#include <ctime>

#define N (1024)

__global__ void kernel (int* arr){
    
  int   idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  arr[idx] = idx*idx;

}

  int main() 
  
{ 
    int* host_arr = new int[N];
 

    float elapsedTimeInMs = 0.0f;

  
    cudaDeviceReset();

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    int* dev;

    HANDLE_ERROR( cudaMalloc((void**)&dev, N * sizeof (int)));
    kernel <<<dim3((N / 512), 1), dim3(512, 1) >>> (dev);
    HANDLE_ERROR(cudaMemcpy(host_arr, dev, N * sizeof (int), cudaMemcpyDeviceToHost));
    cudaFree(dev);

    
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));

    printf("Length %d",N);

    printf("Time in GPU %f\n", elapsedTimeInMs);
    delete []host_arr;

    float start, time2; 
    start2 = clock();

    int* arr = new int[N];
  
    for(int i = 0; i < N; i++){
        arr[i] = i*i;
    }
  
    time2 = clock() - start2;
    printf("Time in CPU %f\n", time2);    
    delete []arr;
    return 0; 
} 
