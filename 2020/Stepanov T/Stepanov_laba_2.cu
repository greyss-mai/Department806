%%cu
#include <iostream>
#include <time.h>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define N (512*512)
const int half = N/2;

__global__ void kernel_1 (int * dev_arr){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N){
      dev_arr[idx] = idx;
      }
}

__global__ void kernel_2 (int * dev_arr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int th_idx = threadIdx.x;
    __shared__ int temporary[512];
    if (idx < half){
        temporary[th_idx] = dev_arr[idx];
        int swap_idx= N - idx - 1 ;
        dev_arr[idx] = dev_arr[swap_idx];
        dev_arr[swap_idx] = temporary[th_idx];
    }
}

void fill_array(int * arr){
    for(int i = 0; i<N ; i++){
        arr[i] = i;
    }
}


void reverse_array(int * arr){
    for (int i = 0; i < half; i++){
        int id = N - i - 1;
        int temporary = arr[i];
        arr[i] = arr[id];
        arr[id] = temporary;
    }
}

int main() {
//CPU
    int cpu_arr[N];
    clock_t start_cpu;
    double time_cpu; 
		start_cpu = clock();

    fill_array(cpu_arr);
    reverse_array(cpu_arr);

    time_cpu =(double)(clock() - start_cpu)/CLOCKS_PER_SEC;
    printf("Time in CPU %f\n", time_cpu);

//GPU
    int * host_arr = new int[N];
    int* dev_arr;

    float elapsedTime = 0.0f;
  
    cudaDeviceReset();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    cudaMalloc((void**)&dev_arr, N * sizeof (int));
    kernel_1 <<<dim3(((N + 511) / 512), 1), dim3(512, 1) >>> (dev_arr);
    cudaThreadSynchronize();
    kernel_2 <<<dim3(((N + 511) / 512), 1), dim3(512, 1) >>> (dev_arr);
    cudaThreadSynchronize();
    cudaMemcpy(host_arr, dev_arr, N * sizeof (int), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize(); 
        
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time in GPU %f\n", elapsedTime/1000);

    cudaFree(dev_arr);
    return 0;
}
