
#include <iostream>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define N (100000)
const int threadsPerBlock = 512;
const int threads_qty = N/2;



__global__ void fill (int * dev_arr){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < N){
      dev_arr[idx] = idx;
      }
}



__global__ void swap (int * dev_arr){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int th_idx = threadIdx.x;
    __shared__ int cache[threadsPerBlock];
    if (idx < threads_qty){
        cache[th_idx] = dev_arr[idx];
        int swap_idx= N - idx - 1 ;
        dev_arr[idx] = dev_arr[swap_idx];
        dev_arr[swap_idx] = cache[th_idx];
    }
    
}


void fill_array(int * arr){
    for(int i = 0; i<N ; i++){
        arr[i] = i;
    }
}
void swap_array(int * arr){
    for (int i = 0; i < threads_qty; i++){
        int id = N - i - 1;
        int cache = arr[i];
        arr[i] = arr[id];
        arr[id] = cache;
    }
}


int main() {
//GPU
    int * host_arr = new int[N];
    int* dev_arr;

    float elapsedTimeInMs = 0.0f;

  
    cudaDeviceReset();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    cudaMalloc((void**)&dev_arr, N * sizeof (int));
    fill <<<dim3(((N + 511) / 512), 1), dim3(threadsPerBlock, 1) >>> (dev_arr);
    cudaThreadSynchronize();
    swap <<<dim3(((N + 511) / 512), 1), dim3(threadsPerBlock, 1) >>> (dev_arr);
    cudaThreadSynchronize();
    cudaMemcpy(host_arr, dev_arr, N * sizeof (int), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize(); 

        
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);
    printf("Time in GPU %f\n", elapsedTimeInMs/1000);



//    for (int i = 0; i < N; i++){
//        printf("from GPU %d\n", cpu_arr[i]);
//    }

    cudaFree(dev_arr);
    delete []host_arr;

//CPU
    int cpu_arr[N];
    clock_t start2;
    double time2; 
		start2 = clock();

    fill_array(cpu_arr);
    swap_array(cpu_arr);

//    for (int i = 0; i < N; i++){
//        printf("from CPU %d\n", cpu_arr[i]);
//    }

    time2 =(double)(clock() - start2)/CLOCKS_PER_SEC;
    printf("Time in CPU %f\n", time2);




    return 0;
}