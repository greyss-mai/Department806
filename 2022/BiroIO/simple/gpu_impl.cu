#include <cuda.h>
#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


#define THREAD_NUM 1024


__global__ void cuda_sqrt(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = sqrt(data[idx]);
}


int GPU_sqrt(float* array, int N) {
           
    int num_bytes = N * sizeof(float);
    int num_blocks = (N < THREAD_NUM) ? 1 : (N / THREAD_NUM);

    dim3 threads(THREAD_NUM);
    dim3 blocks(num_blocks);

    // Memory allocate
    cudaEvent_t start, stop;
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    cudaEventRecord(start);

    float* in_dev;
    {
        checkCudaErrors(cudaMalloc(&in_dev, num_bytes));
        checkCudaErrors(cudaMemcpy(in_dev, array, num_bytes, cudaMemcpyHostToDevice));
    }

    
    cuda_sqrt<<<blocks, threads>>>(in_dev);
    cudaThreadSynchronize();

    // Memory deallocate
    {
        checkCudaErrors(cudaMemcpy(array, in_dev, num_bytes, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(in_dev));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    int microseconds = static_cast<int>(1000.f * milliseconds);

    return microseconds;

}
